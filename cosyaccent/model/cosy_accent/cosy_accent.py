import logging
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from cosyaccent.utils.mask import make_pad_mask

logger = logging.getLogger(__name__)


class CosyAccent(torch.nn.Module):
    def __init__(
        self,
        text_vocab_size: int,
        spk_embed_dim: int,
        frontend: nn.Module,
        speech_encoder: nn.Module,
        speech_decoder: nn.Module,
        duration_predictor: Optional[nn.Module] = None,
        normalize_spk_embed: bool = False,
        ctc_loss_weight: float = 1.0,
        pretraining: bool = False,
    ):
        super(CosyAccent, self).__init__()
        self.spk_embed_dim = spk_embed_dim
        self.normalize_spk_embed = normalize_spk_embed

        if frontend is not None:
            self.frontend = frontend
        else:
            self.frontend = torch.nn.Module()

        self.speech_encoder = speech_encoder
        self.speech_encoder_affine_layer = nn.Linear(
            speech_encoder.output_size(), speech_encoder.output_size()
        )
        # NOTE: We change <blank> token from the last token to the first token for better compatibility
        self.speech_encoder_ctc_layer = nn.Linear(
            speech_encoder.output_size(), text_vocab_size
        )
        self.ctc_blank_idx = 0  # The tokenizer will use 0 as the <blank> token for CTC
        self.speech_decoder = speech_decoder

        self.duration_predictor = duration_predictor

        self.ctc_loss_weight = ctc_loss_weight
        self.pretraining = pretraining

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            batch (dict): batch data:
                - text_token (LongTensor): [B, T_text], text token ids.
                - text_token_len (LongTensor): [B], text token lengths.
                - duration (LongTensor): [B, T_text], duration values of text tokens.
                - speech_feat (FloatTensor): [B, T_feat, D_feat, speech features.
                - speech_feat_len (LongTensor): [B], speech feature lengths.
                - spk_embedding (FloatTensor): [B, D_spk], speaker embedding.
            device (torch.device): device to use.
        Returns:
            dict: output dict.
        """
        text_token = batch["phone_token"].to(device)
        text_token_len = batch["phone_token_len"].to(device)
        speech_feat = batch["speech_feat"].to(device)
        speech_feat_len = batch["speech_feat_len"].to(device)

        if batch["src_feat"] is not None:
            src_feat = batch["src_feat"].to(device)
            src_feat_len = batch["src_feat_len"].to(device)
        else:
            src_wav = batch["speech"].to(device)
            src_wav_len = batch["speech_len"].to(device)
            src_feat, src_feat_len = self.frontend(src_wav, src_wav_len)

        spk_embed = batch["spk_embedding"].to(device)

        # Text encoding
        B, T_text = text_token.size()

        if self.normalize_spk_embed:
            spk_embed = F.normalize(spk_embed, p=2, dim=-1)

        src_cond, ctc_logits, cond_mask = self.encode_speech(src_feat, src_feat_len)
        ctc_loss = (
            self.compute_ctc_loss(ctc_logits, src_feat_len, text_token, text_token_len)
            if self.ctc_loss_weight > 0
            else torch.tensor(0.0, device=device)
        )
        cond_len = src_feat_len
        cond = src_cond

        if self.duration_predictor is not None:
            rel_dur = speech_feat_len / src_feat_len  # (B)
            # NOTE: use detach() to avoid the gradient backpropagation to the encoder
            dp_loss = self.duration_predictor.compute_loss(
                src_cond.detach(), cond_mask, rel_dur, spk_embed
            )
        else:
            dp_loss = torch.tensor(0.0, device=device)

        # Speech decoding
        speech_p_scale, cond_p_scale = self.compute_p_scale(cond_len, speech_feat_len)
        speech_feat_mask = ~make_pad_mask(speech_feat_len, speech_feat.size(1))
        fm_loss = self.speech_decoder.compute_loss(
            speech_feat,
            speech_p_scale,
            speech_feat_mask,
            cond,
            cond_p_scale,
            cond_mask,
            spk_embed,
        )

        loss = fm_loss + ctc_loss * self.ctc_loss_weight + dp_loss
        loss_dict = {
            "loss": loss,
            "fm_loss": fm_loss.detach(),
            "ctc_loss": ctc_loss.detach(),
            "dp_loss": dp_loss.detach(),
            "batch_size": torch.tensor(B, device=device),
        }

        return loss_dict

    @torch.inference_mode()
    def inference(
        self,
        src_feat_or_wav,
        src_feat_or_wav_len,
        spk_embed,
        n_timesteps=32,
        full_cfg=1.0,
        cond_cfg=1.0,
        spk_cfg=0.0,
        use_predicted_duration=True,
        speech_len_ratio=1.0,
        **kwargs,
    ):
        """
        Args:
            src_feat (FloatTensor): [B, T_src, D_src], source speech features.
            src_len (LongTensor): [B], source speech feature lengths.
            spk_embedding (FloatTensor): [B, D_spk], speaker embedding.
            n_timesteps (int): number of diffusion steps.
            full_cfg (float): classifier-free guidance scale for null (condition+speaker) branch.
            cond_cfg (float): classifier-free guidance scale for condition branch.
            spk_cfg (float): classifier-free guidance scale for speaker branch.
            use_predicted_duration (bool): whether to use duration predictor to predict the duration.
            speech_len_ratio (float): if not using duration predictor, use this ratio to determine the speech length.
        Returns:
            speech_feat (FloatTensor): [B, T_feat, D_feat], speech features.
            speech_feat_len (LongTensor): [B], speech feature lengths.
        """
        if self.normalize_spk_embed:
            spk_embed = F.normalize(spk_embed, p=2, dim=-1)

        if src_feat_or_wav.ndim == 2:
            src_feat, src_feat_len = self.frontend(src_feat_or_wav, src_feat_or_wav_len)
        else:
            assert (
                src_feat_or_wav.ndim == 3
            ), "src_feat_or_wav should be either [B, T] or [B, T, D]"
            src_feat = src_feat_or_wav
            src_feat_len = src_feat_or_wav_len

        B, T, D = src_feat.size()
        cond, ctc_logits, cond_mask = self.encode_speech(src_feat, src_feat_len)

        if use_predicted_duration:
            assert self.duration_predictor is not None, "duration predictor is None!"
            pred_dur_ratio = self.duration_predictor(cond, cond_mask, spk_embed)
            speech_feat_len = (pred_dur_ratio * src_feat_len).round().long()
        else:
            speech_feat_len = (speech_len_ratio * src_feat_len).round().long()

        speech_feat_mask = ~make_pad_mask(speech_feat_len)
        cond_mask = ~make_pad_mask(src_feat_len, T)

        speech_p_scale, cond_p_scale = self.compute_p_scale(
            src_feat_len, speech_feat_len
        )
        speech_feat = self.speech_decoder.inference(
            speech_p_scale,
            speech_feat_mask,
            cond,
            cond_p_scale,
            cond_mask,
            spk_embed,
            t_scheduler="cosine",
            n_timesteps=n_timesteps,
            full_cfg=full_cfg,
            cond_cfg=cond_cfg,
            spk_cfg=spk_cfg,
        )

        return speech_feat, speech_feat_len

    def encode_speech(
        self,
        speech_feat: torch.Tensor,
        speech_feat_lengths: torch.Tensor,
        decoding_chunk_size: int = 50,
        num_decoding_left_chunks: int = 30,
    ):
        if self.training:
            # NOTE: this will disable `num_decoding_left_chunks` in training
            # If encoder's `use_dynamic_chunk` is True, it will use dynamic chunk size
            # If not, it will use the preset `static_chunk_size` in the encoder
            # If none of the above, it will use full context
            decoding_chunk_size = 0

        encoder_out, encoder_mask = self.speech_encoder(
            speech_feat,
            speech_feat_lengths,
            decoding_chunk_size=decoding_chunk_size,
            num_decoding_left_chunks=num_decoding_left_chunks,
        )

        encoder_out = self.speech_encoder_affine_layer(encoder_out)
        ctc_logits = self.speech_encoder_ctc_layer(encoder_out)

        encoder_out = encoder_out.detach() if self.pretraining else encoder_out

        return encoder_out, ctc_logits, encoder_mask.squeeze(1)

    def compute_ctc_loss(self, ctc_logits, ctc_len, text_token, text_token_len):
        """
        Computes CTC loss for the speech encoder output.

        Args:
            ctc_logits (torch.Tensor): [B, T, text_vocab_size + 1], CTC logits from the speech encoder.
            ctc_len (torch.Tensor): [B], lengths of CTC logits.
            text_token (torch.Tensor): [B, T_text], text token ids.
            text_token_len (torch.Tensor): [B], lengths of text tokens.

        Returns:
            torch.Tensor: CTC loss value.
        """
        ctc_logp = ctc_logits.log_softmax(dim=-1)  # [B, T, V_ctc]
        ctc_loss = F.ctc_loss(
            ctc_logp.transpose(0, 1),  # [T, B, V_ctc]
            text_token,  # [B, N_text]
            ctc_len,  # [B]
            text_token_len,  # [B]
            blank=self.ctc_blank_idx,
            reduction="mean",
            zero_infinity=True,
        )
        return ctc_loss

    def compute_p_scale(self, text_cond_len, speech_feat_len):
        """
        Computes position scaling factors to align linguistic conditioning and speech feature sequences.

        This function calculates scaling factors based on the ending positions ratio, ensuring
        that the positional embeddings of the two sequences are properly aligned in RoPE space.

        Args:
            text_cond_len (LongTensor): [B], lengths of text conditioning sequences.
            speech_feat_len (LongTensor): [B], lengths of speech feature sequences.

        Returns:
            speech_p_scale (FloatTensor): [B], scaling factors for speech positions, all ones since speech
                positions are kept as the reference frame.
            cond_p_scale (FloatTensor): [B], scaling factors for text conditioning positions, calculated as
                the ratio of (speech_end_position / text_end_position) to align endpoints.
        """
        cond_end_p = text_cond_len.float() - 1  # [B]
        speech_end_p = speech_feat_len.float() - 1  # [B]
        length_ratio = torch.clamp(speech_end_p / cond_end_p, min=0.0)  # [B]

        speech_p_scale = torch.ones_like(speech_feat_len).float()  # [B]
        cond_p_scale = length_ratio  # [B]

        return speech_p_scale, cond_p_scale

    def state_dict(self, *args, **kwargs):
        # Exclude Whisper frontend from state dict
        state_dict = super().state_dict(*args, **kwargs)
        frontend_keys = [
            key for key in state_dict.keys() if key.startswith("frontend.")
        ]
        for key in frontend_keys:
            del state_dict[key]
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        # Load state dict while keeping frontend
        frontend_state_dict = {
            f"frontend.{k}": v for k, v in self.frontend.state_dict().items()
        }
        state_dict.update(frontend_state_dict)
        super().load_state_dict(state_dict, *args, **kwargs)
