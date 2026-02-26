import math
import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16_000
CHUNK_SAMPLES = SAMPLING_RATE * 30  # 30-second chunks
FRAMES_PER_SECOND = 50  # Whisper encoder outputs 50 frames per second


class WhisperFrontend(nn.Module):
    """Online batch-wise Whisper encoder feature extraction module.

    Extracts Whisper encoder features from raw audio waveforms in a batch-wise
    manner. Supports audio longer than 30 seconds by processing in 30-second
    chunks and concatenating the results.

    Args:
        whisper_size: Size of the Whisper model, e.g. "tiny", "base", "small",
            "medium", "large", "large-v2", "large-v3".
        freeze: Whether to freeze the Whisper encoder parameters.
            Default: True.
    """

    def __init__(self, whisper_size: str = "medium"):
        super().__init__()
        model = whisper.load_model(whisper_size, device="cpu")
        self.encoder = model.encoder
        self._output_dim = self.encoder.ln_post.normalized_shape[0]

        for param in self.encoder.parameters():
            param.requires_grad = False

    def output_size(self) -> int:
        return self._output_dim

    @torch.inference_mode()
    def forward(
        self,
        waveforms: torch.Tensor,
        wav_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract Whisper encoder features from a batch of waveforms.

        Args:
            waveforms: (B, T_wav) raw audio waveforms at 16 kHz.
            wav_lens: (B,) actual lengths (in samples) of each waveform.

        Returns:
            feats: (B, T_feat, D) padded encoder features.
            feat_lens: (B,) actual feature lengths for each sample.
        """
        B, T_wav = waveforms.shape

        # Clamp wav_lens to actual waveform length for safety
        wav_lens = wav_lens.clamp(max=T_wav)

        # Compute the number of feature frames for each sample
        feat_lens = torch.ceil(wav_lens.float() / SAMPLING_RATE * FRAMES_PER_SECOND).long()

        # Number of 30-second chunks needed to cover the longest sample
        n_chunks = math.ceil(T_wav / CHUNK_SAMPLES)
        max_feat_len = feat_lens.max().item()

        # Pre-allocate output tensor
        feats = waveforms.new_zeros(B, max_feat_len, self._output_dim)

        for chunk_idx in range(n_chunks):
            start = chunk_idx * CHUNK_SAMPLES
            end = min(start + CHUNK_SAMPLES, T_wav)
            chunk = waveforms[:, start:end]  # (B, chunk_len)

            # Per-sample lengths within this chunk
            chunk_sample_lens = (wav_lens - start).clamp(min=0, max=end - start)

            # Skip samples that have no audio in this chunk
            active_mask = chunk_sample_lens > 0
            if not active_mask.any():
                break

            # Pad to exactly 30 seconds as required by Whisper
            if chunk.shape[1] < CHUNK_SAMPLES:
                chunk = F.pad(chunk, (0, CHUNK_SAMPLES - chunk.shape[1]))

            # Compute log-mel spectrogram for the batch
            mel = whisper.log_mel_spectrogram(chunk)  # (B, n_mels, 3000)

            # Run through Whisper encoder
            enc_out = self.encoder(mel)  # (B, 1500, D)

            # Compute frame counts for this chunk per sample
            chunk_feat_lens = torch.ceil(chunk_sample_lens.float() / SAMPLING_RATE * FRAMES_PER_SECOND).long()

            # Write valid frames into the output feature tensor
            feat_offset = chunk_idx * (CHUNK_SAMPLES // SAMPLING_RATE * FRAMES_PER_SECOND)
            max_enc_frames = enc_out.shape[1]
            for i in range(B):
                # Clamp to both encoder output length and remaining pre-allocated space to
                # guard against float32 rounding making chunk_feat_lens[i] > feat_lens[i] - feat_offset
                n_frames = min(chunk_feat_lens[i].item(), max_enc_frames, max_feat_len - feat_offset)
                if n_frames > 0:
                    feats[i, feat_offset : feat_offset + n_frames] = enc_out[i, :n_frames]

        return feats, feat_lens
