import torch
from torch import nn, Tensor, BoolTensor
from typing import Dict, Optional, Tuple

from cosyaccent.model.rope import RoPESelfAttention, RoPECrossAttention
from cosyaccent.model.rope import compute_r
from cosyaccent.model.dit.layers import LayerNorm, FeedForwardModule, modulate


class DiTDecoderLayer(nn.Module):
    def __init__(self, D: int, D_hidden: int, N_head: int, P_dropout: float):
        super().__init__()
        # Self-attention components
        self.self_attn_norm = LayerNorm(D)
        self.self_attn = RoPESelfAttention(D, N_head, P_dropout=P_dropout)

        # Cross-attention components
        self.cross_attn_norm = LayerNorm(D)
        self.cross_attn = RoPECrossAttention(D, N_head, P_dropout=P_dropout)

        # Feed-forward components
        self.ffn_norm = LayerNorm(D)
        self.ffn = FeedForwardModule(D, D_hidden, P_dropout, bias=True)

        # AdaLN modulation (9 parameters: 3 modules × 3 parameters each)
        self.AdaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(D, 9 * D, bias=True))

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        r_dec: Tensor,
        self_mask: BoolTensor,
        enc_out: Tensor,
        r_enc: Tensor,
        cross_mask: BoolTensor,
        kv_cache: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict]:
        """
        Args:
            x: Decoder sequence [N, ..., T_dec, D]
            c: Conditioning vector [N, ..., D]
            r_dec: Rotation for decoder positions [N, ..., T_dec, C//2]
            self_mask: Self-attention mask [N, T_dec, T_dec]
            enc_out: Encoder output [N, ..., T_enc, D]
            r_enc: Rotation for encoder positions [N, ..., T_enc, C//2]
            cross_mask: Cross-attention mask [N, T_dec, T_enc]
            kv_cache: Optional cache for key/values in self- and cross-attention

        Returns:
            y: Output tensor [N, ..., T_dec, D]
        """
        # Generate all modulation parameters from conditioning
        (
            shift_self_attn,
            scale_self_attn,
            gate_self_attn,
            shift_cross_attn,
            scale_cross_attn,
            gate_cross_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.AdaLN_modulation(c).chunk(9, dim=-1)

        # Handlo KV cache
        if kv_cache is not None:
            self_k_cache, self_v_cache = kv_cache["self_k"], kv_cache["self_v"]
            cross_k_cache, cross_v_cache = kv_cache["cross_k"], kv_cache["cross_v"]
        else:
            self_k_cache, self_v_cache = None, None
            cross_k_cache, cross_v_cache = None, None

        # 1. Self-attention block
        x1 = self.self_attn_norm(x)
        x2 = modulate(x1, shift_self_attn, scale_self_attn)
        x3, self_k_cache, self_v_cache = self.self_attn(x2, r_dec, self_mask, self_k_cache, self_v_cache)
        x4 = gate_self_attn.unsqueeze(-2) * x3
        x = x + x4

        # 2. Cross-attention block
        x5 = self.cross_attn_norm(x)
        x6 = modulate(x5, shift_cross_attn, scale_cross_attn)
        x7, cross_k_cache, cross_v_cache = self.cross_attn(
            x6, enc_out, r_dec, r_enc, cross_mask, cross_k_cache, cross_v_cache
        )
        x8 = gate_cross_attn.unsqueeze(-2) * x7
        x = x + x8

        # 3. Feed-forward block
        x9 = self.ffn_norm(x)
        x10 = modulate(x9, shift_mlp, scale_mlp)
        x11 = self.ffn(x10)
        x12 = gate_mlp.unsqueeze(-2) * x11
        x = x + x12

        kv_cache = {
            "self_k": self_k_cache,
            "self_v": self_v_cache,
            "cross_k": cross_k_cache,
            "cross_v": cross_v_cache,
        }

        return x, kv_cache

    def get_self_attn(self, x: Tensor, c: Tensor, r: Tensor, mask: BoolTensor) -> Tensor:
        """Get self-attention weights for visualization/analysis"""
        state = self.training
        with torch.no_grad():
            self.eval()
            shift_self_attn, scale_self_attn, *_ = self.AdaLN_modulation(c).chunk(9, dim=-1)
            x1 = self.self_attn_norm(x)
            x2 = modulate(x1, shift_self_attn, scale_self_attn)
            attn = self.self_attn.get_attn(x2, r, mask)
        self.train(state)
        return attn

    def get_cross_attn(
        self,
        x: Tensor,
        enc_out: Tensor,
        c: Tensor,
        r_dec: Tensor,
        r_enc: Tensor,
        mask: BoolTensor,
    ) -> Tensor:
        """Get cross-attention weights for visualization/analysis"""
        state = self.training
        with torch.no_grad():
            self.eval()
            # Get modulation parameters (skip first 3 which are for self-attention)
            _, _, _, shift_cross_attn, scale_cross_attn, *_ = self.AdaLN_modulation(c).chunk(9, dim=-1)

            # Run through the first part of cross-attention
            x1 = self.self_attn_norm(x)
            x2 = modulate(x1, shift_cross_attn, scale_cross_attn)

            # Get attention weights
            attn = self.cross_attn.get_attn(x2, enc_out, r_dec, r_enc, mask)
        self.train(state)
        return attn


class DiTDecoder(nn.Module):
    def __init__(
        self,
        D: int,
        D_hidden: int,
        N_head: int,
        N_layer: int,
        P_dropout: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([DiTDecoderLayer(D, D_hidden, N_head, P_dropout) for _ in range(N_layer)])
        self.C = D // N_head

    def forward(
        self,
        x: Tensor,
        p: Tensor,
        t_emb: Tensor,
        self_mask: BoolTensor,
        memory: Tensor,
        memory_p: Tensor,
        cross_mask: BoolTensor,
    ) -> Tensor:
        """
        Args:
            x (Tensor): [N, ..., T_dec, D], decoder input sequence.
            p (Tensor): [N, ..., T_dec], the decoder position tensor.
            t_emb (Tensor): [N, ...], the time embedding tensor.
            self_mask (BoolTensor): [N, T_dec, T_dec], self-attention mask.
            memory (Tensor): [N, ..., T_enc, D], encoder output sequence.
            memory_p (Tensor): [N, ..., T_enc], the encoder position tensor.
            cross_mask (BoolTensor): [N, T_dec, T_enc], cross-attention mask.
        Returns:
            y (Tensor): [N, ..., T_dec, D].
        """
        r_dec = compute_r(p, self.C)
        r_enc = compute_r(memory_p, self.C)

        for block in self.blocks:
            x, _ = block.forward(x, t_emb, r_dec, self_mask, memory, r_enc, cross_mask)

        return x

    def get_self_attn(
        self,
        x: Tensor,
        p: Tensor,
        t_emb: Tensor,
        self_mask: BoolTensor,
        memory: Tensor,
        memory_p: Tensor,
        cross_mask: BoolTensor,
    ) -> Tensor:
        """Compute self-attention matrices.

        Returns:
            attn (Tensor): [N, ..., N_layer, H, T_dec, T_dec], self-attention weights.
        """
        state = self.training
        attns = []
        with torch.inference_mode():
            self.eval()
            r_dec = compute_r(p, self.C)
            r_enc = compute_r(memory_p, self.C)

            for block in self.blocks:
                attn = block.get_self_attn(x, t_emb, r_dec, self_mask)
                x, _ = block.forward(x, t_emb, r_dec, self_mask, memory, r_enc, cross_mask)
                attns.append(attn)

        self.train(state)
        return torch.stack(attns, dim=-4)
