import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import repeat

from cosyaccent.model.conv.conv_layers import ResBlock1d
from cosyaccent.model.dit.layers import ScalarEmbedder, FinalLinear
from cosyaccent.model.dit.dit_encoder import DiTEncoder


class AttentivePooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.linear_kv = nn.Linear(input_dim, hidden_dim * 2)
        self.q = nn.Parameter(torch.randn(hidden_dim))
        self.dropout_rate = dropout_rate

    def forward(self, x, x_mask):
        """
        Args:
            x (torch.tensor): input tensor.
                shape: (B, T, D)
            x_mask (torch.tensor): mask tensor. True for valid positions.
                shape: (B, T)
        Returns:
            torch.tensor: pooled tensor.
                shape: (B, D)
        """
        k, v = self.linear_kv(x).chunk(2, dim=-1)  # (B, T, D), (B, T, D)
        o = torch.nn.functional.scaled_dot_product_attention(
            self.q.unsqueeze(0).unsqueeze(1),
            k,
            v,
            attn_mask=x_mask.unsqueeze(1),
            dropout_p=self.dropout_rate,
            is_causal=False,
        )  # (B, 1, D)
        o = o.squeeze(1)  # (B, D)
        return o


class FlowMatchingTotalDurationPredictor(nn.Module):
    """
    Flow Matching-based Total Duration Predictor for speech synthesis.

    The architecture consists of:
    - A preprocessing network (prenet) for input conditioning
    - A time embedding encoder for flow matching timesteps
    - A DiT (Diffusion Transformer) encoder as the main processing unit
    - Attention pooling to aggregate sequence-level features to utterance level
    - A final linear layer to predict velocity in the flow field

    Args:
        input_dim (int): Dimension of input text representations
        embed_dim (int): Hidden embedding dimension
        num_heads (int): Number of attention heads in DiT encoder
        num_layers (int): Number of layers in DiT encoder
        global_cond_dim (int): Dimension of global conditioning (e.g., speaker embedding)
        cfg_rate (float, optional): Classifier-free guidance rate during training. Defaults to 0.2
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1
        log_scale (bool, optional): Whether to work in log domain. Defaults to True
    """

    def __init__(
        self,
        input_dim,
        embed_dim,
        num_heads,
        num_layers,
        global_cond_dim,
        cfg_rate=0.2,
        dropout_rate=0.1,
        log_scale=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.global_cond_dim = global_cond_dim
        self.dropout_rate = dropout_rate
        self.log_scale = log_scale
        self.cfg_rate = cfg_rate

        # Flow matching components
        self.prenet = ResBlock1d(input_dim + 1 + global_cond_dim, embed_dim, embed_dim)
        self.t_encoder = ScalarEmbedder(256, embed_dim)
        self.encoder = DiTEncoder(
            D=embed_dim,
            D_hidden=embed_dim,
            N_head=num_heads,
            N_layer=num_layers,
            P_dropout=dropout_rate,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pooling = AttentivePooling(embed_dim, embed_dim)
        self.final_linear = FinalLinear(embed_dim, 1)

    def compute_loss(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        lin_d: torch.Tensor,
        c: torch.Tensor = None,
        sigma_min: float = 1e-6,
    ):
        """
        Args:
            x (torch.tensor): batch of text representations.
                shape: (B, T, D)
            x_mask (torch.tensor): batch of sequence mask.
                shape: (B, T)
            lin_d (torch.tensor): batch of ground-truth total duration values.
                shape: (B,)
            c (torch.tensor): batch of global condition representations.
                shape: (B, D_global) or None
            sigma_min (float): minimum noise level. Defaults to 1e-6.
        Returns:
            loss (torch.tensor): flow matching loss
        """
        B, T, D = x.shape
        device = x.device

        # Prepare target values
        lin_d = torch.clamp(lin_d, min=1e-8)  # (B,)
        if self.log_scale:
            d = torch.log(lin_d + 1e-8)  # (B,)
        else:
            d = lin_d  # (B,)

        # Sample time steps
        t = torch.rand([B], device=device, dtype=x.dtype)
        z = torch.randn_like(d)  # (B,)
        y = (1 - (1 - sigma_min) * t) * z + t * d  # (B,)
        u = d - (1 - sigma_min) * z  # Target velocity (B,)

        # Classifier-free guidance
        if self.cfg_rate > 0:
            cfg_prob = torch.rand(B, device=device)
            cfg_mask = cfg_prob > self.cfg_rate  # (B,)
            x = x * cfg_mask.view(-1, 1, 1)  # (B, T, D)
            c = c * cfg_mask.view(-1, 1) if c is not None else None

        v = self._forward(y, x, x_mask, t, c)

        loss = F.mse_loss(v, u, reduction="mean")

        return loss

    def _forward(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor = None,
    ):
        """
        Args:
            y (torch.tensor): batch of noisy duration values.
                shape: (B,)
            x (torch.tensor): batch of text representations.
                shape: (B, T, D)
            x_mask (torch.tensor): batch of sequence mask.
                shape: (B, T)
            t (torch.tensor): batch of time steps for flow matching.
        Returns:
            v (torch.tensor): predicted velocity values.
                shape: (B, T)
        """
        B, T, D = x.shape

        y = y.unsqueeze(-1).expand(-1, T).unsqueeze(-1)  # (B, T, 1)
        if self.global_cond_dim > 0:
            assert c is not None, "Global condition is not provided."
            c = c.unsqueeze(1).expand(-1, T, -1)
            x = torch.cat([y, x, c], dim=-1)
        else:
            x = torch.cat([y, x], dim=-1)

        t_emb = self.t_encoder(t)  # (B, D)

        x = self.prenet(x, x_mask, t_emb)  # (B, T, D)

        pos = self.make_positions(x)  # (B, T)
        attn_mask = x_mask.unsqueeze(1).expand(-1, T, -1)  # (B, T, T)
        x = self.encoder(x, pos, t_emb, attn_mask)  # (B, T, D)
        x = self.norm(x)  # (B, D)

        o = self.pooling(x, x_mask).unsqueeze(1)  # (B, 1, D)
        v = self.final_linear(o, t_emb).squeeze(1, 2)  # (B,)

        return v

    def forward(self, x, x_mask, c=None, n_timesteps=10, t_scheduler="cosine", cfg_rate=0.1):
        """
        Args:
            x (torch.tensor): batch of text representations.
                shape: (B, T, D)
            x_mask (torch.tensor): batch of sequence mask.
                shape: (B, T)
            c (torch.tensor): batch of global condition representations.
                shape: (B, D_global) or None
            n_timesteps (int): Number of timesteps for inference.
            t_scheduler (str): time scheduler. Defaults to "cosine".
            cfg_rate (float): CFG rate. Defaults to 1.0.
        Returns:
            lin_d (torch.tensor): predicted total duration values
                shape: (B,)
        """
        B, T, D = x.shape

        # Start from a Gaussian variable
        pred_d = torch.randn(B, device=x.device)  # (B,)

        # Time span for ODE integration
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=x.device, dtype=x.dtype)
        if t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        # ODE integration
        t, dt = t_span[0].unsqueeze(dim=0), t_span[1] - t_span[0]
        for step in range(1, len(t_span)):
            # Velocity prediction
            dphi_dt = self._forward(pred_d, x, x_mask, t, c)
            # Classifier-free guidance
            if cfg_rate > 0.0:
                cfg_x = torch.zeros_like(x)
                cfg_c = torch.zeros_like(c) if c is not None else None
                cfg_dphi_dt = self._forward(pred_d, cfg_x, x_mask, t, cfg_c)
                dphi_dt = dphi_dt + cfg_rate * (dphi_dt - cfg_dphi_dt)

            # Euler integration step
            pred_d = pred_d + dphi_dt * dt

            # Update time
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t_span[step]

        # Convert predicted values back to linear domain
        if self.log_scale:
            lin_d = torch.exp(pred_d) - 1e-8  # Convert back from log space
        else:
            lin_d = pred_d

        # Ensure positive duration
        lin_d = torch.clamp(lin_d, min=1e-8)

        return lin_d

    def make_positions(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        _p = torch.arange(0, T, 1, dtype=x.dtype, device=x.device)  # [T]
        p = repeat(_p, "t -> b t", b=B)
        return p
