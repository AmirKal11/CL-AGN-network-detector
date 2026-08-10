"""
architectures_v2.py
===================
Model definitions for the redesigned CL-AGN change detector.

Components
----------
SpectraEncoder            2-channel CNN + transformer encoder (the backbone).
SpectraDecoder            Upsampling decoder used only for SSL pretraining.
MaskedSpectraAutoencoder  Stage 1: encoder + decoder, trained by masked
                          reconstruction (self-supervised, no labels).
SiameseChangeNet          Stage 2: shared encoder + symmetric change head,
                          trained on real same-object epoch pairs.

The multi-scale conv block (SpectraBlock), positional encoding
(PositionalEncoding1D), and the attention stage (TransformerStage) are
defined directly in this module so it has no dependency on architectures.py.

Input convention everywhere: x has shape [B, 2, 4096]
    channel 0 = MAD-normalised flux (continuum-subtracted)
    channel 1 = [OIII]-normalised flux (raw physical flux / OIII anchor)
Out-of-coverage pixels are 0.0 in both channels (see preprocessing_oiii.py).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks (inlined from architectures.py so this module is self-contained)
# ---------------------------------------------------------------------------

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels, max_len=1024):
        super(PositionalEncoding1D, self).__init__()
        pe = torch.zeros(max_len, channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2).float() * (-math.log(10000.0) / channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, channels]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, Sequence, Channels]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class SpectraBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, apply_pool=True):
        super().__init__()
        k1, k2, k3 = kernel_sizes
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=k1, padding=k1 // 2)
        self.branch2 = nn.Conv1d(in_channels, out_channels, kernel_size=k2, padding=k2 // 2)
        # Dilated branch to catch broad (1+z) stretched wings
        self.branch3 = nn.Conv1d(in_channels, out_channels, kernel_size=k3,
                                 padding=(k3 - 1) // 2 * 2, dilation=2)
        self.fusion = nn.Conv1d(out_channels * 3, out_channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(1, out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout1d(0.15)
        self.pool = nn.MaxPool1d(2) if apply_pool else nn.Identity()

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.fusion(out)
        out = self.norm(out)
        res = self.shortcut(x)
        out = out + res
        out = self.gelu(out)
        out = self.pool(out)
        return out


class TransformerStage(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.pos_encoder = PositionalEncoding1D(channels=embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.2)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [Batch, Sequence, Channels]
        x = self.pos_encoder(x)
        attn_output, attn_weights = self.mha(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x, attn_weights


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Binary Focal Loss for imbalanced datasets.

        Args:
            alpha (float): Weighting factor for the positive class (0 to 1).
                           If alpha=0.25, positive class gets 0.25 weight, negative gets 0.75.
                           Set to ~ (num_negatives / total_samples) for your dataset.
            gamma (float): Focusing parameter. Higher values strongly down-weight easy examples.
            reduction (str): 'none', 'mean', or 'sum'.
        """
        super(BinaryFocalLossWithLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. Compute standard BCE loss using logits for stability
        # We use reduction='none' so we can apply the focal weight element-wise
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 2. Convert logits to probabilities
        p = torch.sigmoid(inputs)

        # 3. Calculate the probability of the *true* class (p_t)
        # If target is 1, p_t is p. If target is 0, p_t is (1 - p).
        p_t = p * targets + (1 - p) * (1 - targets)

        # 4. Calculate the modulating factor: (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma

        # 5. Apply the alpha weighting
        # If target is 1, use alpha. If target is 0, use (1 - alpha).
        alpha_factor = targets * self.alpha + (1 - targets) * (1.0 - self.alpha)

        # 6. Combine all pieces
        focal_loss = alpha_factor * modulating_factor * bce_loss

        # 7. Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

__all__ = [
    "SpectraEncoder",
    "SpectraDecoder",
    "MaskedSpectraAutoencoder",
    "SiameseChangeNet",
    "BinaryFocalLossWithLogits",
    "apply_span_mask",
    "load_encoder_into",
]

SEQ_LEN = 4096


# ----------------------------------------------------------------------
# Backbone encoder
# ----------------------------------------------------------------------
class SpectraEncoder(nn.Module):
    """
    2-channel spectral encoder.

    Mirrors the feature_extractor + global_corr of the original SpectraNet, so
    it behaves like the backbone you already know -- but it takes 2 input
    channels and exposes clean encode / embed methods instead of being fused
    with a Type-1/Type-2 classification head.

    forward(x)       -> [B, 512]   pooled embedding (default)
    feature_map(x)   -> [B, 256, 512]   pre-pool feature map
    embed(x)         -> [B, 512]   avg+max pooled embedding
    """

    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.in_channels = in_channels

        self.feature_extractor = nn.Sequential(
            # Block 1: long kernels to reach across broad lines.
            SpectraBlock(in_channels, 64, kernel_sizes=[3, 15, 31], apply_pool=True),
            # Block 2: medium reach.
            SpectraBlock(64, 128, kernel_sizes=[3, 11, 21], apply_pool=True),
            # Block 3: deep fusion, no pooling.
            SpectraBlock(128, 256, kernel_sizes=[3, 7, 11], apply_pool=False),
            # Fixed sequence length so the transformer/decoder dims are
            # constant. 512 tokens for the 4096-px wide grid (was 128 for the
            # old 1024-px grid): keeps ~8 px / token so broad-line wing shape
            # stays resolved, and 512 <= PositionalEncoding1D max_len (1024).
            nn.AdaptiveAvgPool1d(512),
        )
        self.global_corr = TransformerStage(embed_dim=256, num_heads=8)
        self.embed_dim = 512  # avg(256) concat max(256)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_channels, 4096] -> feature map [B, 256, 512]."""
        f = self.feature_extractor(x)          # [B, 256, 512]
        f = f.permute(0, 2, 1)                 # [B, 512, 256]
        f, _attn = self.global_corr(f)         # [B, 512, 256]
        f = f.permute(0, 2, 1)                 # [B, 256, 512]
        return f

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_channels, 4096] -> pooled embedding [B, 512]."""
        f = self.feature_map(x)                # [B, 256, 512]
        avg = f.mean(dim=-1)                   # [B, 256]
        mx = f.max(dim=-1).values             # [B, 256]
        return torch.cat([avg, mx], dim=1)     # [B, 512]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


# ----------------------------------------------------------------------
# Decoder (used for SSL pretraining only)
# ----------------------------------------------------------------------
class SpectraDecoder(nn.Module):
    """Reconstruct a [B, out_channels, 4096] spectrum from a [B, 256, 512] map."""

    def __init__(self, out_channels: int = 2):
        super().__init__()

        def up(cin: int, cout: int) -> nn.Sequential:
            # Nearest-neighbour upsampling + conv ("resize-convolution").
            # 'linear' upsampling (aten::upsample_linear1d) is not implemented
            # on Apple-Silicon MPS; nearest + conv is MPS-safe and also avoids
            # the checkerboard artifacts that transposed convs can introduce.
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(cin, cout, kernel_size=5, padding=2),
                nn.GroupNorm(1, cout),
                nn.GELU(),
            )

        # Three x2 upsamples take the 512-token feature map back to 4096 px
        # (same x8 ratio the decoder used for the old 128 -> 1024 grid).
        self.net = nn.Sequential(
            up(256, 128),   # 512  -> 1024
            up(128, 64),    # 1024 -> 2048
            up(64, 32),     # 2048 -> 4096
            nn.Conv1d(32, out_channels, kernel_size=5, padding=2),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)                  # [B, out_channels, 4096]


# ----------------------------------------------------------------------
# Stage 1: masked autoencoder (self-supervised pretraining)
# ----------------------------------------------------------------------
class MaskedSpectraAutoencoder(nn.Module):
    """
    Encoder + decoder trained by masked reconstruction.

    Training loop: blank random spans of the input, reconstruct the original,
    MSE loss on the masked positions only (see pretrain_ssl.py). After
    pretraining, only `.encoder` is kept.
    """

    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.encoder = SpectraEncoder(in_channels=in_channels)
        self.decoder = SpectraDecoder(out_channels=in_channels)

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        feat = self.encoder.feature_map(x_masked)
        return self.decoder(feat)              # [B, in_channels, 4096]


def apply_span_mask(
    x: torch.Tensor,
    valid: torch.Tensor | None = None,
    mask_ratio: float = 0.5,
    min_span: int = 64,
    max_span: int = 384,
):
    """
    Mask random contiguous spans of a batch (SpecAugment-style).

    When `valid` is given, spans are drawn only from each spectrum's covered
    region, so the masking budget is spent on real data rather than on the
    already-zero, out-of-coverage pixels of the wide grid. The reconstruction
    loss (pretrain_ssl.masked_mse) then scores span-masked AND covered pixels.

    Parameters
    ----------
    x     : tensor [B, C, L]
    valid : bool tensor [B, L] or None   True where the spectrum has coverage

    Returns
    -------
    x_masked  : tensor [B, C, L]   masked positions set to 0 in every channel
    span_mask : bool tensor [B, L] True where span-masked
    """
    B, C, L = x.shape
    # Build the mask on CPU with NumPy -> no per-iteration GPU sync.
    rng = np.random.default_rng()
    mask_np = np.zeros((B, L), dtype=bool)
    if valid is not None:
        valid_np = valid.detach().to("cpu").numpy().astype(bool)
    else:
        valid_np = np.ones((B, L), dtype=bool)
    for b in range(B):
        vidx = np.where(valid_np[b])[0]
        if vidx.size == 0:
            continue
        lo, hi = int(vidx[0]), int(vidx[-1])      # contiguous covered window
        region = hi - lo + 1
        if region <= min_span:
            mask_np[b, lo:hi + 1] = True
            continue
        target = int(mask_ratio * region)
        hi_span = min(max_span, region)
        lo_span = min(min_span, hi_span)
        filled, guard = 0, 0
        while filled < target and guard < 2000:
            span = int(rng.integers(lo_span, hi_span + 1))
            start = int(rng.integers(lo, hi - span + 2))
            mask_np[b, start:start + span] = True
            filled = int(mask_np[b].sum())
            guard += 1
    span_mask = torch.from_numpy(mask_np).to(x.device)
    x_masked = x.masked_fill(span_mask.unsqueeze(1), 0.0)
    return x_masked, span_mask


# ----------------------------------------------------------------------
# Stage 2: Siamese change detector
# ----------------------------------------------------------------------
class SiameseChangeNet(nn.Module):
    """
    Shared-encoder change detector for real same-object epoch pairs.

    The two epoch embeddings are fused symmetrically:
        [ e1 + e2 , |e1 - e2| , e1 * e2 ]
    so the prediction is invariant to epoch order (a CL-AGN is a CL-AGN
    regardless of which epoch is called "first") and cannot learn an
    order-dependent shortcut. No epoch-swap augmentation is needed.

    forward(x1, x2) -> [B, 1] logit ; P(change) = sigmoid(logit)
    """

    def __init__(self, in_channels: int = 2, dropout: float = 0.4,
                 prior_pos: float = 0.1):
        super().__init__()
        self.encoder = SpectraEncoder(in_channels=in_channels)
        d = self.encoder.embed_dim             # 512
        fused = d * 3                          # sum / abs-diff / product

        self.head = nn.Sequential(
            nn.Linear(fused, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, 1),
        )

        # Initialise the output bias to the expected positive rate so early
        # training is stable under heavy class imbalance.
        prior_pos = min(max(prior_pos, 1e-4), 1 - 1e-4)
        bias = torch.log(torch.tensor(prior_pos / (1.0 - prior_pos)))
        self.head[-1].bias.data.fill_(bias)

    def encode_pair(self, x1: torch.Tensor, x2: torch.Tensor):
        return self.encoder.embed(x1), self.encoder.embed(x2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        e1, e2 = self.encode_pair(x1, x2)
        fused = torch.cat([e1 + e2, (e1 - e2).abs(), e1 * e2], dim=1)
        return self.head(fused)                # [B, 1]


# ----------------------------------------------------------------------
# Weight transfer: SSL encoder -> Stage-2 model
# ----------------------------------------------------------------------
def load_encoder_into(model, ssl_checkpoint_path: str, device: str = "cpu",
                      verbose: bool = True):
    """
    Load SSL-pretrained encoder weights into `model.encoder`.

    Accepts a checkpoint that either is the encoder state_dict directly or a
    dict containing an 'encoder_state_dict' key (as saved by pretrain_ssl.py).
    """
    ckpt = torch.load(ssl_checkpoint_path, map_location=device)
    state = ckpt["encoder_state_dict"] if isinstance(ckpt, dict) and \
        "encoder_state_dict" in ckpt else ckpt
    missing, unexpected = model.encoder.load_state_dict(state, strict=False)
    if verbose:
        print(f"[load_encoder_into] loaded SSL encoder from {ssl_checkpoint_path}")
        if missing:
            print(f"  missing keys    : {len(missing)}")
        if unexpected:
            print(f"  unexpected keys : {len(unexpected)}")
    return model
