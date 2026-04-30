import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# --- 1. THE GRADIENT REVERSAL LAYER (GRL) ---
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Flips the sign of the gradient and scales it
        output = grad_output.neg() * ctx.alpha
        return output, None

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels, max_len=1024):
        super(PositionalEncoding1D, self).__init__()
        
        # Create a positional encoding matrix of shape (max_len, channels)
        pe = torch.zeros(max_len, channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2).float() * (-math.log(10000.0) / channels))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Shape: (1, max_len, channels) to broadcast across batch size
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [Batch, Sequence, Channels]
        seq_len = x.size(1)
        # Add the positional encoding to the input
        x = x + self.pe[:, :seq_len, :]
        return x

class SpectraBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, apply_pool=True):
        super().__init__()
        k1, k2, k3 = kernel_sizes
        
        # 1. Main Multi-Scale Branches
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=k1, padding=k1//2)
        self.branch2 = nn.Conv1d(in_channels, out_channels, kernel_size=k2, padding=k2//2)
        # Dilated branch to catch broad (1+z) stretched wings
        self.branch3 = nn.Conv1d(in_channels, out_channels, kernel_size=k3, padding=(k3-1)//2 * 2, dilation=2)
        
        self.fusion = nn.Conv1d(out_channels * 3, out_channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, out_channels) 
        
        # 2. --- NEW: RESIDUAL SHORTCUT ---
        # If the number of channels changes, we use a 1x1 conv to match dimensions
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(1, out_channels) # Match the normalization of the main branch
            )
        else:
            # If dimensions already match, just pass the data straight through
            self.shortcut = nn.Identity()
            
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout1d(0.15)
        # Pooling happens at the very end, after the addition
        self.pool = nn.MaxPool1d(2) if apply_pool else nn.Identity()

    def forward(self, x):
        # A. Process the main multi-scale features
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        
        out = torch.cat([x1, x2, x3], dim=1) 
        out = self.fusion(out)               
        out = self.norm(out)
        
        # B. --- NEW: ADD THE RESIDUAL ---
        # Process the raw input through the shortcut to match dimensions
        res = self.shortcut(x)
        # Add the express-lane features to the deep features BEFORE activation
        out = out + res 
        
        # C. Activate and Pool
        out = self.gelu(out)
        out = self.pool(out)
        
        return out


class TransformerStage(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.pos_encoder = PositionalEncoding1D(channels=embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.2)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Adding a small Feed-Forward Network (FFN) makes Transformers much more stable
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: [Batch, Sequence, Channels]
        # 1. Add spatial awareness
        x = self.pos_encoder(x) 
        
        # 2. Attention
        attn_output, attn_weights = self.mha(x, x, x)
        x = self.norm1(x + attn_output)
        
        # 3. Feed Forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x, attn_weights

# --- 3. THE MAIN SPECTRA-NET ARCHITECTURE ---
class SpectraNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # A. Feature Extractor (CNN Backbone using SpectraBlocks)
        self.feature_extractor = nn.Sequential(
            # Block 1: Massive reach to catch broad lines
            SpectraBlock(1, 64, kernel_sizes=[3, 15, 31], apply_pool=True),
            
            # Block 2: Medium reach
            SpectraBlock(64, 128, kernel_sizes=[3, 11, 21], apply_pool=True),
            
            # Block 3: Deep feature fusion
            SpectraBlock(128, 256, kernel_sizes=[3, 7, 11], apply_pool=False),
            
            nn.AdaptiveAvgPool1d(128) # Ensure consistent sequence length
        )
        
        # B. Global Correlation (Multi-Head Attention)
        self.global_corr = TransformerStage(embed_dim=256, num_heads=8)
        
        # C. Global Pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # D. Dynamic Dimension Calculation
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 1024)
            feat = self.feature_extractor(dummy_input) 
            self.flatten_dim = feat.shape[1] * 2 

        # E. Head 1: The Physics Classifier (AGN Type)
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1) 
        )
        prior_prob = 0.15 
        bias_init = torch.log(torch.tensor(prior_prob / (1.0 - prior_prob)))
        self.classifier[-1].bias.data.fill_(bias_init)

        # F. Head 2: The Redshift Adversary
        self.redshift_head = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1) 
        )

    def forward(self, x, alpha=0.0):
        # 1. Local multi-scale features
        x = self.feature_extractor(x)
        
        # 2. Global Attention
        x = x.permute(0, 2, 1)
        x, attn_weights = self.global_corr(x)
        x = x.permute(0, 2, 1) 
        
        # 3. Global Pooling and Flattening
        avg_p = self.avg_pool(x)
        max_p = self.max_pool(x)
        pooled = torch.cat([avg_p, max_p], dim=1) 
        flat = torch.flatten(pooled, 1)
        
        # 4. Primary Classification Task
        class_output = self.classifier(flat)
        
        # 5. Adversarial Redshift Task
        reversed_flat = GradientReversal.apply(flat, alpha)
        redshift_output = self.redshift_head(reversed_flat)
        
        return class_output, redshift_output


class SiameseSpectraNet(nn.Module):
    def __init__(self, pretrained_spectranet, freeze_backbone=True):
        super().__init__()
        
        # 1. The Pretrained Backbone
        self.feature_extractor = pretrained_spectranet.feature_extractor
        self.transformer = pretrained_spectranet.global_corr
        
        # 2. Freeze the backbone to preserve the learned physics
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            for param in self.transformer.parameters():
                param.requires_grad = False
                
        # 3. The New Temporal Change Head
        # Assuming the output of your global pooling is 512 dimensions.
        # Concatenating T1 (512), T2 (512), and Absolute Difference (512) = 1536
        self.temporal_head = nn.Sequential(
            nn.Linear(1536, 512),
            nn.GELU(),
            nn.Dropout(0.5), # Heavy dropout to prevent co-adaptation on the fused vector
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1) # Single logit for Binary Focal Loss
        )

    def forward_one_branch(self, x):
        """
        Extract the same 512-dim embedding used by the original SpectraNet backbone.
        """
        features = self.feature_extractor(x)          # [B, 256, L]
        features = features.permute(0, 2, 1)          # [B, L, 256]

        features, _ = self.transformer(features)      # use full TransformerStage

        features = features.permute(0, 2, 1)          # [B, 256, L]

        avg_pool = torch.mean(features, dim=-1)       # [B, 256]
        max_pool, _ = torch.max(features, dim=-1)     # [B, 256]

        embedding = torch.cat([avg_pool, max_pool], dim=1)  # [B, 512]
        return embedding

    def forward(self, x_t1, x_t2):
        """Processes both epochs and predicts if a change occurred."""
        # 1. Extract embeddings
        emb_t1 = self.forward_one_branch(x_t1)
        emb_t2 = self.forward_one_branch(x_t2)
        
        # 2. Calculate the distance vector in latent space
        abs_diff = torch.abs(emb_t1 - emb_t2)
        
        # 3. Fuse the temporal information
        fused_features = torch.cat([emb_t1, emb_t2, abs_diff], dim=1)
        
        # 4. Predict
        logits = self.temporal_head(fused_features)
        return logits
 
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