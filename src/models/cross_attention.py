"""
KoeMorphクロスアテンションモデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class AudioEncoder(nn.Module):
    """
    音響特徴のエンコーダー
    log-mel (80×7) と eGeMAPS (88×1) を処理
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # Log-mel encoder (80 dim × 7 frames = 560)
        self.mel_encoder = nn.Sequential(
            nn.Linear(80, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # eGeMAPS encoder (88 dim)
        self.egemaps_encoder = nn.Sequential(
            nn.Linear(88, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Positional encoding for mel frames
        self.mel_pos_encoding = nn.Parameter(torch.randn(7, d_model) * 0.02)

    def forward(self, mel: torch.Tensor, egemaps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, 80, 7) log-mel features
            egemaps: (B, 88) eGeMAPS features

        Returns:
            encoded: (B, 8, d_model) encoded audio features
        """
        B = mel.shape[0]

        # Encode mel frames
        mel = mel.transpose(1, 2)  # (B, 7, 80)
        mel_encoded = self.mel_encoder(mel)  # (B, 7, d_model)
        mel_encoded = mel_encoded + self.mel_pos_encoding  # Add positional encoding

        # Encode eGeMAPS
        egemaps_encoded = self.egemaps_encoder(egemaps)  # (B, d_model)
        egemaps_encoded = egemaps_encoded.unsqueeze(1)  # (B, 1, d_model)

        # Concatenate all features
        encoded = torch.cat([mel_encoded, egemaps_encoded], dim=1)  # (B, 8, d_model)

        return encoded


class CrossAttentionLayer(nn.Module):
    """
    クロスアテンション層
    Query: ブレンドシェイプ埋め込み
    Key/Value: 音響特徴
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Query, Key, Value projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (B, 52, d_model) blendshape embeddings
            key_value: (B, 8, d_model) encoded audio features
            mask: optional attention mask

        Returns:
            output: (B, 52, d_model)
        """
        B, N_q, _ = query.shape
        N_kv = key_value.shape[1]

        # Residual connection
        residual = query

        # Project to Q, K, V
        Q = self.w_q(query).reshape(B, N_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key_value).reshape(B, N_kv, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(key_value).reshape(B, N_kv, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Reshape and project
        context = context.transpose(1, 2).reshape(B, N_q, self.d_model)
        output = self.w_o(context)
        output = self.dropout(output)

        # Add & Norm
        output = self.layer_norm(output + residual)

        return output


class BlendshapeMLP(nn.Module):
    """
    変化量から最終ブレンドシェイプ値への変換
    """

    def __init__(self, d_model: int = 256, hidden_dim: int = 512):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Blendshapes are in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 52, d_model) blendshape features

        Returns:
            blendshapes: (B, 52) final blendshape values
        """
        output = self.mlp(x)  # (B, 52, 1)
        output = output.squeeze(-1)  # (B, 52)
        return output


class SmoothingFilter(nn.Module):
    """
    時間的平滑化フィルタ（オプション）
    """

    def __init__(self, window_size: int = 3):
        super().__init__()
        self.window_size = window_size
        self.history = None

    def forward(self, x: torch.Tensor, reset: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, 52) current blendshapes
            reset: whether to reset history

        Returns:
            smoothed: (B, 52) smoothed blendshapes
        """
        if reset or self.history is None:
            self.history = [x.clone() for _ in range(self.window_size)]
            return x

        # Update history
        self.history = self.history[1:] + [x]

        # Apply moving average
        smoothed = torch.stack(self.history).mean(dim=0)

        return smoothed


class KoeMorphCrossAttention(nn.Module):
    """
    KoeMorphメインモデル
    音響特徴からブレンドシェイプを推定
    """

    def __init__(
        self,
        n_blendshapes: int = 52,
        embed_dim: int = 64,
        d_model: int = 256,
        n_heads: int = 8,
        mlp_hidden: int = 512,
        use_smoothing: bool = False,
        smoothing_window: int = 3
    ):
        super().__init__()

        self.n_blendshapes = n_blendshapes
        self.embed_dim = embed_dim
        self.d_model = d_model

        # Base blendshape embeddings (learnable)
        self.base_embeddings = nn.Parameter(
            torch.randn(n_blendshapes, embed_dim) * 0.02
        )

        # Project embeddings to model dimension
        self.embed_proj = nn.Linear(embed_dim, d_model)

        # Audio encoder
        self.audio_encoder = AudioEncoder(d_model)

        # Cross-attention layer
        self.cross_attention = CrossAttentionLayer(d_model, n_heads)

        # MLP for final blendshapes
        self.blendshape_mlp = BlendshapeMLP(d_model, mlp_hidden)

        # Optional smoothing
        self.use_smoothing = use_smoothing
        if use_smoothing:
            self.smoother = SmoothingFilter(smoothing_window)

    def forward(
        self,
        mel: torch.Tensor,
        egemaps: torch.Tensor,
        reset_smoothing: bool = False
    ) -> torch.Tensor:
        """
        Args:
            mel: (B, 80, 7) log-mel features
            egemaps: (B, 88) eGeMAPS features
            reset_smoothing: whether to reset smoothing history

        Returns:
            blendshapes: (B, 52) predicted blendshape values
        """
        B = mel.shape[0]

        # Encode audio features
        audio_features = self.audio_encoder(mel, egemaps)  # (B, 8, d_model)

        # Prepare blendshape queries
        base_embed = self.base_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, 52, embed_dim)
        queries = self.embed_proj(base_embed)  # (B, 52, d_model)

        # Cross-attention
        attended = self.cross_attention(queries, audio_features)  # (B, 52, d_model)

        # Generate blendshapes
        blendshapes = self.blendshape_mlp(attended)  # (B, 52)

        # Apply smoothing if enabled
        if self.use_smoothing and not self.training:
            blendshapes = self.smoother(blendshapes, reset_smoothing)

        return blendshapes

    def freeze_base_embeddings(self):
        """推論時に基礎埋め込みを固定"""
        self.base_embeddings.requires_grad = False

    def unfreeze_base_embeddings(self):
        """学習時に基礎埋め込みを更新可能に"""
        self.base_embeddings.requires_grad = True
