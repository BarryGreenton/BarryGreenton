import torch
from torch import nn
import torch.nn.functional as F

class NodeTransformerEncoder(nn.Module):
    """Map stock histories of shape (batch, stocks, 36, time) to node embeddings."""

    def __init__(self, d_model: int=32, nhead: int=8, num_layers: int=2, dropout: float=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(36, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        B, N, F, L = node_feats.shape
        assert F == 36, 'Expected 36 features per timestep'
        x = node_feats.permute(0, 1, 3, 2).reshape(B * N, L, F)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = x.reshape(B, N, self.d_model)
        return x

class GraphTransformerLayer(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int, dropout: float=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, 'hidden_dim must be divisible by num_heads'
        self.q_lin = nn.Linear(in_dim, hidden_dim)
        self.k_lin = nn.Linear(in_dim, hidden_dim)
        self.v_lin = nn.Linear(in_dim, hidden_dim)
        self.out_lin = nn.Linear(hidden_dim, in_dim)
        self.edge_bias_lin = nn.Linear(1, num_heads, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_dim)

    def forward(self, x: torch.Tensor, edge_matrix: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        q = self.q_lin(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_lin(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_lin(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5
        edge_bias = self.edge_bias_lin(edge_matrix.unsqueeze(-1)).permute(0, 3, 1, 2)
        attn_scores = attn_scores + edge_bias
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        context = torch.matmul(attn_probs, v).transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        output = self.out_lin(context)
        output = self.layer_norm(x + self.dropout(output))
        return output


class GNNModel(nn.Module):

    def __init__(self, node_in_dim: int=32, hidden_dim: int=32, num_layers: int=2, num_heads: int=4, dropout: float=0.1):
        super().__init__()
        self.encoder = nn.ModuleList([GraphTransformerLayer(node_in_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.decoder_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_embeddings: torch.Tensor, edge_matrix: torch.Tensor) -> torch.Tensor:
        x = node_embeddings
        for layer in self.encoder:
            x = layer(x, edge_matrix)
        z = self.decoder_proj(x)
        z_norm = F.normalize(z, p=2, dim=-1)
        pred_corr_matrix = torch.matmul(z_norm, z_norm.transpose(-2, -1))
        return pred_corr_matrix


def frobenius_norm_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    frobenius_norm_sq = torch.sum(diff ** 2, dim=(-2, -1))
    num_elements = target.shape[-1] * target.shape[-2]
    loss = frobenius_norm_sq / num_elements
    return loss.mean()


def relative_frobenius_loss(pred: torch.Tensor, target: torch.Tensor, epsilon: float=1e-08) -> torch.Tensor:
    numerator = torch.sum((pred - target) ** 2, dim=(-2, -1))
    denominator = torch.sum(target ** 2, dim=(-2, -1))
    loss = numerator / (denominator + epsilon)
    return loss.mean()
