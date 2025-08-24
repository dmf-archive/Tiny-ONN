import bayesian_torch.layers as bl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bayesian_config import BayesianConfig


class BayesianBlock(nn.Module):
    def __init__(self, config: BayesianConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.hidden_size)
        self.fc1 = bl.LinearReparameterization(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            prior_mean=0,
            prior_variance=1.0,
            posterior_rho_init=0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        x, _ = self.fc1(x)
        x = F.gelu(x)
        return residual + x


class ImageEmbedding(nn.Module):
    def __init__(self, config: BayesianConfig):
        super().__init__()
        self.config = config
        self.tok_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, config.MAX_GRID_SIZE, config.MAX_GRID_SIZE, config.hidden_size))

    def forward(self, input_grid: torch.Tensor) -> torch.Tensor:
        B, H, W = input_grid.shape
        tok_embeddings = self.tok_embed(input_grid)
        pos_embeddings = self.pos_embed[:, :H, :W, :]
        return tok_embeddings + pos_embeddings


class BayesianTinyOnn(nn.Module):
    def __init__(self, config: BayesianConfig):
        super().__init__()
        self.config = config
        self.embeddings = ImageEmbedding(config)
        self.layers = nn.ModuleList([BayesianBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_ln = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_grid: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_grid)
        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)
        logits = self.lm_head(x)
        return logits
