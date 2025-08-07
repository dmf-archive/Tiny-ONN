from transformers import PretrainedConfig


class TinyOnnConfig(PretrainedConfig):
    model_type = "tiny_onn"

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        head_dim: int,
        num_experts_per_layer: int,
        moe_intermediate_size: int,
        selection_block_size: int,
        max_selected_blocks: int,
        w_entropy: float = 0.1,
        w_sparse: float = 0.1,
        w_ce: float = 1.0,
        w_kl: float = 1.0,
        pi_alpha: float = 1.0,
        pi_gamma: float = 1.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_experts_per_layer = num_experts_per_layer
        self.moe_intermediate_size = (
            moe_intermediate_size
            if moe_intermediate_size != -1
            else self.intermediate_size
        )
        self.selection_block_size = selection_block_size
        self.max_selected_blocks = max_selected_blocks
        self.max_position_embeddings = self.selection_block_size * self.max_selected_blocks
        self.w_entropy = w_entropy
        self.w_sparse = w_sparse
        self.w_ce = w_ce
        self.w_kl = w_kl
        self.pi_alpha = pi_alpha
        self.pi_gamma = pi_gamma
        super().__init__(**kwargs)
