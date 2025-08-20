from transformers import PretrainedConfig


class TinyOnnConfig(PretrainedConfig):
    model_type = "tiny_onn"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 256,
        embedding_size: int = 128,
        num_hidden_layers: int = 3,
        max_position_embeddings: int = 1024,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        # RoPE specific
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        # DynSMHA specific
        max_attention_experts: int = 32,
        min_attention_experts: int = 16,
        head_dim: int = 32,
        # DynNSA specific
        block_size: int = 64,
        # DynMoE specific
        max_moe_experts: int = 32,
        min_moe_experts: int = 16,
        intermediate_size: int = 32,
        # Expert reborn specific
        k_reborn_experts: int = -1,
        # Loss weights
        w_ce_smha: float = 1.0,
        w_kl_smha: float = 1.0,
        w_aux_smha: float = 1.0,
        w_ce_moe: float = 1.0,
        w_kl_moe: float = 1.0,
        w_aux_moe: float = 1.0,
        # Predictive Integrity Score specific
        pi_alpha: float = 32,
        pi_gamma: float = 0.5,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.head_dim = head_dim
        self.block_size = block_size
        self.bias = bias
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_attention_experts = max_attention_experts
        self.min_attention_experts = min_attention_experts
        self.max_moe_experts = max_moe_experts
        self.min_moe_experts = min_moe_experts
        self.k_reborn_experts = k_reborn_experts
        self.w_ce_smha = w_ce_smha
        self.w_kl_smha = w_kl_smha
        self.w_aux_smha = w_aux_smha
        self.w_ce_moe = w_ce_moe
        self.w_kl_moe = w_kl_moe
        self.w_aux_moe = w_aux_moe
        self.pi_alpha = pi_alpha
        self.pi_gamma = pi_gamma
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
