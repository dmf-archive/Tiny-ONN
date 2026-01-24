from transformers import PretrainedConfig

class FlatDynSIHAConfig(PretrainedConfig):
    model_type = "flat_dynsiha"
    
    def __init__(
        self,
        vocab_size=16,
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=8,
        num_experts=4,
        top_k=2,
        max_position_embeddings=4096,
        ffn_scale=4,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_cache=True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.max_position_embeddings = max_position_embeddings
        self.ffn_scale = ffn_scale
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        super().__init__(**kwargs)
