class TinyOnnArcConfig:
    model_type = "tiny_onn_arc"

    vocab_size: int = 11
    hidden_size: int = 128
    num_hidden_layers: int = 8
    max_position_embeddings: int = 1861
    
    # DynSMHA specific
    max_attention_experts: int = 24
    min_attention_experts: int = 24
    head_dim: int = 16

    # DynMoE specific
    max_moe_experts: int = 24
    min_moe_experts: int = 24
    intermediate_size: int = 16

    # Loss weights
    w_ce_smha: float = 1.0
    w_kl_smha: float = 1.0
    w_aux_smha: float = 1.5
    w_ce_moe: float = 1.0
    w_kl_moe: float = 1.0
    w_aux_moe: float = 1.5

    # Predictive Integrity Score specific
    pi_alpha: float = 64.0
    pi_gamma: float = 0.5
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
