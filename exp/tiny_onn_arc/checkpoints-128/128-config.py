class TinyOnnArcConfig:
    model_type = "tiny_onn_arc"

    vocab_size: int = 12
    mask_token_id: int = 11
    hidden_size: int = 128
    num_hidden_layers: int = 16
    max_position_embeddings: int = 1861
    type_vocab_size: int = 2

    # DynSMHA specific
    max_attention_experts: int = 128
    min_attention_experts: int = 32
    head_dim: int = 8

    # DynMoE specific
    max_moe_experts: int = 64
    min_moe_experts: int = 32
    intermediate_size: int = 16

    # Loss weights
    w_ce_smha: float = 1.0
    w_kl_smha: float = 1.0
    w_aux_smha: float = 1.0
    w_ce_moe: float = 1.0
    w_kl_moe: float = 1.0
    w_aux_moe: float = 1.0

    # Predictive Integrity Score specific
    pi_alpha: float = 64.0
    pi_gamma: float = 0.5
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
