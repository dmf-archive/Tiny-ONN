from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class TinyOnnConfig(Qwen3Config):
    def __init__(
        self,
        num_experts_per_layer=32,
        moe_intermediate_size=64,
        num_experts_per_tok=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts_per_layer = num_experts_per_layer
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
