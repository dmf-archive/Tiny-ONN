from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig

class TinyOnnConfig(Qwen2MoeConfig):
    def __init__(
        self,
        num_experts_per_layer=32,
        moe_intermediate_size=64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts_per_layer = num_experts_per_layer
        self.moe_intermediate_size = moe_intermediate_size
