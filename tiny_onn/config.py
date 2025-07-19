from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class TinyOnnConfig(Qwen3Config):
    model_type = "tiny_onn"

    def __init__(
        self,
        num_experts=32,
        moe_intermediate_size=64,
        num_experts_per_tok=-1,
        output_router_logits=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.output_router_logits = output_router_logits
