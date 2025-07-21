from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


class TinyOnnConfig(Qwen3MoeConfig):
    model_type = "tiny_onn"

    def __init__(
        self,
        num_experts=32,
        moe_intermediate_size=64,
        num_experts_per_tok=-1,
        output_router_logits=False,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            moe_intermediate_size=moe_intermediate_size,
            num_experts_per_tok=num_experts_per_tok,
            output_router_logits=output_router_logits,
            **kwargs,
        )
