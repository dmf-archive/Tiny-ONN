from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class TinyOnnConfig(Qwen3Config):
    model_type = "tiny_onn"

    def __init__(
        self,
        num_experts_per_layer=32,
        moe_intermediate_size=-1,
        num_experts_per_tok=-1,
        attn_implementation="eager",
        **kwargs,
    ):
        super().__init__(attn_implementation=attn_implementation, **kwargs)
        self.num_experts_per_layer = num_experts_per_layer
        self.moe_intermediate_size = (
            moe_intermediate_size
            if moe_intermediate_size != -1
            else self.intermediate_size
        )
        self.num_experts_per_tok = num_experts_per_tok
