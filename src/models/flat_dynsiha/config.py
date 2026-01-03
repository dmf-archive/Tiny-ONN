from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.utils import logging

logger = logging.get_logger(__name__)

class FlatDynSIHAConfig(Qwen3Config):
    model_type = "flat_dynsiha"

    def __init__(
        self,
        num_physical_heads=64,
        num_physical_experts=16,
        routing_gain=1.0,
        routing_dropout=0.0,
        top_k_experts=4,
        use_sars=True,
        **kwargs,
    ):
        self.num_physical_heads = num_physical_heads
        self.num_physical_experts = num_physical_experts
        self.routing_gain = routing_gain
        self.routing_dropout = routing_dropout
        self.top_k_experts = top_k_experts
        self.use_sars = use_sars
        super().__init__(**kwargs)
