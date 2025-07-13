from typing import ClassVar

from transformers.configuration_utils import PretrainedConfig


class TinyONNConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`TinyONNModel`].
    It is used to instantiate a TinyONN model according to the specified arguments,
    defining the model architecture.
    """

    model_type = "tinyonn"
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_dropout=0.0,
        # MoE specific parameters
        num_local_experts=96,
        num_experts_per_tok=8,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        # MoE specific
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
