from transformers import AutoConfig


def get_config():
    model_name = "Qwen/Qwen3-0.6B"
    cache_dir = "weights"
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    print(config)


if __name__ == "__main__":
    get_config()
