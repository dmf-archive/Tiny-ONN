def calculate_scaling_ratios():
    # Data for Mixtral-8x7B (approximated from web search)
    mixtral_activated_params = 14e9
    # Training tokens not specified, using a common estimate for models of this size
    mixtral_training_tokens = 2e12 
    
    # Data for DeepSeek-V2
    deepseek_v2_activated_params = 21e9
    deepseek_v2_training_tokens = 8.1e12

    # Data for DeepSeek-V3
    deepseek_v3_activated_params = 37e9
    deepseek_v3_training_tokens = 14.8e12

    # Calculate ratios
    mixtral_ratio = mixtral_training_tokens / mixtral_activated_params
    deepseek_v2_ratio = deepseek_v2_training_tokens / deepseek_v2_activated_params
    deepseek_v3_ratio = deepseek_v3_training_tokens / deepseek_v3_activated_params

    print("--- MoE Model Scaling Ratios (Training Tokens / Activated Params) ---")
    print(f"Mixtral-8x7B: {mixtral_ratio:.2f}")
    print(f"DeepSeek-V2: {deepseek_v2_ratio:.2f}")
    print(f"DeepSeek-V3: {deepseek_v3_ratio:.2f}")

    avg_ratio = (mixtral_ratio + deepseek_v2_ratio + deepseek_v3_ratio) / 3
    print(f"\nAverage Ratio: {avg_ratio:.2f}")
    
    # Estimate data needed for our model
    our_model_params = 32 * 32 * 8 # Simplified estimation
    estimated_tokens = our_model_params * avg_ratio
    
    print(f"\n--- Estimated Data for Tiny-ONN (params: {our_model_params}) ---")
    print(f"Estimated training tokens needed: {estimated_tokens / 1e6:.2f} Million tokens")

if __name__ == "__main__":
    calculate_scaling_ratios()