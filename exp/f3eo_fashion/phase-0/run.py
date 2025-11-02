import sys
from train import main

if __name__ == "__main__":
    # 允许通过命令行参数选择模型类型，例如: python run.py ffn_in_head
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
        if model_type not in ["standard", "ffn_in_head"]:
            print("Usage: python run.py [standard|ffn_in_head]")
            sys.exit(1)
        
        # 直接将模型类型传递给 train.main
        main(model_type)
    
    main(None)