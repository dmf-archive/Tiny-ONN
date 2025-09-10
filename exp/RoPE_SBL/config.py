import torch

CONFIG = {
    "BATCH_SIZE": 8,
    "SEQ_LEN": 128,
    "D_MODEL": 64,
    "VOCAB_SIZE": 22,
    "NUM_TRANSFORMER_BLOCKS": 2,
    "D_FFN_FACTOR": 4,
    "LR": 3e-3,
    "OPERAND_RANGES": [(0, 9), (10, 99)],
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu", 
    "DTYPE": torch.bfloat16,
    "TRAINING_STEPS": 1000,
    "LOG_INTERVAL": 50,
    "KL_PRIOR_EPSILON": 1e-9
}