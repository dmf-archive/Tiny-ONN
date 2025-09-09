import torch

CONFIG = {
    "BATCH_SIZE": 32,
    "SEQ_LEN": 128,
    "D_MODEL": 128,
    "VOCAB_SIZE": 22,
    "NUM_TRANSFORMER_BLOCKS": 4,
    "D_FFN_FACTOR": 4,
    "LR": 3e-3,
    "OPERAND_RANGES": [(0, 9), (10, 99), (100, 999)],
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DTYPE": torch.bfloat16,
    "TRAINING_STEPS": 5000,
    "LOG_INTERVAL": 50,
    "KL_PRIOR_EPSILON": 1e-9,

    "BCAT_CLUSTER_THRESHOLD": 0.5,
    "BCAT_MIN_BLOCK_SIZE": 8,
}