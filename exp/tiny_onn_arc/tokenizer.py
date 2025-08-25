class ArcTokenizer:
    PAD_TOKEN_ID = 10
    
    VOCAB_SIZE = 11
    
    ARC_COLORS = [
        "black", 
        "blue", 
        "red", 
        "green", 
        "yellow", 
        "grey74", 
        "magenta", 
        "orange", 
        "cyan", 
        "white",
        "#303030"
    ]

    @staticmethod
    def get_pad_token_id() -> int:
        return ArcTokenizer.PAD_TOKEN_ID