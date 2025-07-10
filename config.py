# Default configuration for plotting
# Users can adjust these values in the replay_analysis.py UI
PLOT_CONFIG = {
    "activation": {
        "vmin": 0,
        "vmax": 450, # Increased to accommodate high activation values
        "cmap": "viridis"
    },
    "gradient_norm": {
        "vmin": 0,
        "vmax": 1e-7,
        "cmap": "plasma"
    },
    "pi_diff": {
        "vmin": -50,
        "vmax": 450,
        "cmap": "coolwarm",
        "gradient_weight": 1.0e8 # Scale gradients to be comparable to activations
    },
    "absmax": {
        "vmin": 0,
        "vmax": 256,
        "cmap": "cividis"
    }
}

# Mapping from view mode in the UI to the config key
VIEW_MODE_MAP = {
    "Activation": "activation",
    "Gradient Norm": "gradient_norm",
    "PI Diff Scatter": "pi_diff",
    "AbsMax": "absmax"
}

# Quantization scale for storing float values as integers in the database
QUANTIZATION_SCALE = 1000
