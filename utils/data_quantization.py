import numpy as np


def quantize_log_norm(v: float, max_norm_for_scaling: float = 300.0) -> int:
    """Non-linearly quantizes a float (norm) to an 8-bit integer using log1p."""
    v_log = np.log1p(abs(v)) # Use abs for safety, though norms are non-negative
    max_log_val = np.log1p(max_norm_for_scaling)
    v_norm_scaled = min(v_log / max_log_val, 1.0)
    return int(v_norm_scaled * 255)

def dequantize_log_norm(v_int: int, max_norm_for_scaling: float = 300.0) -> float:
    """Dequantizes an 8-bit integer back to a float (norm) using expm1."""
    v_norm_scaled = v_int / 255.0
    max_log_val = np.log1p(max_norm_for_scaling)
    v_log = v_norm_scaled * max_log_val
    return np.expm1(v_log)

def quantize_linear_symmetric(v: float, max_abs_val: float = 300.0) -> int:
    """Linearly quantizes a float from [-max_abs_val, max_abs_val] to [0, 255]."""
    v_shifted = v + max_abs_val # Shift to [0, 2*max_abs_val]
    v_clamped = max(0.0, min(v_shifted, 2 * max_abs_val))
    v_norm = v_clamped / (2 * max_abs_val)
    return int(v_norm * 255)

def dequantize_linear_symmetric(v_int: int, max_abs_val: float = 300.0) -> float:
    """Dequantizes an 8-bit integer [0, 255] back to a float in [-max_abs_val, max_abs_val]."""
    v_norm = v_int / 255.0
    v_shifted = v_norm * (2 * max_abs_val)
    return v_shifted - max_abs_val
