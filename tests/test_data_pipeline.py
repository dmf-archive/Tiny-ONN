import os
import random
import sys

import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import run_analysis_pipeline, setup_database
from utils.plotting import update_plot


class MockTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return f"mock response with {len(ids)} tokens"

def mock_run_forward_pass(model, tokenizer, user_message, history):
    num_tokens = random.randint(5, 15)
    final_response = f"mock response with {num_tokens} tokens"
    full_sequence_ids = list(range(100, 100 + num_tokens))
    per_token_activation_data = []
    
    mock_layer_types = [
        "mlp.down_proj", "self_attn.o_proj", "self_attn.q_proj",
        "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"
    ]

    for _ in range(num_tokens):
        token_data = {}
        for i in range(28):
            for layer_type in mock_layer_types:
                param_name = f"model.layers.{i}.{layer_type}.weight"
                num_blocks = random.randint(8, 16)
                token_data[param_name] = {
                    "activation": [random.uniform(0.1, 50.0) for _ in range(num_blocks)],
                    "weight": [random.uniform(0.1, 10.0) for _ in range(num_blocks)],
                }
        per_token_activation_data.append(token_data)
        
    return final_response, per_token_activation_data, full_sequence_ids

def mock_run_backward_pass(model, full_sequence_ids, last_token_data):
    for _param_name, metrics in last_token_data.items():
        num_blocks = len(metrics.get("activation", []))
        if num_blocks > 0:
            metrics["gradient"] = [random.uniform(0.01, 10.0) for _ in range(num_blocks)]

@pytest.fixture(scope="function")
def test_environment(monkeypatch):
    monkeypatch.setattr("main.run_forward_pass_and_capture_activations", mock_run_forward_pass)
    monkeypatch.setattr("inference.backward_pass.run_per_token_backward_pass", mock_run_backward_pass)
    
    db_conn = setup_database(memory=True)
    
    mock_model = "mock_model_object"
    mock_tokenizer = MockTokenizer()
    
    yield mock_model, mock_tokenizer, db_conn
    
    db_conn.close()
    if os.path.exists("test_plot_output.png"):
        os.remove("test_plot_output.png")

def test_pipeline_equivalence_and_data_integrity(test_environment):
    mock_model, mock_tokenizer, db_conn = test_environment
    
    history, response = run_analysis_pipeline(
        current_model=mock_model,
        current_tokenizer=mock_tokenizer,
        user_message="Hello, world!",
        history=[],
        current_db_conn=db_conn,
        tokens_processed_offset=0
    )
    
    assert "mock response" in response
    assert len(history) == 1
    
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM block_metrics")
    count = cursor.fetchone()[0]
    assert count > 0, "Pipeline ran but no data was inserted into the database."
    
    cursor.execute("SELECT * FROM block_metrics ORDER BY RANDOM() LIMIT 1")
    record = cursor.fetchone()
    assert record is not None
    
    token_idx, param_name, block_idx, activation, grad_norm, absmax = record
    assert isinstance(token_idx, int)
    assert "model.layers" in param_name
    assert isinstance(block_idx, int)
    assert isinstance(activation, float)
    assert isinstance(grad_norm, float)
    assert isinstance(absmax, float)

    cursor.execute("SELECT SUM(grad_norm) FROM block_metrics")
    total_grad_norm = cursor.fetchone()[0]
    assert total_grad_norm > 0, "Backward pass did not seem to add gradient data."

def test_plot_generation_from_pipeline_data(test_environment):
    mock_model, mock_tokenizer, db_conn = test_environment
    
    run_analysis_pipeline(
        current_model=mock_model,
        current_tokenizer=mock_tokenizer,
        user_message="Generate data for plotting.",
        history=[],
        current_db_conn=db_conn,
        tokens_processed_offset=0
    )
    
    import main
    total_tokens = main.total_tokens_processed
    assert total_tokens > 0
    
    # Test all new view modes
    for view_mode in ["Activation", "Gradient Norm", "AbsMax", "Activation Z-Score", "Gradient Z-Score", "S_p"]:
        plot_fig = update_plot(
            token_idx=total_tokens - 1,
            view_mode=view_mode,
            db_conn=db_conn,
            total_tokens_processed=total_tokens,
            w_act=1.5,  # Provide dummy weights for S_p
            w_grad=0.5
        )
        
        assert plot_fig is not None, f"update_plot returned None for view_mode: {view_mode}"
        assert isinstance(plot_fig, plt.Figure)
        
        # Save one plot to ensure file generation works
        if view_mode == "S_p":
            plot_filename = "test_plot_output.png"
            plot_fig.savefig(plot_filename)
            assert os.path.exists(plot_filename)
        
        plt.close(plot_fig)
