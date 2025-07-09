import os
import random
import sys
import pytest
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions to be tested
from main import run_analysis_pipeline, setup_database
from utils.plotting import update_plot

# --- Mock Objects and Functions ---

class MockTokenizer:
    """A lightweight mock tokenizer."""
    def decode(self, ids, skip_special_tokens=True):
        return f"mock response with {len(ids)} tokens"

def mock_run_forward_pass(model, tokenizer, user_message, history):
    """
    A mock version of run_forward_pass that returns predictable,
    structured data without needing a real model.
    """
    num_tokens = random.randint(5, 15)
    generated_ids = list(range(num_tokens))
    per_token_block_data = []
    
    mock_layer_types = [
        "mlp.down_proj", "self_attn.o_proj", "self_attn.q_proj",
        "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"
    ]

    for _ in range(num_tokens):
        token_data = {}
        for i in range(28):  # 28 layers
            for layer_type in mock_layer_types:
                param_name = f"model.layers.{i}.{layer_type}.weight"
                num_blocks = random.randint(8, 16)
                token_data[param_name] = {
                    "activation": [random.uniform(0.1, 50.0) for _ in range(num_blocks)],
                    "gradient": [random.uniform(0.01, 10.0) for _ in range(num_blocks)],
                    "weight": [random.uniform(0.1, 10.0) for _ in range(num_blocks)],
                }
        per_token_block_data.append(token_data)
        
    # The third return value (input_ids) is not used in the pipeline being tested
    return generated_ids, per_token_block_data, None

# --- Fixtures ---

@pytest.fixture(scope="function")
def test_environment(monkeypatch):
    """
    Sets up a clean test environment for each test function.
    - Creates an in-memory SQLite database.
    - Mocks the model and tokenizer.
    - Patches the forward pass function to use our mock data generator.
    - Resets global token counter.
    """
    # Patch the real forward pass with our mock version
    monkeypatch.setattr("main.run_forward_pass", mock_run_forward_pass)
    
    # Setup a clean in-memory database for each test
    db_conn = setup_database(memory=True)
    
    # Reset the global token counter from main.py
    monkeypatch.setattr("main.total_tokens_processed", 0)
    
    # Create mock model and tokenizer objects
    mock_model = "mock_model_object"  # Can be a simple string as it's not used by the mock
    mock_tokenizer = MockTokenizer()
    
    yield mock_model, mock_tokenizer, db_conn
    
    # Teardown
    db_conn.close()
    if os.path.exists("test_plot_output.png"):
        os.remove("test_plot_output.png")

# --- Tests ---

def test_pipeline_equivalence_and_data_integrity(test_environment):
    """
    Tests that the analysis pipeline correctly processes data and stores it
    with integrity, faithfully simulating the main script's logic.
    """
    mock_model, mock_tokenizer, db_conn = test_environment
    
    # 1. Run the actual analysis pipeline from main.py
    history, response = run_analysis_pipeline(
        current_model=mock_model,
        current_tokenizer=mock_tokenizer,
        user_message="Hello, world!",
        history=[],
        current_db_conn=db_conn
    )
    
    # 2. Assertions
    assert "mock response" in response
    assert len(history) == 1
    
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM block_metrics")
    count = cursor.fetchone()[0]
    assert count > 0, "Pipeline ran but no data was inserted into the database."
    
    # 3. Verify a random record for data integrity
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

def test_plot_generation_from_pipeline_data(test_environment):
    """
    Tests that a plot can be successfully generated from the data
    produced by the analysis pipeline.
    """
    mock_model, mock_tokenizer, db_conn = test_environment
    
    # 1. Run pipeline to populate the database
    run_analysis_pipeline(
        current_model=mock_model,
        current_tokenizer=mock_tokenizer,
        user_message="Generate data for plotting.",
        history=[],
        current_db_conn=db_conn
    )
    
    # 2. Retrieve total tokens processed (from the patched global)
    import main
    total_tokens = main.total_tokens_processed
    assert total_tokens > 0
    
    # 3. Generate the plot
    plot_fig = update_plot(
        token_idx=total_tokens - 1,
        view_mode="Activation",
        db_conn=db_conn,
        total_tokens=total_tokens
    )
    
    # 4. Assertions
    assert plot_fig is not None, "update_plot returned None."
    assert isinstance(plot_fig, plt.Figure)
    
    plot_filename = "test_plot_output.png"
    plot_fig.savefig(plot_filename)
    assert os.path.exists(plot_filename)
    plt.close(plot_fig)
