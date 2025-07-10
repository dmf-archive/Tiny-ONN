import gradio as gr


def create_visualization_ui(update_callback_fn, get_total_tokens_fn, db_conn):
    """
    Creates a reusable Gradio UI for plotting and configuration.

    Args:
        update_callback_fn: The function to call when a control changes.
        get_total_tokens_fn: A function that returns the total number of tokens.
        db_conn: The database connection object.
    """
    total_tokens = get_total_tokens_fn(db_conn)
    
    with gr.Column(scale=1):
        plot_output = gr.Plot(label="Neural Activity Heatmap")
        
        with gr.Row():
            view_selector = gr.Radio(
                ["Activation", "Gradient Norm", "AbsMax", "Activation Z-Score", "Gradient Z-Score", "S_p"],
                label="Select View", 
                value="S_p"
            )
        
        time_slider = gr.Slider(
            minimum=0, 
            maximum=total_tokens, 
            step=1, 
            value=total_tokens,
            label="Timeline (Token Index)", 
            interactive=True
        )
        
        with gr.Accordion("Plotting Configuration", open=True):
            vmin_slider = gr.Number(label="Min Value (vmin)", value=-3.0)
            vmax_slider = gr.Number(label="Max Value (vmax)", value=3.0)
            w_act_slider = gr.Slider(minimum=0.0, maximum=5.0, value=1.0, step=0.1, label="Activation Weight (w_act)")
            w_grad_slider = gr.Slider(minimum=0.0, maximum=5.0, value=1.0, step=0.1, label="Gradient Weight (w_grad)")

    # Define the inputs for the update function
    inputs = [time_slider, view_selector, vmin_slider, vmax_slider, w_act_slider, w_grad_slider]
    
    # Link controls to the plot update function
    for control in inputs:
        control.change(update_callback_fn, inputs, [plot_output])
        
    # Return all the components that might be needed by the parent UI
    return plot_output, time_slider, view_selector, vmin_slider, vmax_slider, w_act_slider, w_grad_slider
