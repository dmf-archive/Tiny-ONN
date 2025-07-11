import gradio as gr


def create_main_ui(
    process_input_fn,
    update_plot_fn,
    get_total_tokens_fn,
    db_conn_fn,
    is_replay_mode=False
):
    """
    Builds the complete Gradio UI by integrating all components.
    """
    with gr.Blocks(css=".gradio-container {max-width: 1200px; margin: auto;}") as demo:
        gr.Markdown("# Digital fMRI: A Tiny-ONN Pilot Study")
        gr.Markdown("This interface provides a real-time visualization of a large language model's internal states. It operates by conducting a per-token (time-slice) scan of the model's activation values (digital neurons) and gradients (approximating hemodynamic responses), offering insights into its cognitive processes.")

        total_tokens = get_total_tokens_fn()

        with gr.Row():
            # Chat and Controls Column
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(height=400, label="Chat History", type="messages", bubble_full_width=False)
                with gr.Row():
                    use_fmri_checkbox = gr.Checkbox(
                        label="Enable fMRI Scan",
                        value=True,
                        info="Uncheck for pure inference mode",
                        visible=not is_replay_mode
                    )
                    submit_btn = gr.Button("Send", visible=not is_replay_mode)
                msg = gr.Textbox(label="Your Message", visible=not is_replay_mode)

            # Visualization Column
            with gr.Column(scale=1):
                plot_output = gr.Plot(label="Neural Activity Heatmap")
                view_selector = gr.Radio(
                    ["Activation", "Gradient Norm", "AbsMax", "Activation Z-Score", "Gradient Z-Score", "S_p"],
                    label="Select View",
                    value="S_p"
                )
                time_slider = gr.Slider(
                    minimum=0,
                    maximum=max(0, total_tokens - 1),
                    step=1,
                    value=max(0, total_tokens - 1),
                    label="Timeline (Token Index)",
                    interactive=True
                )
                with gr.Accordion("Plotting Configuration", open=True):
                    vmin_slider = gr.Number(label="Min Value (vmin)", value=-3.0)
                    vmax_slider = gr.Number(label="Max Value (vmax)", value=3.0)

        # Define inputs for the update functions
        plot_update_inputs = [time_slider, view_selector, vmin_slider, vmax_slider]

        # Link controls to the plot update function
        for control in plot_update_inputs:
            control.change(update_plot_fn, plot_update_inputs, [plot_output])

        if not is_replay_mode:
            submit_inputs = [msg, chatbot, view_selector, vmin_slider, vmax_slider, use_fmri_checkbox]
            submit_outputs = [msg, chatbot, plot_output, time_slider]
            submit_btn.click(process_input_fn, submit_inputs, submit_outputs)
            msg.submit(process_input_fn, submit_inputs, submit_outputs)

    return demo
