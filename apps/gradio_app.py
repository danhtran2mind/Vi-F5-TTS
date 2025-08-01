import gradio as gr
from gradio_app.components import (
    get_files_in_ckpts, handle_file_upload, 
    run_tts_inference,
    run_setup_script
)
from gradio_app.asr_utils import transcribe_audio

def create_gradio_app():
    """Create Gradio interface for F5-TTS inference with Whisper ASR."""
    # Run setup script to ensure dependencies are installed
    run_setup_script()

    # Function to update reference text based on audio file and Whisper checkbox
    def update_ref_text(audio_file_path, use_whisper):
        if use_whisper and audio_file_path:
            return transcribe_audio(audio_file_path)
        return gr.update()

    def toggle_model_inputs(use_upload):
        return (
            gr.update(visible=not use_upload),
            gr.update(visible=not use_upload),
            gr.update(visible=not use_upload),
            gr.update(visible=use_upload),
            gr.update(visible=use_upload),
            gr.update(visible=use_upload)
        )

    CSS = open("apps/gradio_app/static/styles.css", "r").read()
    with gr.Blocks(css=CSS) as demo:
        gr.Markdown("# F5-TTS Audio Generation")
        gr.Markdown("Generate high-quality audio with a fine-tuned F5-TTS model. Upload reference audio, use Whisper ASR for transcription, enter text, adjust speed, and select or upload model files.")
        with gr.Row():
            with gr.Column():
                ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                with gr.Group():
                    use_whisper = gr.Checkbox(label="Use Whisper ASR for Transcription", value=False)
                    ref_text = gr.Textbox(
                        label="Reference Text",
                        placeholder="e.g., Sau nhà Ngô, lần lượt các triều Đinh...",
                        lines=1
                    )
                    gen_text = gr.Textbox(
                        label="Generated Text",
                        placeholder="e.g., Nhà Tiền Lê, Lý và Trần đã chống trả...",
                        lines=1
                    )
                generate_btn = gr.Button("Generate Audio")
            with gr.Column():
                output_audio = gr.Audio(label="Generated Audio")
                output_text = gr.Textbox(label="Status", interactive=False)
                with gr.Group():
                    speed = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Speed")
                    model_cfg = gr.Dropdown(
                        choices=get_files_in_ckpts([".yaml"]),
                        label="Model Config (*.yaml)",
                        value=get_files_in_ckpts([".yaml"])[0],
                        visible=True
                    )
                    ckpt_file = gr.Dropdown(
                        choices=get_files_in_ckpts([".pt", ".safetensors"], include_subdirs=True),
                        label="Checkpoint File (*.pt or *.safetensors)",
                        value=get_files_in_ckpts([".pt", ".safetensors"], include_subdirs=True)[0],
                        visible=True
                    )
                    vocab_file = gr.Dropdown(
                        choices=get_files_in_ckpts([".txt", ".safetensors"]),
                        label="Vocab File (*.txt or *.safetensors)",
                        value=get_files_in_ckpts([".txt", ".safetensors"])[0],
                        visible=True
                    )
                    use_upload = gr.Checkbox(label="Upload Custom Model Files", value=False)
                    model_cfg_upload = gr.File(label="Model Config (*.yaml)", file_types=[".yaml"], visible=False)
                    ckpt_file_upload = gr.File(label="Checkpoint File (*.pt or *.safetensors)", file_types=[".pt", ".safetensors"], visible=False)
                    vocab_file_upload = gr.File(label="Vocab File (*.txt or *.safetensors)", file_types=[".txt", ".safetensors"], visible=False)
        
        ref_audio.change(fn=update_ref_text, inputs=[ref_audio, use_whisper], outputs=ref_text)
        use_whisper.change(fn=update_ref_text, inputs=[ref_audio, use_whisper], outputs=ref_text)
        use_upload.change(
            fn=toggle_model_inputs,
            inputs=[use_upload],
            outputs=[model_cfg, ckpt_file, vocab_file, model_cfg_upload, ckpt_file_upload, vocab_file_upload]
        )
        generate_btn.click(
            fn=run_tts_inference,
            inputs=[ref_audio, ref_text, gen_text, speed, use_upload, model_cfg, ckpt_file, vocab_file],
            outputs=[output_audio, output_text]
        )
    return demo

if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch()
