import argparse
from pathlib import Path

import gradio as gr
import yaml
from PIL import Image

from src.model import MiniVLM


def build_app(checkpoint, stage, load_in_4bit=False):
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    resample_tokens = cfg.get("model", {}).get("resample_tokens")
    model = MiniVLM(
        cfg["model"]["vision_model"],
        cfg["model"]["language_model"],
        use_lora=False,
        lora_r=cfg["training"]["lora_r"],
        lora_alpha=cfg["training"]["lora_alpha"],
        lora_dropout=cfg["training"]["lora_dropout"],
        resample_tokens=resample_tokens,
        load_in_4bit=load_in_4bit,
    )
    if stage == "stage1":
        model.load_projector(checkpoint)
    else:
        model.load_projector(Path(checkpoint) / "projector.pt")
        model.load_lora(Path(checkpoint) / "lora")
    model.eval()

    def user_chat(image, user_message, chat_history):
        if image is None:
            return chat_history, "Please upload an image first!"
        if not user_message or not user_message.strip():
            return chat_history, ""

        chat_history = chat_history or []
        answer = model.answer(
            image=image,
            question=user_message,
            history=chat_history,
            max_new_tokens=cfg["model"]["max_new_tokens"],
        )
        chat_history.append((user_message, answer))
        return chat_history, ""

    with gr.Blocks(title="Mini Vision-Language Model") as demo:
        gr.Markdown(
            "# 🤖 Mini Vision-Language Model (LLaVA-Style)\n"
            "Upload an image and have a multi-turn conversation with the model."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Image")
                clear_btn = gr.Button("🗑️ Clear Conversation", variant="secondary")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=420)
                with gr.Row():
                    txt_input = gr.Textbox(
                        label="Your Question",
                        placeholder="What do you see in this image?",
                        scale=4,
                        show_label=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

        send_btn.click(
            user_chat,
            inputs=[image_input, txt_input, chatbot],
            outputs=[chatbot, txt_input],
        )
        txt_input.submit(
            user_chat,
            inputs=[image_input, txt_input, chatbot],
            outputs=[chatbot, txt_input],
        )
        clear_btn.click(
            lambda: ([], ""),
            outputs=[chatbot, txt_input],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/stage2")
    parser.add_argument("--stage", choices=["stage1", "stage2"], default="stage2")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load Qwen in 4-bit precision")
    args = parser.parse_args()
    build_app(args.checkpoint, args.stage, load_in_4bit=args.load_in_4bit).launch()
