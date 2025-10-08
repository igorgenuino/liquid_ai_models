#!/usr/bin/env python3
# Gradio web UI for the SmolVLM2-500M Base Model

import torch
import gradio as gr
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import traceback
import os

# --- 1. CONFIGURATION (MODIFIED) ---
# Updated model name for clarity
MODEL_NAME = "SmolVLM2-500M (Base Model)" 
BASE_MODEL_ID = r".\SmolVLM2-500M-Video-Instruct" 

# --- PATH FOR THE GR.IMAGE COMPONENT ---
project_path = r"."
logo_path = os.path.join(project_path, "logo.png")
favicon_path = os.path.join(project_path, "icon.png")

# --- 2. CUSTOM CSS ---
CUSTOM_CSS = """
.gradio-container {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    font-family: 'Inter', system-ui, sans-serif;
}
.container {
    max-width: 1200px !important;
}
#header-markdown h1, #header-markdown h3 {
    color: white;
    font-weight: 300;
    margin: 0;
    padding: 0;
}
#chatbot {
    height: 600px !important;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    background: rgba(0, 0, 0, 0.3);
}
#chatbot .message {
    font-size: 15px;
    line-height: 1.6;
}
.input-row {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 15px;
    margin-top: 20px;
}
"""

# --- 3. LOAD MODEL & PROCESSOR (MODIFIED) ---
print(f"🚀 Initializing {MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📊 System Info: Device: {device}")

print(f"\n⏳ Loading base model from: {BASE_MODEL_ID}...")
# MODIFIED: Load the model directly into the 'model' variable
model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)


model.eval()
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
print("✅ Base model and processor loaded successfully!\n")

# --- 4. CORE CHAT LOGIC ---
SYSTEM_MESSAGE = (
    "You are a helpful multimodal assistant. "
    "Provide a concise answer based on the image and question."
)
conversation_state = []
current_image = None

def resize_image(image, max_size=512):
    if image.width <= max_size and image.height <= max_size:
        return image
    scale = min(max_size / image.width, max_size / image.height)
    new_width = int(image.width * scale)
    new_height = int(image.height * scale)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

@torch.inference_mode()
def generate_response(
    message, image, max_new_tokens, temperature, top_p,
    repetition_penalty, do_sample, resize_images,
):
    global conversation_state, current_image
    try:
        user_content = []
        if image is not None:
            if not isinstance(image, Image.Image): image = Image.fromarray(image)
            if resize_images: image = resize_image(image)
            current_image = image
            user_content.append({"type": "image", "image": image})
        elif current_image is not None:
            user_content.append({"type": "image", "image": current_image})

        # MODIFIED: More generic default message
        msg_txt = (message or "Describe the image.").strip()
        user_content.append({"type": "text", "text": msg_txt})
        
        conversation_state.append({"role": "user", "content": user_content})
        
        conversation = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]}] + conversation_state
        
        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, return_tensors="pt", return_dict=True, tokenize=True,
        )
        
        if device == "cuda": inputs = {k: v.to(device) for k, v in inputs.items()}
        
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens), "pad_token_id": processor.tokenizer.pad_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": float(top_p), "repetition_penalty": float(repetition_penalty)})
        else:
            gen_kwargs["do_sample"] = False
        
        output_ids = model.generate(**inputs, **gen_kwargs)
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip() or "I could not generate a response."
        
        conversation_state.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        return response, conversation_state
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        print(f"⚠️ {error_msg}\n{traceback.format_exc()}")
        return error_msg, conversation_state

def format_conversation_for_display(conv_state):
    formatted = []
    for i in range(0, len(conv_state), 2):
        user_turn = conv_state[i]
        assistant_turn = conv_state[i+1] if i + 1 < len(conv_state) else None
        user_text = next((c["text"] for c in user_turn["content"] if c["type"] == "text"), "")
        has_image = any(c["type"] == "image" for c in user_turn["content"])
        user_display = f"📷 {user_text}" if has_image and user_text else ("📷 [Image Analyzed]" if has_image else user_text)
        assistant_text = next((c["text"] for c in assistant_turn["content"] if c["type"] == "text"), "") if assistant_turn else ""
        formatted.append((user_display, assistant_text))
    return formatted

def chat(message, image, history, *args):
    response, updated_state = generate_response(message, image, *args)
    history = format_conversation_for_display(updated_state)
    return "", None, history

def clear_chat():
    global conversation_state, current_image
    conversation_state, current_image = [], None
    return None, None, []

# --- 5. BUILD GRADIO INTERFACE ---
with gr.Blocks(title=MODEL_NAME, theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
    
    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            gr.Image(logo_path, height=40, interactive=False, show_label=False, show_download_button=False, container=False, show_fullscreen_button=False)
        with gr.Column(scale=10):
            gr.Markdown(
                f"""
                <div id="header-markdown">
                    <h1>{MODEL_NAME}</h1>
                    <h3>The original model for describing images and video</h3>
                </div>
                """,
            )

    chatbot = gr.Chatbot(label="💬 Conversation", height=600, elem_id="chatbot", show_copy_button=True)
    
    with gr.Group(elem_classes="input-row"):
        with gr.Row():
            with gr.Column(scale=7):
                msg = gr.Textbox(
                    label="✍️ Your Message", placeholder="Upload an image and ask a question, or just click Send to get a description.",
                    lines=3, elem_id="msg-box",
                )
            with gr.Column(scale=3):
                img = gr.Image(label="📷 Upload Image", type="pil", height=150)
        with gr.Row():
            send = gr.Button("🚀 Send / Ask", variant="primary")
            clear = gr.Button("🗑️ Clear Chat", variant="secondary")
    
    with gr.Accordion("⚙️ Generation Settings", open=False):
        max_new_tokens = gr.Slider(16, 512, value=50, step=8, label="Max New Tokens")
        do_sample = gr.Checkbox(value=False, label="Use Sampling")
        resize_images = gr.Checkbox(value=True, label="Resize images to 512px max (recommended)")
        temperature = gr.Slider(0.0, 1.5, value=0.1, step=0.05, label="Temperature")
        top_p = gr.Slider(0.01, 1.0, value=0.95, step=0.01, label="Top-p")
        repetition_penalty = gr.Slider(1.0, 1.5, value=1.0, step=0.01, label="Repetition Penalty")
    
    all_inputs = [msg, img, chatbot, max_new_tokens, temperature, top_p, repetition_penalty, do_sample, resize_images]
    
    send.click(chat, inputs=all_inputs, outputs=[msg, img, chatbot], queue=False)
    clear.click(clear_chat, None, [msg, img, chatbot], queue=False)

# --- 6. LAUNCH THE APP ---
if __name__ == "__main__":   
    demo.queue().launch(server_port=7860, favicon_path=favicon_path)