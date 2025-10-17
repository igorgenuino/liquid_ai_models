#!/usr/bin/env python3
# Enhanced Gradio web UI for LiquidAI/LFM2-VL (Base Model)

import torch
import gradio as gr
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import traceback
import os

# --- 1. CONFIGURATION (MODIFIED) ---
# Updated model name for clarity
MODEL_NAME = "LFM2-VL-450M (Base Model)"
# This is now the only model we will load
BASE_MODEL_ID = r".\LFM2-VL-450M"
# REMOVED: ADAPTER_ID is no longer necessary

# --- PATH FOR THE GR.IMAGE COMPONENT ---
project_path = r"."
logo_path = os.path.join(project_path, "logo.png")
favicon_path = os.path.join(project_path, "icon.png")

# Custom CSS for modern dark theme
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
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
dtype = torch.bfloat16 if use_bf16 else torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"📊 System Info:")
print(f"  • Device: {device}")
print(f"  • Dtype: {dtype}")
print(f"  • CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  • GPU: {torch.cuda.get_device_name(0)}")

print(f"\n⏳ Loading base model from: {BASE_MODEL_ID}...")
# MODIFIED: The model is now loaded directly, without a 'base_model' intermediate variable.
model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=dtype if device == "cuda" else torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

print("✅ Base model loaded successfully.")

model.eval()

# The processor is loaded from the base model's directory
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

# Enable TF32 for faster inference on Ampere GPUs
if device == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

print("✅ Model loaded successfully!\n")

# Store conversation state and images
conversation_state = []
current_image = None

def resize_image(image, max_size=512):
    """
    Resize image to fit within max_size while maintaining aspect ratio
    """
    if image.width <= max_size and image.height <= max_size:
        return image
    
    # Calculate the scaling factor
    scale = min(max_size / image.width, max_size / image.height)
    new_width = int(image.width * scale)
    new_height = int(image.height * scale)
    
    # Use high-quality resampling
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

@torch.inference_mode()
def generate_response(
    message,
    image,
    max_new_tokens,
    temperature,
    top_p,
    min_p,
    repetition_penalty,
    do_sample,
    min_image_tokens,
    max_image_tokens,
    do_image_splitting,
    resize_images,
):
    """
    Generate response from the model
    """
    global conversation_state, current_image
    
    try:
        user_content = []
        if image is not None:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            
            if resize_images:
                original_size = image.size
                image = resize_image(image)
                print(f"📸 Resized image from {original_size} to {image.size}")
            else:
                print(f"📸 Processing image: {image.size}")
            
            current_image = image
            user_content.append({"type": "image", "image": image})
        elif current_image is not None and len(conversation_state) > 0:
            user_content.append({"type": "image", "image": current_image})
        
        msg_txt = (message or "").strip()
        if msg_txt:
            user_content.append({"type": "text", "text": msg_txt})
        elif not user_content:
            return "Please provide a message or image.", conversation_state
        
        conversation_state.append({"role": "user", "content": user_content})
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful multimodal assistant by Liquid AI."}]}
        ] + conversation_state
        
        processor_kwargs = {}
        if hasattr(processor, 'image_processor'):
            processor_kwargs.update({
                "min_image_tokens": min_image_tokens,
                "max_image_tokens": max_image_tokens,
                "do_image_splitting": do_image_splitting,
            })
        
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
            **processor_kwargs
        )
        
        if device == "cuda":
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "pad_token_id": processor.tokenizer.pad_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }
        
        if do_sample:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "min_p": float(min_p),
                "repetition_penalty": float(repetition_penalty),
            })
        else:
            gen_kwargs["do_sample"] = False
        
        print(f"🤖 Generating with params: max_tokens={max_new_tokens}, temp={temperature}, sampling={do_sample}")
        
        output_ids = model.generate(**inputs, **gen_kwargs)
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        if not response:
            response = "I apologize, but I couldn't generate a response. Please try again."
        
        print(f"✅ Generated response: {response[:100]}...")
        
        conversation_state.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })
        
        return response, conversation_state
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"⚠️ {error_msg}\n{traceback.format_exc()}")
        return error_msg, conversation_state

def format_conversation_for_display(conversation_state):
    formatted = []
    i = 0
    while i < len(conversation_state):
        if i < len(conversation_state) and conversation_state[i]["role"] == "user":
            user_text = ""
            has_image = False
            for content in conversation_state[i]["content"]:
                if content["type"] == "text":
                    user_text = content["text"]
                elif content["type"] == "image":
                    has_image = True
            
            assistant_text = ""
            if i + 1 < len(conversation_state) and conversation_state[i + 1]["role"] == "assistant":
                for content in conversation_state[i + 1]["content"]:
                    if content["type"] == "text":
                        assistant_text = content["text"]
                i += 2
            else:
                i += 1
            
            if has_image and user_text:
                user_display = f"📷 {user_text}"
            elif has_image:
                user_display = "📷 [Image uploaded]"
            else:
                user_display = user_text
            
            formatted.append([user_display, assistant_text])
        else:
            i += 1
    
    return formatted

def chat(message, image, history, *args):
    response, updated_state = generate_response(message, image, *args)
    history = format_conversation_for_display(updated_state)
    return "", None, history

def clear_chat():
    global conversation_state, current_image
    conversation_state = []
    current_image = None
    print("🗑️ Clearing chat history")
    return None, None, []

# --- Build Gradio Interface ---
with gr.Blocks(title="LFM2-VL Base", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            gr.Image(logo_path, height=40, interactive=False, show_label=False, show_download_button=False, container=False, show_fullscreen_button=False)
        with gr.Column(scale=10):
            gr.Markdown(
                f"""
                <div id="header-markdown">
                    <h1>{MODEL_NAME}</h1>
                    <h3>The original model to describe images</h3>
                </div>
                """,
            )

    chatbot = gr.Chatbot(
        label="💬 Conversation",
        height=600,
        elem_id="chatbot",
        show_copy_button=True,
        type="tuples",
    )
    
    with gr.Group(elem_classes="input-row"):
        with gr.Row():
            with gr.Column(scale=7):
                msg = gr.Textbox(
                    label="✍️ Your Message",
                    placeholder="Ask me anything about the image or just chat...",
                    lines=3,
                    elem_id="msg-box",
                    max_lines=10,
                )
            with gr.Column(scale=3):
                img = gr.Image(
                    label="📷 Upload Image (Optional)",
                    type="pil",
                    elem_classes="image-upload",
                    height=150,
                )
        
        with gr.Row():
            send = gr.Button("🚀 Send", variant="primary", elem_id="send-btn", scale=1)
            clear = gr.Button("🗑️ Clear Chat", variant="secondary", scale=1)
    
    with gr.Accordion("⚙️ Generation Settings", open=False, elem_classes="accordion"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Text Generation")
                max_new_tokens = gr.Slider(16, 2048, value=256, step=16, label="Max New Tokens")
                do_sample = gr.Checkbox(value=True, label="Use Sampling")
                
                temperature = gr.Slider(0.0, 2.0, value=0.1, step=0.05, label="Temperature")
                top_p = gr.Slider(0.01, 1.0, value=0.95, step=0.01, label="Top-p")
                min_p = gr.Slider(0.0, 0.5, value=0.15, step=0.01, label="Min-p")
                repetition_penalty = gr.Slider(1.0, 1.5, value=1.05, step=0.01, label="Repetition Penalty")
            
            with gr.Column():
                gr.Markdown("### Vision Processing")
                resize_images = gr.Checkbox(value=True, label="Resize images to 512×512 max (faster)")
                min_image_tokens = gr.Slider(16, 512, value=64, step=16, label="Min Image Tokens")
                max_image_tokens = gr.Slider(64, 1024, value=256, step=16, label="Max Image Tokens")
                do_image_splitting = gr.Checkbox(value=True, label="Enable Image Splitting (for large images)")
    
    # Event handlers
    msg.submit(
        chat,
        inputs=[
            msg, img, chatbot,
            max_new_tokens, temperature, top_p, min_p, repetition_penalty,
            do_sample, min_image_tokens, max_image_tokens, do_image_splitting,
            resize_images
        ],
        outputs=[msg, img, chatbot],
        queue=False,
    )
    
    send.click(
        chat,
        inputs=[
            msg, img, chatbot,
            max_new_tokens, temperature, top_p, min_p, repetition_penalty,
            do_sample, min_image_tokens, max_image_tokens, do_image_splitting,
            resize_images
        ],
        outputs=[msg, img, chatbot],
        queue=False,
    )
    
    clear.click(clear_chat, None, [msg, img, chatbot], queue=False)

# --- LAUNCH THE APP ---
if __name__ == "__main__":   
    demo.queue().launch(server_port=7862
    , favicon_path=favicon_path)