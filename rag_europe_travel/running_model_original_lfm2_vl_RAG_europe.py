#!/usr/bin/env python3
# Enhanced Gradio web UI for LiquidAI/LFM2-VL with Multimodal RAG

import torch
import gradio as gr
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import traceback
import os
# --- RAG ---: Import new libraries for RAG functionality
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURATION (MODIFIED) ---
MODEL_NAME = "LFM2-VL-450M (Base Model) RAG Europe travel guide"
BASE_MODEL_ID = r"..\LFM2-VL-450M"

# --- RAG ---: Configuration for the RAG system
RAG_ENABLED_BY_DEFAULT = True
RAG_DATA_DIR = "rag_data"
INDEX_PATH = os.path.join(RAG_DATA_DIR, "rag_index.faiss")
CONTENT_MAP_PATH = os.path.join(RAG_DATA_DIR, "rag_content.pkl")
EMBEDDING_MODEL_ID = 'clip-ViT-B-32'
TOP_K_RESULTS = 3 # Number of relevant items to retrieve from the PDF

# --- PATH FOR THE GR.IMAGE COMPONENT ---
project_path = r"."
logo_path = os.path.join(project_path, "logo.png")
favicon_path = os.path.join(project_path, "icon.png")

# Custom CSS for modern dark theme
CUSTOM_CSS = """
/* [Your existing CSS remains unchanged] */
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
model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=dtype if device == "cuda" else torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print("✅ Base model loaded successfully.")
model.eval()

processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

if device == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

print("✅ Model and processor loaded successfully!\n")

# --- RAG ---: Load RAG components
rag_index = None
rag_content_map = None
embedding_model = None
if os.path.exists(INDEX_PATH) and os.path.exists(CONTENT_MAP_PATH):
    print("📚 Loading RAG components...")
    try:
        rag_index = faiss.read_index(INDEX_PATH)
        with open(CONTENT_MAP_PATH, "rb") as f:
            rag_content_map = pickle.load(f)
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)
        print("✅ RAG components loaded successfully.")
        print(f"   - Index contains {rag_index.ntotal} vectors.")
    except Exception as e:
        print(f"⚠️ Could not load RAG components: {e}")
        print("   - RAG will be disabled.")
else:
    print("⚠️ RAG data not found. Please run `preprocess_pdf.py` first.")
    print("   - RAG will be disabled for this session.")


# Store conversation state and images
conversation_state = []
current_image = None

def resize_image(image, max_size=512):
    """
    Resize image to fit within max_size while maintaining aspect ratio
    """
    if image.width <= max_size and image.height <= max_size:
        return image
    
    scale = min(max_size / image.width, max_size / image.height)
    new_width = int(image.width * scale)
    new_height = int(image.height * scale)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

# --- RAG ---: Function to search the RAG database
def search_rag_database(query_text, query_image, k=TOP_K_RESULTS):
    """
    Searches the FAISS index for the most relevant text and images.
    """
    if rag_index is None or embedding_model is None:
        return []

    query_embeddings = []
    if query_text:
        query_embeddings.append(embedding_model.encode(query_text))
    if query_image:
        query_embeddings.append(embedding_model.encode(query_image))

    if not query_embeddings:
        return []

    # Average the embeddings if both text and image are provided
    final_query_embedding = np.mean(query_embeddings, axis=0).reshape(1, -1).astype('float32')

    # Search the FAISS index
    distances, indices = rag_index.search(final_query_embedding, k)
    
    # Retrieve the content
    results = [rag_content_map[i] for i in indices[0]]
    return results

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
    use_rag, # --- RAG ---: New parameter from the UI
):
    """
    Generate response from the model, with optional RAG augmentation.
    """
    global conversation_state, current_image
    
    try:
        # --- RAG ---: Perform search and augment the message if RAG is enabled
        rag_context = ""
        retrieved_images = []
        if use_rag and (rag_index is not None):
            print("🔍 Performing RAG search...")
            search_results = search_rag_database(message, image)
            if search_results:
                context_texts = []
                for result in search_results:
                    if result['type'] == 'text':
                        context_texts.append(f"- (Page {result['page']}): {result['content']}")
                    elif result['type'] == 'image':
                        retrieved_images.append(Image.open(result['path']))
                
                rag_context = "Based on the provided document, a travel guide on europe, here is some relevant context:\n"
                rag_context += "\n".join(context_texts)
                if retrieved_images:
                    rag_context += "\nI have also found some relevant images from the document."
                print(f"✅ RAG Context found:\n{rag_context[:300]}...")

        # Build user content for the conversation
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

        # --- RAG ---: Add retrieved images to the user content
        for img in retrieved_images:
            user_content.append({"type": "image", "image": img})

        # --- RAG ---: Prepend RAG context to the user's text message
        augmented_message = (rag_context + "\n\n" + (message or "")).strip()

        if augmented_message:
            user_content.append({"type": "text", "text": augmented_message})
        elif not user_content:
            return "Please provide a message or image.", conversation_state
        
        conversation_state.append({"role": "user", "content": user_content})
        
        # [The rest of the generation logic remains largely the same]
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful multimodal assistant by Liquid AI. If context from a document is provided, use it to form your answer."}]}
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
    # [This function remains unchanged]
    formatted = []
    i = 0
    while i < len(conversation_state):
        if i < len(conversation_state) and conversation_state[i]["role"] == "user":
            user_text = ""
            has_image = False
            for content in conversation_state[i]["content"]:
                if content["type"] == "text":
                    # --- RAG ---: Don't show the augmented context in the chat history
                    if "Based on the provided document" in content["text"]:
                        # Extract the original message
                        parts = content["text"].split("\n\n")
                        user_text = parts[-1]
                    else:
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
                user_display = "📷 [Image(s) uploaded]"
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
with gr.Blocks(title="LFM2-VL Base with RAG", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
    # [Header Markdown remains unchanged]
    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            gr.Image(logo_path, height=40, interactive=False, show_label=False, show_download_button=False, container=False, show_fullscreen_button=False)
        with gr.Column(scale=10):
            gr.Markdown(
                f"""
                <div id="header-markdown">
                    <h1>{MODEL_NAME}</h1>
                    <h3>The original model to describe images, now with RAG!</h3>
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
                # --- RAG ---: Add the checkbox to enable/disable RAG
                use_rag_checkbox = gr.Checkbox(
                    value=RAG_ENABLED_BY_DEFAULT and (rag_index is not None),
                    label="📚 Use Document RAG (Search Travel Guide Europe PDF)",
                    interactive=(rag_index is not None)
                )

    # All the parameters for the generate function
    generation_params = [
        max_new_tokens, temperature, top_p, min_p, repetition_penalty,
        do_sample, min_image_tokens, max_image_tokens, do_image_splitting,
        resize_images, use_rag_checkbox  # --- RAG ---: Add the checkbox value
    ]

    # Event handlers
    msg.submit(
        chat,
        inputs=[msg, img, chatbot] + generation_params,
        outputs=[msg, img, chatbot],
        queue=False,
    )
    
    send.click(
        chat,
        inputs=[msg, img, chatbot] + generation_params,
        outputs=[msg, img, chatbot],
        queue=False,
    )
    
    clear.click(clear_chat, None, [msg, img, chatbot], queue=False)

# --- LAUNCH THE APP ---
if __name__ == "__main__":   
    demo.queue().launch(server_port=7863, favicon_path=favicon_path)