import fitz  # PyMuPDF
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
import faiss
import numpy as np
import os
import pickle
import shutil

# --- CONFIGURATION ---
PDF_PATH = "history_of the_automobile.pdf"
RAG_DATA_DIR = "rag_data"
IMAGE_DIR = os.path.join(RAG_DATA_DIR, "images")
INDEX_PATH = os.path.join(RAG_DATA_DIR, "rag_index.faiss")
CONTENT_MAP_PATH = os.path.join(RAG_DATA_DIR, "rag_content.pkl")
EMBEDDING_MODEL = 'clip-ViT-B-32'

def setup_directories():
    """Deletes old data and creates the necessary directories."""
    if os.path.exists(RAG_DATA_DIR):
        print(f"🧹 Deleting existing '{RAG_DATA_DIR}' folder to start fresh.")
        shutil.rmtree(RAG_DATA_DIR)
    os.makedirs(RAG_DATA_DIR)
    os.makedirs(IMAGE_DIR)

def extract_content_from_pdf(pdf_path):
    """
    Extracts text chunks (paragraphs) and images from the PDF.
    Returns a list of content items, where each item is a dictionary.
    """
    print(f"📖 Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    content_items = []
    image_count = 0

    for page_num, page in enumerate(doc):
        print(f"📄 Processing Page {page_num + 1}/{len(doc)}...", end='\r')
        
        # Extract text blocks and chunk them
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4].strip().replace('\n', ' ')
            if len(text) > 40:
                content_items.append({
                    "type": "text",
                    "content": text,
                    "page": page_num + 1
                })

        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                image_filename = f"image_p{page_num + 1}_{img_index}.png"
                image_path = os.path.join(IMAGE_DIR, image_filename)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                content_items.append({
                    "type": "image",
                    "path": image_path,
                    "page": page_num + 1
                })
                image_count += 1
            except Exception as e:
                print(f"\n⚠️ Warning: Could not extract image on page {page_num + 1}. Skipping. Error: {e}")
            
    print(f"\n\n✅ Extracted {len(content_items) - image_count} text chunks and {image_count} images.")
    return content_items

def create_embeddings(content_items, model):
    """
    Creates embeddings for all text and image content items.
    """
    embeddings = []
    print("\n🧠 Creating embeddings (this may take a while)...")

    # --- NEW FIX ---: The correct attribute is '.model', not '.auto_model'.
    embedding_dim = model[0].model.config.projection_dim

    for i, item in enumerate(content_items):
        print(f"   - Embedding item {i+1}/{len(content_items)} ({item['type']})", end='\r')
        if item['type'] == 'text':
            embedding = model.encode(item['content'], convert_to_tensor=True)
        elif item['type'] == 'image':
            try:
                img = Image.open(item['path']).convert("RGB")
                embedding = model.encode(img, convert_to_tensor=True)
            except Exception as e:
                print(f"\n   - ⚠️ Could not process image {item['path']}: {e}")
                embedding = torch.zeros(embedding_dim)
        
        embeddings.append(embedding.cpu().numpy())
        
    print("\n✅ All embeddings created.")
    return np.array(embeddings).astype('float32')


def build_faiss_index(embeddings):
    """
    Builds and saves a FAISS index for similarity search.
    """
    print("\n🗂️ Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print(f"   - Index built with {index.ntotal} vectors.")
    faiss.write_index(index, INDEX_PATH)
    print(f"✅ FAISS index saved to: {INDEX_PATH}")


if __name__ == "__main__":
    print("🚀 Starting RAG preprocessing pipeline...")
    setup_directories()

    print(f"⏳ Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Model loaded.")

    content_data = extract_content_from_pdf(PDF_PATH)
    all_embeddings = create_embeddings(content_data, embedding_model)
    build_faiss_index(all_embeddings)

    with open(CONTENT_MAP_PATH, "wb") as f:
        pickle.dump(content_data, f)
    print(f"✅ Content mapping saved to: {CONTENT_MAP_PATH}")

    print("\n🎉 Preprocessing complete! You are now ready to run the main Gradio app.")