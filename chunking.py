import os
import re
import time
import requests
from dotenv import load_dotenv
from unstructured.documents.elements import Image, FigureCaption, Table
from unstructured.partition.pdf import partition_pdf


POPPLER_PATH = r"C:\Users\psran\Downloads\poppler-26.09.0\Library\bin"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR"
TESSERACT_EXE = os.path.join(TESSERACT_PATH, "tesseract.exe")

os.environ["PATH"] = os.pathsep.join([
    POPPLER_PATH,
    TESSERACT_PATH,
    os.environ.get("PATH", "")
])


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_VISION_MODEL = "gemma3:4b"
OLLAMA_TEXT_MODEL = "gemma3:4b"


def groq_generate(prompt, image_b64=None, retries=5):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing in .env")

    if image_b64:
        model = GROQ_VISION_MODEL
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ]
    else:
        model = GROQ_TEXT_MODEL
        content = prompt

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1000,
    }

    for attempt in range(retries):
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=120)

        if r.status_code == 429:
            wait = 5 * (attempt + 1)
            print(f"Groq rate limit reached, waiting {wait} seconds...", flush=True)
            time.sleep(wait)
            continue

        if r.status_code != 200:
            print(f"Groq error {r.status_code}: {r.text[:300]}", flush=True)
            return None

        return r.json()["choices"][0]["message"]["content"]

    return None


def ollama_generate(prompt, image_b64=None):
    data = {
        "model": OLLAMA_VISION_MODEL if image_b64 else OLLAMA_TEXT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 1000},
    }
    if image_b64:
        data["images"] = [image_b64]

    r = requests.post(OLLAMA_URL, json=data, timeout=300)
    r.raise_for_status()
    text = r.json().get("response", "")
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate(prompt, image_b64=None, provider="groq", fallback=True):
    if provider == "ollama":
        try:
            return ollama_generate(prompt, image_b64)
        except Exception as e:
            print(f"Ollama error: {type(e).__name__}: {e}", flush=True)
            return None

    try:
        out = groq_generate(prompt, image_b64)
        if out:
            return out
    except Exception as e:
        print(f"Groq error: {type(e).__name__}: {e}", flush=True)

    if fallback:
        print("No response from Groq, switching to Ollama...", flush=True)
        try:
            return ollama_generate(prompt, image_b64)
        except Exception as e:
            print(f"Ollama error: {type(e).__name__}: {e}", flush=True)
    return None


def process_images_with_caption(raw_chunks, provider="groq"):
    processed_images = []

    for idx, chunk in enumerate(raw_chunks):
        if not isinstance(chunk, Image):
            continue

        print(f"Processing image {idx + 1}...", flush=True)

        caption = None
        if (
            idx + 1 < len(raw_chunks)
            and isinstance(raw_chunks[idx + 1], FigureCaption)
        ):
            caption = raw_chunks[idx + 1].text

        image_data = {
            "caption": caption or "No caption",
            "image_text": chunk.text or "",
            "base64_image": getattr(chunk.metadata, "image_base64", None),
            "content": chunk.text or "",
            "content_type": "image",
            "filename": getattr(chunk.metadata, "filename", None),
        }

        image_base64 = image_data["base64_image"]
        if not image_base64:
            print("Image payload missing; skipping.", flush=True)
            processed_images.append(image_data)
            continue

        prompt = (
            "Describe this image in detail. Explain its "
            "important elements, text, and insights.\n"
            f"Caption: {image_data['caption']}\n"
            f"Image text: {image_data['image_text']}"
        )

        result = generate(prompt, image_b64=image_base64, provider=provider)
        if result:
            image_data["content"] = result

        processed_images.append(image_data)
        time.sleep(2)

    return processed_images


def process_table_with_description(raw_chunks, provider="groq"):
    process_table = []

    for idx, element in enumerate(raw_chunks):
        if not isinstance(element, Table):
            continue

        print(f"Processing table {idx + 1}...", flush=True)

        table_data = {
            "table_as_html": element.metadata.text_as_html,
            "table_text": element.text,
            "content": element.text,
            "content_type": "table",
            "filename": element.metadata.filename,
        }

        html = (table_data["table_as_html"] or element.text)[:12000]

        prompt = (
            "Analyze the following HTML table and provide a detailed description "
            "of its contents, including its structure, key data points, "
            "notable trends, and insights.\n\n"
            f"Table in HTML format:\n{html}\n\n"
            "Analyze the table directly without additional introductory text."
        )

        result = generate(prompt, provider=provider)
        if result:
            table_data["content"] = result

        process_table.append(table_data)
        time.sleep(2)

    return process_table


def create_semantic_chunks(text_chunks):
    from unstructured.documents.elements import CompositeElement

    processed_chunk = []
    for chunk in text_chunks:
        if isinstance(chunk, CompositeElement):
            chunk_data = {
                "content": chunk.text,
                "content_type": "text",
                "filename": chunk.metadata.filename if chunk.metadata else None,
            }
            processed_chunk.append(chunk_data)

    return processed_chunk


if __name__ == "__main__":

    pdf_file_path = os.path.abspath("files/2312.10997v5.pdf")

    if not os.path.isfile(pdf_file_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_file_path}")

    print("Starting PDF partitioning...", flush=True)

    raw_chunks = partition_pdf(
        filename=pdf_file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
    )

    print(f"Extracted {len(raw_chunks)} elements.", flush=True)

    processed_images = process_images_with_caption(raw_chunks, provider="groq")
    print(f"Processed {len(processed_images)} images.", flush=True)
    for image in processed_images:
        print(image)

    process_tables = process_table_with_description(raw_chunks, provider="groq")
    for table in process_tables:
        print(table)

    text_chunk = partition_pdf(
        filename=pdf_file_path,
        strategy="hi_res",
        chunking_strategy="by_title",
        max_characters=2000,
        combine_text_under_n_chars=500,
        new_after_n_chars=1500,
    )

    semantic_chunks = create_semantic_chunks(text_chunk)
    for chunk in semantic_chunks:
        print(chunk)