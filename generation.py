import json
import os
import time

import requests
from dotenv import load_dotenv
from groq import Groq
from langchain.prompts import PromptTemplate

# Import retrieval functions
from retrieval import hybrid_search, keyword_search, semantic_search

# Load environment variables
load_dotenv()

# Configure Groq API with validation
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = None
if not groq_api_key:
    print("ERROR: GROQ_API_KEY not found in environment variables")
else:
    print(f"Configuring Groq with API key: {groq_api_key[:5]}...")
    groq_client = Groq(api_key=groq_api_key)

# Override via .env: GROQ_MODEL=llama-3.1-8b-instant
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Used only if the requested model is not available for your API key
PREFERRED_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
]
_model_cache = {}


def _resolve_groq_model(requested):
    """Return `requested` if your key can use it, else the best available chat model."""
    if requested in _model_cache:
        return _model_cache[requested]

    resolved = requested
    try:
        ids = [m.id for m in groq_client.models.list().data]
        if requested not in ids:
            skip = ("whisper", "guard", "tts", "playai", "orpheus", "safeguard")
            chat_ids = [i for i in ids if not any(k in i for k in skip)]
            fallback = next((m for m in PREFERRED_GROQ_MODELS if m in ids), None)
            if fallback is None and chat_ids:
                fallback = chat_ids[0]
            if fallback:
                print(f"[groq] '{requested}' not available. Using '{fallback}' instead.")
                print(f"[groq] Models available to your key: {ids}")
                resolved = fallback
    except Exception as e:
        print(f"[groq] Could not list models: {e}")

    _model_cache[requested] = resolved
    return resolved

# Define RAG prompt template
RAG_PROMPT_TEMPLATE = """
You are an AI assistant helping answer questions about Retrieval-Augmented Generation (RAG).
Use the following retrieved documents to answer the user's question.
If the retrieved documents don't contain relevant information, say that you don't know.

RETRIEVED DOCUMENTS:
{context}

USER QUESTION:
{question}

YOUR ANSWER (be comprehensive, accurate, and helpful):
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------
def _groq_stream(messages, model_name, generation_config):
    """Generator: yields text chunks from Groq."""
    try:
        print("Starting streaming response generation...")
        t0 = time.time()
        first = True
        stream = groq_client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            **generation_config,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                text = chunk.choices[0].delta.content
                if text:  # only yield non-empty text
                    if first:
                        print(f"[timing] Groq first token: {time.time() - t0:.2f}s")
                        first = False
                    yield text
    except Exception as e:
        import traceback

        error_msg = f"Error with Groq generation: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        yield error_msg


def generate_with_groq(prompt_text, model_name=DEFAULT_GROQ_MODEL, stream=False):
    """Generate response using Groq. Returns a generator if stream=True, else a string."""
    if groq_client is None:
        error_msg = "Error: GROQ_API_KEY is not configured."
        return iter([error_msg]) if stream else error_msg

    try:
        model_name = _resolve_groq_model(model_name)
        print(f"Initializing Groq model: {model_name}")

        # Safety check for prompt length
        if len(prompt_text) > 30000:
            prompt_text = prompt_text[:30000] + "...[truncated due to length]"
            print("Warning: Prompt was truncated to 30000 characters")

        # Groq does not support top_k, so it is removed
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 2048,
        }

        messages = [{"role": "user", "content": prompt_text}]

        if stream:
            return _groq_stream(messages, model_name, generation_config)

        print("Requesting non-streaming response...")
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=messages,
            **generation_config,
        )
        return response.choices[0].message.content or ""

    except Exception as e:
        import traceback

        error_msg = f"Error with Groq generation: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return iter([error_msg]) if stream else error_msg


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
def _ollama_stream(url, data):
    try:
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    if "response" in chunk:
                        yield chunk["response"]
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        yield f"Error generating response with Ollama: {str(e)}"


def generate_with_ollama(prompt_text, model_name="deepseek-r1:1.5b", stream=False):
    """Generate response using Ollama with Deepseek model"""
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": stream,
        "options": {"temperature": 0.7},
    }

    if stream:
        return _ollama_stream(url, data)

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json().get("response", "No response generated")
    except Exception as e:
        return f"Error generating response with Ollama: {str(e)}"


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
def _build_rag_prompt(query, search_type, top_k):
    """Retrieve chunks and return the formatted prompt (or None if no results)."""
    t0 = time.time()
    if search_type == "keyword":
        results = keyword_search(query, top_k=top_k)
    elif search_type == "semantic":
        results = semantic_search(query, top_k=top_k)
    else:  # hybrid
        results = hybrid_search(query, top_k=top_k)

    print(f"[timing] retrieval ({search_type}): {time.time() - t0:.2f}s")

    if not results:
        return None

    contexts = []
    for i, hit in enumerate(results):
        source = hit["_source"]
        content = source.get("content", "")
        content_type = source.get("content_type", "unknown")

        # Add metadata if available
        metadata_info = ""
        if "metadata" in source and source["metadata"]:
            if "caption" in source["metadata"] and source["metadata"]["caption"]:
                metadata_info += f"\nCaption: {source['metadata']['caption']}"

        contexts.append(f"[Document {i+1} - {content_type}]{metadata_info}\n{content}")

    context_text = "\n\n---\n\n".join(contexts)
    return prompt.format(context=context_text, question=query)


def generate_rag_response(
    query, search_type="hybrid", top_k=5, model_type="groq", stream=False
):
    """
    Generate RAG response using retrieved chunks.

    Args:
        query: User query
        search_type: Type of search (keyword, semantic, hybrid)
        top_k: Number of chunks to retrieve
        model_type: Type of model to use (groq, ollama)
        stream: Whether to stream the response

    Returns:
        Generated string, or a generator of text chunks if stream=True
    """
    try:
        prompt_text = _build_rag_prompt(query, search_type, top_k)

        if prompt_text is None:
            message = "No relevant information found. Please try a different search type or refine your question."
            return iter([message]) if stream else message

        if model_type == "groq":
            return generate_with_groq(prompt_text, stream=stream)
        else:  # ollama
            return generate_with_ollama(prompt_text, stream=stream)

    except Exception as e:
        error_message = f"Error in RAG process: {str(e)}"
        return iter([error_message]) if stream else error_message


# For testing
if __name__ == "__main__":
    query = "How does RAG work?"

    # Groq streaming
    print("Groq Streaming Response: ", end="", flush=True)
    for chunk in generate_rag_response(query, "hybrid", 3, "groq", True):
        print(chunk, end="", flush=True)

    # Ollama streaming
    print("\n\nOllama Streaming Response: ", end="", flush=True)
    for chunk in generate_rag_response(query, "hybrid", 3, "ollama", True):
        print(chunk, end="", flush=True)