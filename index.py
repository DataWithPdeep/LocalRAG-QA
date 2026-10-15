import os
import hashlib

from opensearchpy.helpers import bulk

from helper import get_embedding, get_opensearch_client
from chunking import (
    process_images_with_caption,
    process_table_with_description,
    create_semantic_chunks,
)

from unstructured.partition.pdf import partition_pdf


PDF_FILE_PATH = os.path.abspath("files/2312.10997v5.pdf")
INDEX_NAME = "localrag"


def get_token_count_safe(text):
    """Return token count, with a simple fallback."""
    try:
        from helper import get_token_count
        return get_token_count(text)
    except Exception:
        return len(text.split())


def make_document_id(filename, content_type, content):
    """Generate a stable ID to avoid duplicate documents on reruns."""
    raw = f"{filename}|{content_type}|{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def prepare_chunks(raw_chunks, image_chunks, table_chunks, text_chunks):
    """Convert extracted text, table, and image results to indexable documents."""
    documents = []

    # Text chunks
    for chunk in text_chunks:
        content = str(chunk.get("content", "")).strip()

        if content:
            documents.append({
                "content": content,
                "content_type": "text",
                "filename": os.path.basename(PDF_FILE_PATH),
            })

    # Table descriptions
    for table in table_chunks:
        content = str(
            table.get("description")
            or table.get("content")
            or table.get("table_text")
            or ""
        ).strip()

        if content:
            documents.append({
                "content": content,
                "content_type": "table",
                "filename": os.path.basename(PDF_FILE_PATH),
            })

    # Image descriptions
    for image in image_chunks:
        content = str(
            image.get("description")
            or image.get("content")
            or image.get("caption")
            or ""
        ).strip()

        if content:
            documents.append({
                "content": content,
                "content_type": "image",
                "filename": os.path.basename(PDF_FILE_PATH),
            })

    # Remove duplicate chunks
    unique_documents = []
    seen = set()

    for document in documents:
        key = (
            document["content_type"],
            document["content"],
        )

        if key not in seen:
            seen.add(key)
            unique_documents.append(document)

    return unique_documents


def index_documents(documents):
    client = get_opensearch_client("localhost", 9200)

    actions = []

    for position, document in enumerate(documents, start=1):
        content = document["content"]

        print(
            f"Creating embedding {position}/{len(documents)} "
            f"({document['content_type']})...",
            flush=True,
        )

        embedding = get_embedding(content)

        if not embedding:
            print(f"Skipping chunk {position}: embedding is empty.")
            continue

        if len(embedding) != 768:
            raise ValueError(
                f"Expected 768 embedding dimensions, got {len(embedding)}"
            )

        document_id = make_document_id(
            document["filename"],
            document["content_type"],
            content,
        )

        actions.append({
            "_op_type": "index",
            "_index": INDEX_NAME,
            "_id": document_id,
            "_source": {
                "content": content,
                "content_type": document["content_type"],
                "filename": document["filename"],
                "token_count": get_token_count_safe(content),
                "embedding": embedding,
            },
        })

    if not actions:
        print("No documents to index.")
        return

    success, errors = bulk(
        client,
        actions,
        raise_on_error=False,
    )

    print(f"Successfully indexed: {success}")

    if errors:
        print("Some documents failed to index:")
        for error in errors[:5]:
            print(error)

    client.indices.refresh(index=INDEX_NAME)

    count = client.count(index=INDEX_NAME)["count"]
    print(f"Total documents in {INDEX_NAME}: {count}")


def main():
    if not os.path.isfile(PDF_FILE_PATH):
        raise FileNotFoundError(
            f"PDF file not found: {PDF_FILE_PATH}"
        )

    print("Starting PDF partitioning...", flush=True)

    raw_chunks = partition_pdf(
        filename=PDF_FILE_PATH,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
    )

    print(f"Extracted {len(raw_chunks)} elements.", flush=True)

    print("Processing images...", flush=True)
    image_chunks = process_images_with_caption(
        raw_chunks,
        provider="groq",
    )

    print("Processing tables...", flush=True)
    table_chunks = process_table_with_description(
        raw_chunks,
        provider="groq",
    )

    print("Creating text chunks...", flush=True)
    text_elements = partition_pdf(
        filename=PDF_FILE_PATH,
        strategy="hi_res",
        chunking_strategy="by_title",
        max_characters=2000,
        combine_text_under_n_chars=500,
        new_after_n_chars=1500,
    )

    text_chunks = create_semantic_chunks(text_elements)

    documents = prepare_chunks(
        raw_chunks,
        image_chunks,
        table_chunks,
        text_chunks,
    )

    print(f"Prepared {len(documents)} unique chunks.", flush=True)

    index_documents(documents)


if __name__ == "__main__":
    main()