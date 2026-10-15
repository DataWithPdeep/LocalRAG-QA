# 📚 LocalRAG-QA — Intelligent Document Question Answering System

**A Retrieval-Augmented Generation (RAG) application for document-based question answering using Hybrid Search, OpenSearch, Groq and Ollama.**

LocalRAG-QA is an AI-powered question-answering system that retrieves relevant information from indexed documents and generates context-aware answers using Large Language Models (LLMs). It combines keyword-based retrieval and semantic search to improve the relevance of retrieved information.

The project supports both cloud-based inference through Groq and local inference through Ollama, with an interactive Gradio web interface.

## ✨ Key Features

- **Retrieval-Augmented Generation (RAG):** Generates answers using retrieved document context.
- **Keyword Search:** Finds relevant information using text-based matching.
- **Semantic Search:** Retrieves information based on the meaning of a query using embeddings.
- **Hybrid Search:** Combines keyword and semantic retrieval.
- **Multiple LLM Providers:** Supports Groq cloud inference and local Ollama inference.
- **Streaming Responses:** Displays generated answers progressively.
- **PDF Processing:** Extracts and processes document text, tables and images.
- **Table and Image Understanding:** Generates descriptions for extracted visual content using AI models.
- **Vector Search:** Stores embeddings in OpenSearch for similarity-based retrieval.
- **Configurable Retrieval:** Adjust the number of retrieved chunks using the Top-K setting.
- **Interactive UI:** Ask questions and configure retrieval and model settings through Gradio.

## 🏗️ System Architecture

```text
PDF Document
     |
     v
Document Extraction
(Unstructured)
     |
     +------------------+
     |                  |
     v                  v
  Text Chunks       Tables & Images
     |                  |
     |             AI Descriptions
     |                  |
     +--------+---------+
              |
              v
      Generate Embeddings
      (Ollama Embeddings)
              |
              v
       OpenSearch Index
              |
User Question |
      |       |
      v       v
 Keyword Search
 Semantic Search
 Hybrid Search
      |
      v
 Retrieve Top-K Chunks
      |
      v
 Build RAG Prompt
      |
      +--------------------+
      |                    |
      v                    v
   Groq LLM            Ollama LLM
   (Cloud)             (Local)
      |                    |
      +---------+----------+
                |
                v
       Generated Answer
                |
                v
          Gradio UI
```

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| RAG Orchestration | LangChain |
| Document Processing | Unstructured |
| Vector Database and Search | OpenSearch |
| Embeddings | Ollama `nomic-embed-text` |
| Cloud LLM | Groq |
| Local LLM | Ollama with DeepSeek |
| Web Interface | Gradio |
| Containerization | Docker Compose |

## 📁 Project Structure

```text
LocalRAG-QA/
│
├── files/
│   └── 2312.10997v5.pdf
│
├── app.py                # Gradio application
├── chunking.py           # Text, table and image processing
├── generation.py         # RAG pipeline and LLM generation
├── helper.py             # Embeddings and utility functions
├── index.py              # Document indexing pipeline
├── ingection.py          # Chunk preparation and ingestion utilities
├── retrieval.py          # Keyword, semantic and hybrid search
├── mapping.json          # Mapping/configuration data
├── requirements.txt      # Python dependencies
├── docker-compose.yml    # OpenSearch services
├── dev.ipynb             # Development notebook
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/DataWithPdeep/LocalRAG-QA.git
cd LocalRAG-QA
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install groq python-dotenv requests
```

The additional installation command covers imports used by the current application that are not explicitly listed in `requirements.txt`.

### 4. Start OpenSearch

Make sure Docker Desktop is installed and running, then execute:

```bash
docker compose up -d
```

This starts OpenSearch and OpenSearch Dashboards using the repository's Docker Compose configuration.

Verify OpenSearch is running:

```bash
curl http://localhost:9200
```

On Windows PowerShell, you can also open `http://localhost:9200` in your browser.

### 5. Configure Ollama

Install Ollama from [ollama.com](https://ollama.com/) and make sure the Ollama service is running.

Pull the embedding and generation models:

```bash
ollama pull nomic-embed-text
ollama pull deepseek-r1:1.5b
```

The embedding model is used for document and query embeddings. The DeepSeek model is used for local answer generation.

### 6. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Get your API key from [Groq Console](https://console.groq.com/).

Keep your API key private and never commit your `.env` file to GitHub.

For local Ollama generation, select Ollama in the application interface. Embedding generation still requires the configured embedding service.

## 🚀 Run the Application

### Step 1: Index the document

The repository's indexing pipeline processes the sample PDF and indexes extracted content in OpenSearch.

```bash
python index.py
```

The indexing process includes:

1. Extracting document elements from the PDF.
2. Processing text, tables and images.
3. Generating embeddings for indexable content.
4. Storing the resulting records in the `localrag` OpenSearch index.

**Note:** The sample PDF must exist at `files/2312.10997v5.pdf`. PDF processing may also require system dependencies such as Poppler and Tesseract, depending on the extraction operations used.

### Step 2: Launch the Gradio interface

```bash
python app.py
```

Open the local Gradio URL printed in the terminal, typically:

`http://127.0.0.1:7860`

### Step 3: Ask questions

1. Enter a question in the input box.
2. Choose Keyword, Semantic or Hybrid search.
3. Select Groq or Ollama as the answer-generation provider.
4. Set the Top-K value to control how many chunks are retrieved.
5. Enable streaming if you want to see the answer as it is generated.
6. Click **Generate Answer**.

## 🔍 Search Strategies

### Keyword Search

Uses text matching to find relevant indexed content. Useful when a question contains specific technical terms or phrases.

### Semantic Search

Uses embeddings and vector similarity to find content related to the meaning of the question, even when the wording differs.

### Hybrid Search

Combines keyword and semantic retrieval to balance exact-term matching with meaning-based retrieval. This is the recommended default in the application.

## 🤖 LLM Options

| Provider | Description |
|---|---|
| Groq | Cloud-based inference with a configured Groq API key |
| Ollama | Local inference using the configured DeepSeek model |

Groq requires internet access and a valid API key. Ollama requires the local service and model to be available.

## 💡 Example Questions

The included sample document concerns Retrieval-Augmented Generation. You can try questions such as:

- How does RAG work?
- What are the benefits of RAG compared to fine-tuning?
- Explain the architecture of a RAG system.
- What are the common challenges in RAG implementations?

The system retrieves relevant context from the indexed content and uses it to generate an answer.

## 🔧 Troubleshooting

**OpenSearch connection error**

- Confirm Docker Desktop is running.
- Run `docker compose ps` to check service status.
- Verify that `http://localhost:9200` is accessible.

**No relevant information found**

- Run the indexing pipeline.
- Verify that the `localrag` index contains documents.
- Try another search strategy or refine your question.

**Groq API errors**

- Check that `GROQ_API_KEY` is configured correctly.
- Verify your API key, model availability and API limits.

**Ollama connection error**

- Confirm that Ollama is running.
- Run `ollama list` to verify the installed models.
- Check that the model names match the configuration.

**PDF extraction errors**

- Verify the PDF path.
- Install required system dependencies, including Poppler or Tesseract where needed.
- Check the Unstructured extraction logs.

## 🔐 Privacy and Security

- Do not upload API keys or credentials to source control.
- Local Ollama generation can avoid sending prompts to a cloud LLM provider.
- When using Groq or AI-powered external description services, relevant data may be sent to those providers.
- OpenSearch is configured for local development; review authentication and network exposure before deploying it in production.

## 🗺️ Future Improvements

- Support uploading and indexing multiple documents through the UI.
- Add document-level source citations and retrieved-context previews.
- Add persistent conversation history.
- Improve retrieval with reranking and configurable chunking.
- Add automated tests and production deployment configuration.
- Add evaluation metrics for retrieval quality and answer faithfulness.

## 👨‍💻 Author

**Pradeep Singh**

GitHub: [@DataWithPdeep](https://github.com/DataWithPdeep)

Project: [LocalRAG-QA](https://github.com/DataWithPdeep/LocalRAG-QA)

---

If you find this project useful, consider giving the repository a ⭐ on GitHub.

**Built with Python, RAG, OpenSearch and Generative AI.**
