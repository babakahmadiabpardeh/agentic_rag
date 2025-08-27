# Agentic RAG with Streamlit and Ollama

This project implements a Retrieval-Augmented Generation (RAG) application using a Streamlit web interface, LangChain for orchestration, Ollama for local LLM and embedding models, and ChromaDB as the vector store.

## Features

- **Interactive UI**: A user-friendly web interface built with Streamlit.
- **Dynamic File Ingestion**: Upload PDF and text documents directly through the UI.
- **Local & Private**: Uses local models via Ollama, ensuring your data remains private.
- **Easy Setup**: Leverages Docker Compose for a simple, one-command setup of the ChromaDB vector store.

## Project Structure

```
.
├── .env
├── docker-compose.yml
├── main.py
├── README.md
└── requirements.txt
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- [Ollama](https://ollama.com/) installed and running locally.

### Step 1: Get the Required Local Models

1.  Pull the default models required by the application. Open your terminal and run:

    ```bash
    ollama pull nomic-embed-text
    ollama pull llama3
    ```

2.  If you wish to use different models, you can pull them and update the `.env` file accordingly.

### Step 2: Clone and Set Up the Environment

1.  **Set up the Vector Store**: Start the ChromaDB service using Docker Compose. This will start the database in the background and store its data in a `./chroma_db` directory.

    ```bash
    docker-compose up -d
    ```

2.  **Create a Virtual Environment**: It's recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**: Install the required Python packages.

    ```bash
    pip install -r requirements.txt
    ```

### Step 3: Configure Your Models

1.  The project uses an `.env` file to manage the Ollama model names.
2.  If you want to use different models from the default, edit the `.env` file:

    ```
    EMBEDDING_MODEL=your-embedding-model
    LLM_MODEL=your-llm-model
    ```

## How to Run the Application

1.  Ensure your ChromaDB container is running (`docker-compose up -d`).
2.  Run the Streamlit application:

    ```bash
    streamlit run main.py
    ```

3.  Open your web browser to the local URL provided by Streamlit.

## How to Use the App

1.  Use the sidebar to upload one or more `.pdf` or `.txt` files.
2.  Wait for the processing spinner to complete. This indicates the documents have been embedded and stored in ChromaDB.
3.  Type your question into the chat input at the bottom of the page and press Enter.
4.  The agent will use the content of your uploaded documents to answer the question.
