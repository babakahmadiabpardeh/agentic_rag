import streamlit as st
import os
import chromadb
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Get models and collection name from environment variables
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_model_name = os.getenv("LLM_MODEL")
CHROMA_COLLECTION_NAME = "rag-chroma-collection"

# App config
st.set_page_config(page_title="Chat with your documents", page_icon="💬")
st.title("Chat with your documents")


def process_and_ingest_files(uploaded_files, collection_name):
    """Processes uploaded files and ingests them into a Chroma collection."""
    if not uploaded_files:
        return

    documents = []
    temp_dir = "./temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(temp_file_path)
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(temp_file_path)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            continue
        documents.extend(loader.load())

    if not documents:
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(documents)

    # This will create the collection if it doesn't exist, and ingest the documents.
    Chroma.from_documents(
        documents=document_chunks,
        embedding=OllamaEmbeddings(model=embedding_model_name),
        collection_name=collection_name,
        client=chromadb.HttpClient(host="localhost", port=8000)
    )
    st.session_state.collection_name = collection_name

def get_context_retriever_chain(vector_store):
    llm = ChatOllama(model=llm_model_name)
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOllama(model=llm_model_name)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Sidebar for file uploads
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload your PDF or TXT files and click 'Process'", accept_multiple_files=True, type=['pdf', 'txt'])

    if st.button("Process") and uploaded_files:
        with st.spinner("Processing documents..."):
            process_and_ingest_files(uploaded_files, CHROMA_COLLECTION_NAME)
            st.success("Documents processed successfully!")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! I'm your RAG agent. Upload some documents and I'll answer your questions about them.")]
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None

# Chat input from the user
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query.strip() != "":
    if st.session_state.collection_name is None:
        st.warning("Please upload and process documents before asking questions.")
    else:
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.spinner("Thinking..."):
            # Connect to the existing vector store
            client = chromadb.HttpClient(host="localhost", port=8000)
            vector_store = Chroma(
                client=client,
                collection_name=st.session_state.collection_name,
                embedding_function=OllamaEmbeddings(model=embedding_model_name)
            )

            retriever_chain = get_context_retriever_chain(vector_store)
            conversational_rag_chain = get_conversational_rag_chain(retriever_chain)

            response = conversational_rag_chain.invoke({
                "chat_history": st.session_state.chat_history,
                "input": user_query
            })

            st.session_state.chat_history.append(AIMessage(content=response['answer']))

# Display the chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)