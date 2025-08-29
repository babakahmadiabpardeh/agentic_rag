import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from utils import ChromaDBManager, FileProcessor, LangChainHelper, Config, AgentManager
from langchain_text_splitters import RecursiveCharacterTextSplitter

def render_sidebar(chromadb_manager, file_processor, config):
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader("Upload your PDF or TXT files", accept_multiple_files=True, type=['pdf', 'txt'])

        processed_files = chromadb_manager.get_processed_files()

        new_files = []
        if uploaded_files:
            st.subheader("Uploaded Files")
            for uploaded_file in uploaded_files:
                if uploaded_file.name in processed_files:
                    st.write(f"- {uploaded_file.name} (Already Processed)")
                else:
                    st.write(f"- {uploaded_file.name}")
                    new_files.append(uploaded_file)

        if st.button("Process", disabled=not new_files) and new_files:
            with st.spinner("Processing documents..."):
                documents = file_processor.process_files(new_files)
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
                document_chunks = text_splitter.split_documents(documents)

                chromadb_manager.ingest_documents(document_chunks, config.chroma_collection_name, config.embedding_model)
                
                for new_file in new_files:
                    chromadb_manager.add_processed_file(new_file.name)

                st.session_state.collection_name = config.chroma_collection_name
                st.success("Documents processed successfully!")

def handle_chat_input(chromadb_manager, langchain_helper, agent_manager, config):
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        if st.session_state.collection_name is None:
            st.session_state.chat_history.append(AIMessage(content="I don't have any documents to reference. Please upload some documents first."))
        else:
            with st.spinner("Thinking..."):
                vector_store = chromadb_manager.get_vector_store(st.session_state.collection_name, config.embedding_model)
                retriever = langchain_helper.get_retriever(vector_store)
                agent_executor = agent_manager.get_agent(retriever)

                response = agent_executor.invoke({
                    "chat_history": st.session_state.chat_history,
                    "input": user_query
                })
                
                answer = response['output']

                st.session_state.chat_history.append(AIMessage(content=answer))

def display_chat_history():
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

def main():
    # App config
    st.set_page_config(page_title="Chat with your documents", page_icon="💬")
    st.title("Chat with your documents")

    # Load configuration
    config = Config()

    # Initialize helpers and managers
    chromadb_manager = ChromaDBManager()
    file_processor = FileProcessor()
    langchain_helper = LangChainHelper(
        config.llm_model,
        config.retriever_k,
        config.system_prompt,
        config.retriever_prompt,
        config.retriever_type
    )
    agent_manager = AgentManager(config.llm_model, config.system_prompt)

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello! I'm your RAG agent. Upload some documents and I'll answer your questions about them.")]
    if "collection_name" not in st.session_state:
        try:
            chromadb_manager.client.get_collection(name=config.chroma_collection_name)
            st.session_state.collection_name = config.chroma_collection_name
        except Exception:
            st.session_state.collection_name = None

    # Render UI and handle logic
    render_sidebar(chromadb_manager, file_processor, config)
    handle_chat_input(chromadb_manager, langchain_helper, agent_manager, config)
    display_chat_history()

if __name__ == "__main__":
    main()