import os
import chromadb
import yaml
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

class Config:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.embedding_model = config.get("embedding_model", "all-minilm")
        self.llm_model = config.get("llm_model", "phi3")
        self.chroma_collection_name = config.get("chroma_collection_name", "rag-chroma-collection")
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.retriever_k = config.get("retriever_k", 4)
        self.retriever_type = config.get("retriever_type", "mmr")
        self.system_prompt = config.get("system_prompt", "Answer the user's questions based on the below context. Also provide the source document for each piece of information:\n\n{context}")
        self.retriever_prompt = config.get("retriever_prompt", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")

class ChromaDBManager:
    def __init__(self, host="localhost", port=8000):
        self.client = chromadb.HttpClient(host=host, port=port)
        self.processed_files_collection_name = "processed_files"
        self.processed_files_collection = self.client.get_or_create_collection(name=self.processed_files_collection_name)

    def get_processed_files(self):
        try:
            processed_files_data = self.processed_files_collection.get()
            return set(processed_files_data['ids'])
        except Exception:
            return set()

    def add_processed_file(self, filename):
        self.processed_files_collection.add(
            ids=[filename],
            documents=[filename]
        )

    def get_vector_store(self, collection_name, embedding_model_name):
        return Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=OllamaEmbeddings(model=embedding_model_name)
        )

    def ingest_documents(self, documents, collection_name, embedding_model_name):
        Chroma.from_documents(
            documents=documents,
            embedding=OllamaEmbeddings(model=embedding_model_name),
            collection_name=collection_name,
            client=self.client
        )

class FileProcessor:
    def __init__(self, temp_dir="./data"):
        self.temp_dir = temp_dir
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def process_files(self, uploaded_files):
        documents = []
        for uploaded_file in uploaded_files:
            temp_file_path = os.path.join(self.temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(temp_file_path)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(temp_file_path)
            else:
                continue
            
            loaded_documents = loader.load()
            for doc in loaded_documents:
                doc.metadata = {"source": uploaded_file.name}
            documents.extend(loaded_documents)
        return documents

class LangChainHelper:
    def __init__(self, llm_model_name, retriever_k, system_prompt, retriever_prompt, retriever_type):
        self.llm = ChatOllama(model=llm_model_name)
        self.retriever_k = retriever_k
        self.system_prompt = system_prompt
        self.retriever_prompt = retriever_prompt
        self.retriever_type = retriever_type

    def get_retriever(self, vector_store):
        if self.retriever_type == "mmr":
            return vector_store.as_retriever(search_type="mmr", search_kwargs={'k': self.retriever_k})
        elif self.retriever_type == "contextual_compression":
            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=vector_store.as_retriever(search_kwargs={'k': self.retriever_k})
            )
            return compression_retriever
        else:
            return vector_store.as_retriever(search_kwargs={'k': self.retriever_k})

class AgentManager:
    def __init__(self, llm_model_name, system_prompt):
        self.llm = ChatOllama(model=llm_model_name)
        self.system_prompt = system_prompt

    def get_agent(self, retriever):
        tool = create_retriever_tool(
            retriever,
            "document_retriever",
            "This tool is for retrieving documents."
        )
        tools = [tool]
        
        template = '''Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum and keep the answer as concise as possible. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''
        
        prompt = PromptTemplate.from_template(template)
        
        agent = create_react_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        return agent_executor