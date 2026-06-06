import streamlit as st
import os
import tempfile
from langchain_mistralai import ChatMistralAI # llm model 
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate # the prompt template for llm 
from langchain_mistralai import MistralAIEmbeddings # embedding model 
# from langchain_community.document_loaders import TextLoader # to load the text file >> depracted in future 
from langchain_community.document_loaders import PyPDFLoader # to load the pdf files 
from langchain_text_splitters import RecursiveCharacterTextSplitter # text splitter 
from langchain_classic.retrievers.multi_query import MultiQueryRetriever # for variations 
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
## this changes is related to streamlit cloud 
mistral_api_key = st.secrets.get("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY")

if not mistral_api_key:
    st.error("🔑 Mistral API Key not found! Please configure it in your Streamlit Secrets or .env file.")
    st.stop()
# --- Page Config ---
st.set_page_config(page_title="PDF Chat Assistant", layout="centered")
st.title("📄 PDF Chatbot with Mistral AI")

# --- Initialize Models (Cached to avoid reloading) ---
@st.cache_resource
def init_models():
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0.2,api_key=mistral_api_key)
    embeddings = MistralAIEmbeddings(model="mistral-embed",api_key=mistral_api_key)
    return llm, embeddings

llm, embeddings = init_models()

# --- App State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file and st.button("Process PDF"):
        with st.spinner("Processing PDF and building vector store..."):
            # 1. Save uploaded file to a temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # 2. Load and Split
            loader = PyPDFLoader(file_path=tmp_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)

            # 3. Create Vector Store
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                collection_name="streamlit_rag"
            )
            st.session_state.vector_store = vector_store
            os.remove(tmp_path) # Clean up temp file
            st.success("PDF Ready!")

# --- Main Chat Interface ---

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if query := st.chat_input("Ask something about your PDF..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.vector_store is None:
        st.error("Please upload and process a PDF first!")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1. Setup Retriever
                base_retriever = st.session_state.vector_store.as_retriever(
                    search_type="mmr", 
                    search_kwargs={"k": 2}
                )
                multi_query_retriever = MultiQueryRetriever.from_llm(
                    retriever=base_retriever, 
                    llm=llm
                )

                # 2. Get Context
                retrieved_docs = multi_query_retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])

                # 3. Create Prompt
                rag_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant. Answer based only on context. If unknown, say you don't know."),
                    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
                ])
                
                chain = rag_prompt | llm
                response = chain.invoke({"context": context, "question": query})
                
                # 4. Display and Save
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})