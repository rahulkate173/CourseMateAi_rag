from langchain_mistralai import ChatMistralAI # llm model 
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate # the prompt template for llm 
from langchain_mistralai import MistralAIEmbeddings # embedding model 
# from langchain_community.document_loaders import TextLoader # to load the text file >> depracted in future 
from langchain_community.document_loaders import PyPDFLoader # to load the pdf files 
from langchain_text_splitters import RecursiveCharacterTextSplitter # text splitter 
from langchain_classic.retrievers.multi_query import MultiQueryRetriever # for variations 
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv() # to get api key 

## creating llm 
llm = ChatMistralAI(
    model = "mistral-small-latest",
    temperature = 0.2
)
## creating embedding model 
embeddings = MistralAIEmbeddings(
    model="mistral-embed",
)
## creating pdf loader 
pdf_loader = PyPDFLoader(
    file_path = "_SuperKalam.pdf"
)
documents = pdf_loader.load()
## creating text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
text = text_splitter.split_documents(documents) # documents are splitted 
## creting vectorstore 
vector_store = Chroma(
    collection_name = "coursemate_rag",
    embedding_function=embeddings,
    persist_directory="./chromadb"
)
## adding documents to vector store 
vector_store.add_documents(documents=text)
## creating reteriver
reteriver = vector_store.as_retriever(
    search_type = "mmr", # better than similarity
    search_kwargs = {"k":2}
)
## creating multiquery
### Some time halucationations are thier 
multi_query = MultiQueryRetriever.from_llm(
    retriever= reteriver,
    llm = llm
)

## getting user query 
print("==== User Response ====")
query = input("User: ")

# Create a prompt specifically for RAG

rag_prompt = ChatPromptTemplate(
    messages=[
        ("system","You are a helpful assistant. Answer the question based only on the provided context below. If you do not know the answer or if it's not in the context, say \"I don't know based on the provided documents.\""),
        ("human","""
        Context:
        {context}
        
        Question:
        {question}
        """)
    ]
)

def get_context(query):
    ## first multiquery create variation 
    retrieved_docs = multi_query.invoke(query)
    # blocks are contexed 
    context_block = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return context_block

context = get_context(query)
rag_prompt = rag_prompt.invoke(
    {"context":context,
    "question":query}
)
# print(f'[INFO] Rag prompt:\n{rag_prompt}')

output = llm.invoke(rag_prompt)
print("==== LLM Response ====")
print(f"Agent:\n{output.content}")
