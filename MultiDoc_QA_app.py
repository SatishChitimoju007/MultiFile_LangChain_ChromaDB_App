
import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

# Load and process the text files
loader = DirectoryLoader('./text_files', glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create the vector database
persist_directory = "database"
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)

# Persist the database to disk
vectordb.persist()
vectordb = None

# Load the persisted database from disk
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Create a retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# Create a language model chain
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)

@app.post("/query")
def process_query(request: QueryRequest):
    query = request.query
    llm_response = qa_chain(query)
    result = llm_response['result']
    sources = []
    for source in llm_response['source_documents']:
        sources.append(source.metadata['source'])
    return {"result": result, "sources": sources}