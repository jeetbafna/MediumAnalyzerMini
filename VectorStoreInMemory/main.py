import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
load_dotenv()

if __name__ == "__main__":
    print("welcome to InstoreVectorDB")
    pdf_path = "./ReactPaper.pdf"
    loader = PyPDFLoader(file_path= pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embeddings)
    vectorStore.save_local("faiss_index_react")

    newVectorStore = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)

    qa = RetrievalQA.from_chain_type(llm= OpenAI(), chain_type="stuff", retriever= newVectorStore.as_retriever())

    res = qa.run("Give me the gist of ReAct in 3 sentences")
    print(res)
