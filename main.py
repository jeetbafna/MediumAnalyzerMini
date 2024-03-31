from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as pine
from dotenv import load_dotenv
from langchain import VectorDBQA, OpenAI
from pinecone import Pinecone
from langchain.chains import RetrievalQA

import os

load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
if __name__ == "__main__":
    print("Hello everyone")

    loader = TextLoader("./mediumAnalyzer/mediumBlog.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = pine.from_documents(
        texts, embeddings, index_name="medium-blog-embeddings-index"
    )

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    query = "What is a vector DB? Give me a 15 word answer for a beginner"
    result = qa({"query": query})
    print(result)
