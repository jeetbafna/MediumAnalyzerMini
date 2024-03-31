from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import  OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
import pinecone
import os

load_dotenv()
pinecone.init(api_key = os.environ.get("PINECONE_API_KEY"))
if __name__ == "__main__":
    print("Hello everyone")

    loader = TextLoader("./mediumAnalyzer/mediumBlog.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))


