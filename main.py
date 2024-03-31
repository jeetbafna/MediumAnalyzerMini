from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

if __name__ == "__main__":
    print("Hello everyone")
    loader = TextLoader("./mediumAnalyzer/mediumBlog.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
