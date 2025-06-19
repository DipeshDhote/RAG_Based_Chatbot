from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv


load_dotenv()



sys_promt="""
            You are an intelligent AI booking assistant at restaurant name Jamun to handle restaurant-related queries and bookings.
            Answer user questions about the restaurant’s menu, location, timings, facilities, and booking process based on the information provided in your knowledge base.
            Use the retriever tool to search for accurate answers based on the available content.
            Always cite specific parts of the document or database you reference in your answers.
            Keep your responses short, clear, and helpful. When booking, confirm the details with the user before proceeding.
        """





# This function load the pdf
def load_file(file_path):
    file = Path(file_path)

    pdf_file = PyPDFLoader(file)

    # Ensure file loaded properly
    try:
        pages = pdf_file.load()

        print(f"file loaded successfully total {len(pages)} is loaded")
    except Exception as e:
        print("File not loaded")
        raise

    return pages


def get_retriever(document,collection_name):
    
    # Embedding
    embeddings = HuggingFaceEmbeddings()

    # Split the document
    splitter = RecursiveCharacterTextSplitter()
    chunks = splitter.split_documents(document)

    # Store in vectore store
    vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding= embeddings,
                        collection_name=collection_name
                        )
    
    retriever = vector_store.as_retriever(
                                search_type="similarity",
                                search_kwargs={'k': 5} 
    )
    print("Retriever Created Successfully")

    return retriever