import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from app.config import FAISS_PATH


def get_pdf_files(directory: str) -> list:
    """Lấy danh sách file PDF trong thư mục."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pdf")]


def load_documents(file_paths: list) -> list:
    """Load tất cả tài liệu PDF thành documents."""
    documents = []
    for file in file_paths:
        print(f"Loading: {file}")
        loader = PyPDFLoader(file)
        docs = loader.load()
        documents.extend(docs)
    return documents


def split_documents(documents: list, chunk_size=1000, chunk_overlap=150) -> list:
    """Chia nhỏ tài liệu thành các chunk."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def create_and_save_faiss_index(chunks: list, faiss_path: str):
    """Tạo FAISS index từ chunks và lưu lại."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(faiss_path)
    print(f"FAISS index saved to {faiss_path}")


def load_and_index_pdf(data_dir="law_data"):
    """Pipeline chính: load PDF, chunk, tạo index và lưu."""
    print(f"Scanning folder: {data_dir}")
    pdf_files = get_pdf_files(data_dir)

    if not pdf_files:
        print("Không tìm thấy file PDF nào trong thư mục.")
        return

    documents = load_documents(pdf_files)
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(pdf_files)} files.")

    create_and_save_faiss_index(chunks, FAISS_PATH)
