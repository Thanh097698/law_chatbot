import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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


def split_by_chapter(documents: list) -> list:
    """Chia tài liệu theo các chương (Chương I, Chương II, ...)."""
    all_chunks = []

    for doc in documents:
        # Tách văn bản theo các chương
        splits = re.split(r'(Chương\s+[^\n]+)', doc.page_content)

        for i in range(1, len(splits), 2):
            title = splits[i].strip()
            content = splits[i + 1].strip() if i + 1 < len(splits) else ""
            full_text = f"{title}\n{content}"

            all_chunks.append(Document(page_content=full_text, metadata=doc.metadata))

    return all_chunks


def create_and_save_faiss_index(chunks: list, faiss_path: str):
    """Tạo FAISS index từ chunks và lưu lại."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(faiss_path)
    print(f"FAISS index saved to {faiss_path}")


def load_and_index_pdf(data_dir="law_data"):
    """Pipeline chính: load PDF, chia theo chương, tạo index và lưu."""
    print(f"Scanning folder: {data_dir}")
    pdf_files = get_pdf_files(data_dir)

    if not pdf_files:
        print("Không tìm thấy file PDF nào trong thư mục.")
        return

    documents = load_documents(pdf_files)
    chunks = split_by_chapter(documents)
    print(f"Created {len(chunks)} chương từ {len(pdf_files)} file PDF.")

    create_and_save_faiss_index(chunks, FAISS_PATH)
