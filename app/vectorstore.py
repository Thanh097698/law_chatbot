import os
from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from app.config import FAISS_PATH

def load_vectorstore() -> FAISS:
    """Load FAISS vector store đã được lưu trữ trước đó."""
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục FAISS index tại '{FAISS_PATH}'. "
            f"Bạn cần chạy hàm 'load_and_index_pdf()' trước."
        )

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local(
        folder_path=FAISS_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore