from fastapi import FastAPI
from app.router import router
from app.config import FAISS_PATH, DATA_DIR
import os
from app.load_data import load_and_index_pdf

app = FastAPI(title="LawBot (Gemini)")

@app.on_event("startup")
def init_index():
    """Tự động tạo FAISS index nếu chưa tồn tại khi app khởi động."""
    if not os.path.exists(FAISS_PATH) or not os.listdir(FAISS_PATH):
        print("FAISS index chưa tồn tại. Đang tạo FAISS index từ PDF...")
        load_and_index_pdf(data_dir=DATA_DIR)
    else:
        print("FAISS index đã có sẵn.")

app.include_router(router)