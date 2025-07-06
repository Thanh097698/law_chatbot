from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from app.rag_chain import generate_answer

router = APIRouter(prefix="/ask", tags=["Q&A"])


class Question(BaseModel):
    query: str


@router.post("/")
async def ask_question(payload: Question):
    """Nhận câu hỏi từ người dùng và trả về câu trả lời từ hệ thống RAG."""
    try:
        answer = generate_answer(payload.query)
        return JSONResponse(content={"question": payload.query, "answer": answer})
    except Exception as e:
        print(f"[ERROR] Lỗi trong /ask: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})