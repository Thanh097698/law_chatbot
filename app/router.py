from fastapi import APIRouter
from pydantic import BaseModel
from app.rag_chain import generate_answer

router = APIRouter(prefix="/ask", tags=["Q&A"])


class Question(BaseModel):
    query: str


@router.post("/")
async def ask_question(payload: Question):
    """Nhận câu hỏi từ người dùng và trả về câu trả lời từ hệ thống RAG."""
    answer = generate_answer(payload.query)
    return {"question": payload.query, "answer": answer}
