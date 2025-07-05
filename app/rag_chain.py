from langchain.chains import RetrievalQA
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from app.vectorstore import load_vectorstore


def get_prompt_template() -> PromptTemplate:
    template = """
    Bạn là một trợ lý pháp lý thông minh. Dưới đây là một số thông tin trích từ tài liệu pháp luật:

    {context}

    Dựa vào thông tin trên, hãy trả lời câu hỏi sau một cách chính xác, ngắn gọn và dễ hiểu:

    Câu hỏi: {question}

    Trả lời:
    """
    return PromptTemplate.from_template(template)


def get_retriever(k: int = 3):
    """Load vectorstore và lấy retriever."""
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_llm():
    """Khởi tạo mô hình Gemini Flash."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.2
    )


def build_qa_chain():
    """Tạo chuỗi RAG RetrievalQA chain."""
    prompt = get_prompt_template()
    retriever = get_retriever()
    llm = get_llm()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )


def generate_answer(question: str) -> str:
    """Trả lời câu hỏi bằng hệ thống RAG."""
    qa_chain = build_qa_chain()
    result = qa_chain.run(question)
    return result
