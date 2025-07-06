from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
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
    """Tải vectorstore và trả về retriever."""
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_llm():
    """Khởi tạo mô hình Gemini Flash từ Google."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7
    )


def build_qa_chain():
    """Tạo chuỗi RAG: LLM + Retriever + Prompt."""
    prompt = get_prompt_template()
    retriever = get_retriever()
    llm = get_llm()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False  
    )


def generate_answer(question: str) -> str:
    """Trả về chỉ phần câu trả lời từ hệ thống."""
    qa_chain = build_qa_chain()
    response = qa_chain.invoke({"query": question})
    return response["result"].strip()


def print_answer(answer: str):
    """In câu trả lời đơn giản."""
    print("\nCâu trả lời:")
    print("---------------")
    print(answer)
