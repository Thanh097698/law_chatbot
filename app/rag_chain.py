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
        temperature=0.5
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
        return_source_documents=True
    )


def generate_answer(question: str) -> dict:
    """Sinh câu trả lời từ câu hỏi, trả về kèm context và metadata."""
    qa_chain = build_qa_chain()
    response = qa_chain.invoke({"query": question})

    answer = response["result"]
    source_documents = response.get("source_documents", [])

    contexts = [doc.page_content for doc in source_documents]
    metadata = [doc.metadata for doc in source_documents]

    return {
        "answer": answer,
        "contexts": contexts,
        "metadata": metadata
    }


def print_answer(result: dict):
    """In câu trả lời và context một cách rõ ràng, có trích nguồn."""
    print("\nCâu trả lời:")
    print("--------------")
    print(result["answer"].strip())

    print("\nTrích dẫn từ tài liệu:")
    print("-------------------------")
    for i, (context, meta) in enumerate(zip(result["contexts"], result["metadata"]), start=1):
        clean_context = context.strip().replace("\\n", "\n")
        source = meta.get("source", "Không rõ")
        page = meta.get("page", "Không rõ")
        print(f"\n[{i}] Tài liệu: {source}, Trang: {page}")
        print(clean_context)
