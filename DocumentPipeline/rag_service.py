import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_KWARGS = {"device": "cpu"}
EMBEDDING_ENCODE_KWARGS = {"normalize_embeddings": False}
OLLAMA_MODEL_NAME = "qwen2.5:7b"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 5

PROMPT_TEMPLATE = """
Bạn là một trợ lý thông minh của hệ thống SmartDoc AI.
Hãy sử dụng những đoạn thông tin (context) dưới đây để trả lời câu hỏi của người dùng.
Nếu thông tin không có trong context, hãy lịch sự nói rằng bạn không biết, đừng cố bịa ra câu trả lời.
Nếu câu hỏi là câu hỏi tiếp theo, hãy ưu tiên lịch sử hội thoại gần đây để hiểu đúng ngữ cảnh.

Lịch sử hội thoại gần đây:
{chat_history}

Ngữ cảnh (Context):
{context}

Câu hỏi của người dùng:
{input}

Câu trả lời của bạn (Hãy trả lời bằng ngôn ngữ mà người dùng hỏi):
"""


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=EMBEDDING_MODEL_KWARGS,
        encode_kwargs=EMBEDDING_ENCODE_KWARGS,
    )


def create_vector_store(chunks, metadatas=None):
    if not chunks:
        raise ValueError("Không có dữ liệu văn bản để tạo vector store. Vui lòng kiểm tra file PDF đầu vào.")

    embeddings = get_embeddings()
    if metadatas and len(metadatas) == len(chunks):
        vector_store = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
    else:
        vector_store = FAISS.from_texts(chunks, embeddings)

    print(f"vector đầu tiên trong vector store: {vector_store.index.reconstruct(0)}")
    return vector_store


def _format_chat_history(chat_history, max_turns: int = 3) -> str:
    if not chat_history:
        return "Không có lịch sử hội thoại trước đó."

    recent_turns = chat_history[-max_turns:]
    lines = []
    for item in recent_turns:
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()
        if question:
            lines.append(f"Người dùng: {question}")
        if answer:
            lines.append(f"Trợ lý: {answer}")

    return "\n".join(lines) if lines else "Không có lịch sử hội thoại trước đó."


def _build_prompt(context_text: str, query: str, chat_history_text: str = "") -> str:
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt.format(context=context_text, input=query, chat_history=chat_history_text)


def _retrieve_docs(vector_store, query: str, k: int = DEFAULT_TOP_K):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(query)
    print(f"Retrieved docs count: {len(retrieved_docs)}")
    return retrieved_docs


def _apply_metadata_filter(retrieved_docs, metadata_filter) -> list:
    if not metadata_filter:
        return retrieved_docs

    filtered_docs = []
    for doc in retrieved_docs:
        metadata = doc.metadata or {}
        is_match = True
        for key, value in metadata_filter.items():
            if str(metadata.get(key, "")).lower() != str(value).lower():
                is_match = False
                break
        if is_match:
            filtered_docs.append(doc)

    print(f"Filtered docs count: {len(filtered_docs)}")
    return filtered_docs


def _format_sources(retrieved_docs) -> list:
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata or {},
        }
        for doc in retrieved_docs
    ]


def answer_query_with_sources(vector_store, query, chat_history=None, *args, **kwargs):
    if chat_history is None:
        chat_history = kwargs.get("chat_history")
    metadata_filter = kwargs.get("metadata_filter")
    if not query or not query.strip():
        return {"answer": "Vui lòng nhập câu hỏi hợp lệ.", "sources": []}

    retrieved_docs = _retrieve_docs(vector_store, query, k=max(DEFAULT_TOP_K * 4, 20))
    retrieved_docs = _apply_metadata_filter(retrieved_docs, metadata_filter)
    retrieved_docs = retrieved_docs[:DEFAULT_TOP_K]
    if not retrieved_docs:
        return {
            "answer": "Mình không tìm thấy thông tin liên quan trong tài liệu để trả lời câu hỏi này.",
            "sources": [],
        }

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    chat_history_text = _format_chat_history(chat_history)
    final_prompt = _build_prompt(context_text=context_text, query=query, chat_history_text=chat_history_text)

    llm = OllamaLLM(model=OLLAMA_MODEL_NAME, temperature=DEFAULT_TEMPERATURE)
    response = llm.invoke(final_prompt)
    answer_text = response if isinstance(response, str) else str(response)
    sources = _format_sources(retrieved_docs)

    return {"answer": answer_text, "sources": sources}


def answer_query(vector_store, query):
    result = answer_query_with_sources(vector_store, query)
    return result["answer"]
