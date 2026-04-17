import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from sentence_transformers import CrossEncoder


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_KWARGS = {"device": "cpu"}
EMBEDDING_ENCODE_KWARGS = {"normalize_embeddings": False}
OLLAMA_MODEL_NAME = "qwen2.5:7b"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 5
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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


@st.cache_resource
def get_cross_encoder():
    """Tải Cross-Encoder model (cache để không load lại mỗi lần)."""
    return CrossEncoder(CROSS_ENCODER_MODEL)


def get_llm():
    """Tạo LLM instance."""
    return OllamaLLM(model=OLLAMA_MODEL_NAME, temperature=DEFAULT_TEMPERATURE)


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


def _retrieve_docs(vector_store, query: str, k: int = DEFAULT_TOP_K, use_hybrid: bool = False):
    if use_hybrid:
        # Lấy tất cả docs từ vector store để xây BM25
        try:
            all_docs = list(vector_store.docstore._dict.values())
            if all_docs:
                bm25_retriever = BM25Retriever.from_documents(all_docs, k=k)
                faiss_retriever = vector_store.as_retriever(search_kwargs={"k": k})
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[faiss_retriever, bm25_retriever],
                    weights=[0.6, 0.4]
                )
                retrieved_docs = ensemble_retriever.invoke(query)
                print(f"Hybrid retrieval docs count: {len(retrieved_docs)}")
                return retrieved_docs
        except Exception as e:
            print(f"Hybrid search failed, falling back to FAISS: {e}")
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


def _rerank_docs(query: str, docs: list, top_k: int = DEFAULT_TOP_K) -> list:
    """
    Câu 9: Re-ranking bằng Cross-Encoder.
    Cross-encoder đánh giá từng cặp (query, doc) độc lập → chính xác hơn bi-encoder.
    Trả về docs đã sắp xếp lại kèm điểm rerank.
    """
    if not docs:
        return docs
    cross_encoder = get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    # Gắn điểm vào metadata để hiển thị trên UI
    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = round(float(score), 4)
    # Sắp xếp theo điểm giảm dần, lấy top_k
    reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    result = [doc for _, doc in reranked[:top_k]]
    print(f"Reranked docs: top score={reranked[0][0]:.4f}, bottom score={reranked[-1][0]:.4f}")
    return result


def _rewrite_query(query: str, chat_history_text: str = "") -> str:
    """
    Câu 10: Query rewriting — LLM cải thiện câu hỏi trước khi retrieval.
    Giúp xử lý follow-up questions, từ viết tắt, câu hỏi mơ hồ.
    """
    rewrite_prompt = f"""Bạn là trợ lý cải thiện câu hỏi tìm kiếm.
Dựa vào lịch sử hội thoại (nếu có), hãy viết lại câu hỏi sau thành câu hỏi rõ ràng, đầy đủ ngữ cảnh, phù hợp để tìm kiếm trong tài liệu.
Chỉ trả về câu hỏi đã viết lại, không giải thích thêm.

Lịch sử hội thoại:
{chat_history_text}

Câu hỏi gốc: {query}

Câu hỏi đã viết lại:"""
    try:
        llm = get_llm()
        rewritten = llm.invoke(rewrite_prompt).strip()
        # Nếu model trả về quá dài hoặc lỗi, fallback về query gốc
        if not rewritten or len(rewritten) > 500:
            return query
        print(f"Query rewritten: '{query}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"Query rewriting failed: {e}")
        return query


def _self_evaluate(query: str, answer: str, context_text: str) -> dict:
    """
    Câu 10: Self-RAG — LLM tự đánh giá câu trả lời của mình.
    Trả về confidence score (0-10) và nhận xét ngắn.
    """
    eval_prompt = f"""Bạn là một chuyên gia đánh giá chất lượng câu trả lời RAG.
Hãy đánh giá câu trả lời dưới đây dựa trên:
1. Câu trả lời có dựa trên ngữ cảnh được cung cấp không?
2. Câu trả lời có chính xác và đầy đủ không?

Câu hỏi: {query}
Ngữ cảnh: {context_text[:800]}
Câu trả lời: {answer[:500]}

Trả lời theo định dạng JSON sau (không thêm gì khác):
{{"score": <số từ 0 đến 10>, "reason": "<lý do ngắn gọn trong 1 câu>"}}"""
    try:
        llm = get_llm()
        raw = llm.invoke(eval_prompt).strip()
        # Parse JSON từ response
        import re, json
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score = max(0, min(10, int(data.get("score", 5))))
            reason = str(data.get("reason", ""))[:200]
            return {"score": score, "reason": reason}
    except Exception as e:
        print(f"Self-evaluation failed: {e}")
    return {"score": -1, "reason": "Không thể đánh giá"}


def answer_query_with_sources(vector_store, query, chat_history=None, *args, **kwargs):
    if chat_history is None:
        chat_history = kwargs.get("chat_history")
    metadata_filter = kwargs.get("metadata_filter")
    use_hybrid = kwargs.get("use_hybrid", False)
    use_reranking = kwargs.get("use_reranking", False)   # Câu 9
    use_self_rag = kwargs.get("use_self_rag", False)     # Câu 10
    if not query or not query.strip():
        return {"answer": "Vui lòng nhập câu hỏi hợp lệ.", "sources": []}

    chat_history_text = _format_chat_history(chat_history)

    # Câu 10: Query rewriting
    rewritten_query = query
    if use_self_rag:
        rewritten_query = _rewrite_query(query, chat_history_text)

    retrieved_docs = _retrieve_docs(vector_store, rewritten_query, k=max(DEFAULT_TOP_K * 2, 10), use_hybrid=use_hybrid)
    retrieved_docs = _apply_metadata_filter(retrieved_docs, metadata_filter)

    # Câu 9: Re-ranking với Cross-Encoder
    if use_reranking and retrieved_docs:
        retrieved_docs = _rerank_docs(rewritten_query, retrieved_docs, top_k=DEFAULT_TOP_K)
    else:
        retrieved_docs = retrieved_docs[:DEFAULT_TOP_K]

    if not retrieved_docs:
        return {
            "answer": "Mình không tìm thấy thông tin liên quan trong tài liệu để trả lời câu hỏi này.",
            "sources": [],
            "rewritten_query": rewritten_query if use_self_rag else None,
            "confidence": None,
        }

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    final_prompt = _build_prompt(context_text=context_text, query=query, chat_history_text=chat_history_text)

    llm = get_llm()
    response = llm.invoke(final_prompt)
    answer_text = response if isinstance(response, str) else str(response)

    # Câu 10: Self-evaluation
    confidence = None
    if use_self_rag:
        confidence = _self_evaluate(query, answer_text, context_text)

    sources = _format_sources(retrieved_docs)
    return {
        "answer": answer_text,
        "sources": sources,
        "rewritten_query": rewritten_query if use_self_rag else None,
        "confidence": confidence,
    }


def answer_query(vector_store, query):
    result = answer_query_with_sources(vector_store, query)
    return result["answer"]
