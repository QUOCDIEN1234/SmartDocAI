import json
import os
from datetime import datetime
import re
import glob
import html

import streamlit as st
from DocumentPipeline.Processing import (
    load_docx,
    load_pdf_pages,
    split_text_data,
    split_pdf_pages_with_metadata,
)
from DocumentPipeline.model_llm import create_vector_store, answer_query_with_sources

# Thư mục lưu lịch sử chat
HISTORY_DIR = "chat_history"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)


def apply_ui_theme():
    st.markdown(
        """
        <style>
        :root {
            --primary: #007BFF;
            --secondary: #FFC107;
            --bg-main: #F8F9FA;
            --bg-sidebar: #2C2F33;
            --text-main: #212529;
            --text-sidebar: #FFFFFF;
            --panel: #FFFFFF;
            --border: rgba(33, 37, 41, 0.08);
        }
        .stApp {
            background: var(--bg-main);
            color: var(--text-main);
        }
        section[data-testid="stSidebar"] {
            background: var(--bg-sidebar);
        }
        section[data-testid="stSidebar"] * {
            color: var(--text-sidebar);
        }
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stMarkdown li,
        section[data-testid="stSidebar"] .stMarkdown span,
        section[data-testid="stSidebar"] .stCaption {
            color: var(--text-sidebar);
        }
        .block-container {
            max-width: 980px;
            margin: 0 auto;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: var(--text-main);
        }
        .smartdoc-hero {
            background: linear-gradient(135deg, #FFFFFF 0%, #EAF3FF 100%);
            border: 1px solid rgba(0, 123, 255, 0.12);
            border-radius: 18px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(33, 37, 41, 0.06);
        }
        .smartdoc-panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 24px rgba(33, 37, 41, 0.05);
            margin-bottom: 1rem;
        }
        section[data-testid="stSidebar"] .smartdoc-panel.model-highlight {
            background: linear-gradient(135deg, #FFFFFF 0%, #EAF3FF 100%);
            border: 1px solid rgba(0, 123, 255, 0.35);
            box-shadow: 0 10px 24px rgba(0, 123, 255, 0.18);
        }
        section[data-testid="stSidebar"] .smartdoc-panel.model-highlight,
        section[data-testid="stSidebar"] .smartdoc-panel.model-highlight p,
        section[data-testid="stSidebar"] .smartdoc-panel.model-highlight strong,
        section[data-testid="stSidebar"] .smartdoc-panel.model-highlight span {
            color: #111111 !important;
        }
        .chat-shell {
            background: #FDFDFE;
            border: 1px solid rgba(33, 37, 41, 0.12);
            border-radius: 18px;
            padding: 1rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 26px rgba(33, 37, 41, 0.06);
        }
        [data-testid="stChatMessage"] {
            border-radius: 14px;
            border: 1px solid rgba(33, 37, 41, 0.08);
            padding: 0.6rem 0.85rem;
            margin-bottom: 0.7rem;
            color: var(--text-main);
        }
        [data-testid="stChatMessage"][aria-label*="user"] {
            background: rgba(0, 123, 255, 0.08);
            border-color: rgba(0, 123, 255, 0.2);
            margin-left: 8%;
        }
        [data-testid="stChatMessage"][aria-label*="assistant"] {
            background: #FFFFFF;
            margin-right: 8%;
        }
        [data-testid="chatAvatarIcon-user"],
        [data-testid="chatAvatarIcon-assistant"] {
            color: var(--primary);
        }
        .chat-empty {
            color: #6C757D;
            text-align: center;
            padding: 0.8rem;
        }
        .smartdoc-badge {
            display: inline-block;
            background: var(--primary);
            color: white;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 0.5rem;
        }
      
        .smartdoc-note {
            background: #FFF9E6;
            border-left: 4px solid #FFC107;
            padding: 0.75rem 0.9rem;
            border-radius: 8px;
            color: #212529;
        }
        div[data-testid="stFileUploader"] section {
            border: 1px dashed var(--primary);
            border-radius: 14px;
            background: #FFFFFF;
        }
        div[data-testid="stFileUploader"] button {
            background: var(--secondary) !important;
            color: #212529 !important;
            border: none !important;
            font-weight: 700 !important;
        }
        div[data-testid="stForm"] {
            background: #FFFFFF;
            border: 1px solid rgba(33, 37, 41, 0.08);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(33, 37, 41, 0.05);
        }
        div[data-testid="stForm"] input {
            border-radius: 12px !important;
            border: 1px solid rgba(0, 123, 255, 0.25) !important;
        }
        div[data-testid="stForm"] button {
            border-radius: 10px !important;
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_guidance():
    st.sidebar.markdown("## SmartDoc AI")
    st.sidebar.markdown("<div class='smartdoc-note' style='color:#212529;'>Hướng dẫn nhanh: tải PDF, chờ hệ thống xử lý, rồi đặt câu hỏi để xem câu trả lời cùng nguồn trích dẫn.</div>", unsafe_allow_html=True)
    st.sidebar.markdown("### Instructions")
    st.sidebar.markdown("1. Chọn file PDF hoặc DOCX\n2. Chờ xử lý tài liệu\n3. Nhập câu hỏi\n4. Xem câu trả lời và nguồn gốc")




def render_sidebar_model_info():
    st.sidebar.markdown("### Model Configuration Display")
    st.sidebar.markdown(
        f"""
        <div class='smartdoc-panel model-highlight'>
            <p><strong>Embedding:</strong> sentence-transformers/all-MiniLM-L6-v2</p>
            <p><strong>LLM:</strong> qwen2.5:7b</p>
            <p><strong>Chunk size:</strong> {st.session_state.chunk_size}</p>
            <p><strong>Chunk overlap:</strong> {st.session_state.chunk_overlap}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_file_name(file_name: str) -> str:
    """
    Chuẩn hóa tên file để sử dụng làm định danh
    Loại bỏ các ký tự đặc biệt, giữ lại tên file rõ ràng
    """
    # Loại bỏ phần mở rộng
    name_without_ext = os.path.splitext(file_name)[0]
    # Chỉ giữ lại các ký tự chữ, số, gạch ngang và gạch dưới
    cleaned = re.sub(r'[^\w\-]', '_', name_without_ext)
    # Loại bỏ gạch dưới dư thừa
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned.strip('_')


def get_history_file_path(file_identifier: str) -> str:
    """
    Lấy đường dẫn file lưu lịch sử cho một tài liệu
    file_identifier: tên file đã được chuẩn hóa
    """
    return os.path.join(HISTORY_DIR, f"{file_identifier}.json")


def load_history_from_file(file_identifier: str) -> list:
    """
    Tải lịch sử chat từ file
    file_identifier: tên file đã được chuẩn hóa
    """
    history_file = get_history_file_path(file_identifier)
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Không thể tải lịch sử: {e}")
            return []
    return []


def save_history_to_file(file_identifier: str, history: list):
    """
    Lưu lịch sử chat vào file
    file_identifier: tên file đã được chuẩn hóa
    """
    history_file = get_history_file_path(file_identifier)
    try:
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Không thể lưu lịch sử: {e}")


def initialize_session_state():
    """Khởi tạo các biến session cần thiết cho ứng dụng"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_file_identifier" not in st.session_state:
        st.session_state.current_file_identifier = None
    if "current_file_name" not in st.session_state:
        st.session_state.current_file_name = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "document_text" not in st.session_state:
        st.session_state.document_text = ""
    if "selected_history_index" not in st.session_state:
        st.session_state.selected_history_index = 0
    if "confirm_clear_history" not in st.session_state:
        st.session_state.confirm_clear_history = False
    if "confirm_clear_vector_store" not in st.session_state:
        st.session_state.confirm_clear_vector_store = False
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 1000
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 200
    if "last_chunk_params" not in st.session_state:
        st.session_state.last_chunk_params = (1000, 200)
    if "document_pages" not in st.session_state:
        st.session_state.document_pages = []
    if "document_catalog" not in st.session_state:
        st.session_state.document_catalog = []


def clear_chat_history():
    """Xóa toàn bộ lịch sử chat (tất cả file JSON trong thư mục history)."""
    for history_file in glob.glob(os.path.join(HISTORY_DIR, "*.json")):
        try:
            os.remove(history_file)
        except OSError:
            pass

    st.session_state.chat_history = []
    st.session_state.selected_history_index = 0


def clear_vector_store_data():
    """Xóa toàn bộ dữ liệu liên quan tới file đã upload (session + file lưu trên ổ đĩa)."""
    # Xóa toàn bộ file tạm đã upload trong thư mục làm việc hiện tại.
    for pattern in ("temp_*.pdf", "temp_*.docx"):
        for temp_file in glob.glob(pattern):
            try:
                os.remove(temp_file)
            except OSError:
                pass


    st.session_state.vector_store = None
    st.session_state.document_text = ""
    st.session_state.chat_history = []
    st.session_state.current_file_identifier = None
    st.session_state.current_file_name = None
    st.session_state.selected_history_index = 0
    st.session_state.last_chunk_params = (st.session_state.chunk_size, st.session_state.chunk_overlap)
    st.session_state.document_pages = []
    st.session_state.document_catalog = []


def reset_session_for_new_document(file_identifier: str, file_name: str):
    """
    Đặt lại session khi tải tài liệu mới
    - Tải lịch sử cũ nếu file này từng được xử lý trước đó
    - Xóa vector store để tạo lại cho document mới
    """
    history = load_history_from_file(file_identifier)
    
    st.session_state.chat_history = history
    st.session_state.current_file_identifier = file_identifier
    st.session_state.current_file_name = file_name
    st.session_state.vector_store = None
    st.session_state.document_text = ""
    st.session_state.document_pages = []
    st.session_state.selected_history_index = 0
    
    if history:
        st.sidebar.success(f"✓ Tìm thấy {len(history)} câu hỏi trước đó cho file này!")


def render_chat_history_sidebar():
    """
    Hiển thị lịch sử chat trong sidebar
    - Danh sách các câu hỏi
    - Cho phép chọn một câu hỏi để xem chi tiết
    """
    if st.sidebar.button("Clear History"):
        st.session_state.confirm_clear_history = True

    if st.session_state.confirm_clear_history:
        st.sidebar.warning("Bạn có chắc muốn xóa toàn bộ lịch sử chat (tất cả file JSON)?")
        confirm_col, cancel_col = st.sidebar.columns(2)
        if confirm_col.button("Confirm", key="confirm_clear_history_btn"):
            clear_chat_history()
            st.session_state.confirm_clear_history = False
            st.sidebar.success("Đã xóa toàn bộ lịch sử chat.")
        if cancel_col.button("Cancel", key="cancel_clear_history_btn"):
            st.session_state.confirm_clear_history = False

    if st.sidebar.button("Clear Vector Store"):
        st.session_state.confirm_clear_vector_store = True

    if st.session_state.confirm_clear_vector_store:
        st.sidebar.warning("Bạn có chắc muốn xóa vector store và tài liệu đã upload?")
        confirm_col, cancel_col = st.sidebar.columns(2)
        if confirm_col.button("Confirm", key="confirm_clear_vector_store_btn"):
            clear_vector_store_data()
            st.session_state.uploader_key += 1
            st.session_state.confirm_clear_vector_store = False
            st.sidebar.success("Đã xóa toàn bộ dữ liệu file đã upload.")
            st.rerun()
        if cancel_col.button("Cancel", key="cancel_clear_vector_store_btn"):
            st.session_state.confirm_clear_vector_store = False

    st.sidebar.header("📜 Lịch sử câu hỏi")

    if not st.session_state.chat_history:
        st.sidebar.info("Chưa có câu hỏi nào trong phiên này.")
        return

    # Tạo danh sách hiển thị các câu hỏi
    options = [
        f"{index + 1}. {item['question'][:45]}"
        for index, item in enumerate(st.session_state.chat_history)
    ]

    selected_label = st.sidebar.selectbox(
        "Chọn một câu hỏi để xem lại",
        options,
        index=min(st.session_state.selected_history_index, len(options) - 1),
        key="history_selector",
    )
    st.session_state.selected_history_index = options.index(selected_label)

    # Hiển thị chi tiết câu hỏi và câu trả lời
    selected_item = st.session_state.chat_history[st.session_state.selected_history_index]
    st.sidebar.markdown("---")
    st.sidebar.markdown("**❓ Câu hỏi:**")
    st.sidebar.write(selected_item["question"])
    st.sidebar.markdown("**✅ Câu trả lời:**")
    st.sidebar.write(selected_item["answer"])
    
    # Hiển thị thời gian (nếu có)
    if "timestamp" in selected_item:
        st.sidebar.caption(f"Thời gian: {selected_item['timestamp']}")


def build_history_display():
    """
    Hiển thị lịch sử theo dạng hội thoại chatbot (kiểu ChatGPT)
    Không thay đổi dữ liệu hay logic lưu lịch sử
    """
    

    if not st.session_state.chat_history:
        st.markdown(
            "<div class='chat-empty'>Hãy tải lên tài liệu và đặt câu hỏi để bắt đầu hội thoại.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(item["question"])

        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            item_sources = item.get("sources") or []
            if item_sources:
                with st.expander("Nguồn tham chiếu", expanded=False):
                    render_answer_sources(item_sources, st.session_state.document_pages, show_title=False)
            if "timestamp" in item:
                st.caption(f"Thời gian: {item['timestamp']}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_answer_sources(sources, document_pages, show_title=True):
    """Hiển thị nguồn: trang, dòng, context gốc và highlight đoạn đã dùng."""
    if not sources:
        return

    if show_title:
        st.markdown("### Nguồn thông tin")
    page_map = {}
    for p in document_pages:
        source_file = p.get("file_name", "")
        page_map[(source_file, p["page"])] = p["text"]

    for idx, source in enumerate(sources, start=1):
        content = (source.get("content") or "").strip()
        metadata = source.get("metadata") or {}
        source_file = metadata.get("file_name", "Unknown")
        page = metadata.get("page", "N/A")
        start_line = metadata.get("start_line", "N/A")
        end_line = metadata.get("end_line", "N/A")

        with st.expander(f"Nguồn {idx} | File {source_file} | Trang {page} | Dòng {start_line} - {end_line}"):
            st.markdown("**Đoạn được dùng để trả lời:**")
            st.info(content)

            page_text = page_map.get((source_file, page), "")
            if page_text and content:
                escaped_page = html.escape(page_text)
                escaped_content = html.escape(content)
                highlighted_page = escaped_page.replace(escaped_content, f"<mark>{escaped_content}</mark>", 1)

                st.markdown("**Context gốc của trang (đã highlight):**")
                st.markdown(
                    f"<div style='white-space: pre-wrap; line-height:1.6;'>{highlighted_page}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("**Context gốc:**")
                st.write(content)


def main():    
    initialize_session_state()
    apply_ui_theme()

    st.title("SmartDoc AI - Trợ lý thông minh cho tài liệu của bạn")
    st.markdown(
        "<div class='smartdoc-hero'><h3>Hiển thị nguồn gốc câu trả lời, cho phép xem context gốc và theo dõi tài liệu dễ dàng.</h3><p>Giao diện được tối ưu cho accessibility với bảng màu rõ ràng, sidebar hướng dẫn, và khu vực trả lời tập trung.</p></div>",
        unsafe_allow_html=True,
    )
    st.write("Tải lên tài liệu PDF hoặc DOCX của bạn và đặt câu hỏi để nhận câu trả lời thông minh dựa trên nội dung của tài liệu.")

    render_chat_history_sidebar()
    render_sidebar_guidance()
    render_sidebar_model_info()

    uploaded_files = st.file_uploader(
        "Chọn một hoặc nhiều tệp PDF/DOCX",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}",
    )
    
    if uploaded_files:
        uploaded_names = [f.name for f in uploaded_files]
        combined_identifier = normalize_file_name("__".join(sorted(uploaded_names)))

        if st.session_state.current_file_identifier != combined_identifier:
            display_name = uploaded_names[0] if len(uploaded_names) == 1 else f"{len(uploaded_names)} files"
            reset_session_for_new_document(combined_identifier, display_name)

        st.success(f"Đã tải lên {len(uploaded_files)} tệp thành công!")
        processing_progress = st.progress(0)
        processing_status = st.empty()
        
        if not st.session_state.document_text:
            all_text_parts = []
            all_document_pages = []
            document_catalog = []

            processing_status.info("Đang đọc nội dung các tài liệu...")
            processing_progress.progress(25)

            for uploaded_file in uploaded_files:
                file_extension = uploaded_file.name.split(".")[-1].lower()
                file_identifier = normalize_file_name(uploaded_file.name)
                temp_file_path = f"temp_{file_identifier[:16]}.{file_extension}"
                upload_date = datetime.now().isoformat()

                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                document_catalog.append(
                    {
                        "file_name": uploaded_file.name,
                        "upload_date": upload_date,
                        "document_type": file_extension,
                    }
                )

                if file_extension == "pdf":
                    pages = load_pdf_pages(temp_file_path)
                    for p in pages:
                        p["file_name"] = uploaded_file.name
                    all_document_pages.extend(pages)
                    all_text_parts.append("\n\n".join(p["text"] for p in pages))
                elif file_extension == "docx":
                    all_text_parts.append(load_docx(temp_file_path))

            st.session_state.document_pages = all_document_pages
            st.session_state.document_text = "\n\n".join(part for part in all_text_parts if part and part.strip())
            st.session_state.document_catalog = document_catalog
            processing_progress.progress(50)

        st.markdown("### Cấu hình Chunk")
        chunk_col_1, chunk_col_2 = st.columns(2)
        selected_chunk_size = chunk_col_1.selectbox(
            "Chunk size",
            options=[500, 1000, 1500, 2000],
            index=[500, 1000, 1500, 2000].index(st.session_state.chunk_size) if st.session_state.chunk_size in [500, 1000, 1500, 2000] else 1,
        )
        selected_chunk_overlap = chunk_col_2.selectbox(
            "Chunk overlap",
            options=[50, 100, 200],
            index=[50, 100, 200].index(st.session_state.chunk_overlap) if st.session_state.chunk_overlap in [50, 100, 200] else 2,
        )

        if (
            selected_chunk_size != st.session_state.chunk_size
            or selected_chunk_overlap != st.session_state.chunk_overlap
        ):
            st.session_state.chunk_size = selected_chunk_size
            st.session_state.chunk_overlap = selected_chunk_overlap
            st.session_state.vector_store = None

        text = st.session_state.document_text
        if not text or not text.strip():
            st.warning("Không trích xuất được nội dung văn bản từ PDF. Nếu đây là file scan ảnh, bạn cần thêm OCR.")
            return

        if st.session_state.vector_store is None:
            processing_status.info("Đang chia nhỏ tài liệu và tạo vector store...")
            chunks = []
            metadatas = []

            for uploaded_file in uploaded_files:
                file_extension = uploaded_file.name.split(".")[-1].lower()
                file_identifier = normalize_file_name(uploaded_file.name)
                temp_file_path = f"temp_{file_identifier[:16]}.{file_extension}"
                upload_date = next(
                    (
                        item["upload_date"]
                        for item in st.session_state.document_catalog
                        if item["file_name"] == uploaded_file.name
                    ),
                    datetime.now().isoformat(),
                )

                if file_extension == "pdf":
                    pages = [p for p in st.session_state.document_pages if p.get("file_name") == uploaded_file.name]
                    page_chunks, page_metadatas = split_pdf_pages_with_metadata(
                        pages,
                        chunk_size=st.session_state.chunk_size,
                        chunk_overlap=st.session_state.chunk_overlap,
                    )
                    for metadata in page_metadatas:
                        metadata["file_name"] = uploaded_file.name
                        metadata["upload_date"] = upload_date
                        metadata["document_type"] = "pdf"

                    chunks.extend(page_chunks)
                    metadatas.extend(page_metadatas)
                elif file_extension == "docx":
                    docx_text = load_docx(temp_file_path)
                    docx_chunks = split_text_data(
                        docx_text,
                        chunk_size=st.session_state.chunk_size,
                        chunk_overlap=st.session_state.chunk_overlap,
                    )
                    chunks.extend(docx_chunks)
                    metadatas.extend(
                        {
                            "file_name": uploaded_file.name,
                            "upload_date": upload_date,
                            "document_type": "docx",
                            "page": "N/A",
                            "start_line": "N/A",
                            "end_line": "N/A",
                        }
                        for _ in docx_chunks
                    )

            if not chunks:
                st.warning("Không tạo được đoạn văn bản nào từ tài liệu. Vui lòng kiểm tra lại PDF đầu vào.")
                return

            processing_progress.progress(75)
            st.session_state.vector_store = create_vector_store(chunks, metadatas=metadatas)
            processing_progress.progress(100)
            processing_status.success("Tài liệu đã sẵn sàng để hỏi đáp.")
            st.session_state.last_chunk_params = (
                st.session_state.chunk_size,
                st.session_state.chunk_overlap,
            )

        st.caption(
            f"Chunk hiện tại: size={st.session_state.last_chunk_params[0]}, overlap={st.session_state.last_chunk_params[1]}"
        )

        vector_store = st.session_state.vector_store
        build_history_display()

        st.markdown("### Bộ lọc tìm kiếm theo metadata")
        available_files = ["All"] + [item["file_name"] for item in st.session_state.document_catalog]
        selected_file_filter = st.selectbox("Lọc theo file", options=available_files, index=0)
        selected_type_filter = st.selectbox("Lọc theo loại", options=["All", "pdf", "docx"], index=0)

        metadata_filter = {}
        if selected_file_filter != "All":
            metadata_filter["file_name"] = selected_file_filter
        if selected_type_filter != "All":
            metadata_filter["document_type"] = selected_type_filter

        with st.form("question_form", clear_on_submit=True):
            query = st.text_input("Nhập câu hỏi của bạn về tài liệu:")
            submitted = st.form_submit_button("Gửi câu hỏi")
        if submitted and query is not None and query.strip() != "":
            result = answer_query_with_sources(
                vector_store,
                query,
                chat_history=st.session_state.chat_history,
                metadata_filter=metadata_filter,
            )
            answer = result.get("answer", "")
            sources = result.get("sources", [])

            source_documents = sorted({(s.get("metadata") or {}).get("file_name", "Unknown") for s in sources})
            if source_documents:
                st.caption(f"Trả lời dựa trên tài liệu: {', '.join(source_documents)}")

            # Lưu câu hỏi và câu trả lời vào lịch sử
            st.session_state.chat_history.append(
                {
                    "question": query.strip(),
                    "answer": answer,
                    "sources": sources,
                    "timestamp": datetime.now().isoformat()
                }
            )
            st.session_state.selected_history_index = len(st.session_state.chat_history) - 1
            
            # Lưu lịch sử vào file
            save_history_to_file(st.session_state.current_file_identifier, st.session_state.chat_history)
            st.rerun()
           
if __name__ == "__main__":
    main()