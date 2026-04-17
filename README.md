# SmartDoc AI

SmartDoc AI là một ứng dụng hỏi đáp tài liệu theo mô hình RAG, xây dựng bằng Streamlit. Người dùng có thể tải lên file PDF hoặc DOCX, hệ thống sẽ trích xuất nội dung, chia nhỏ văn bản, tạo vector database bằng FAISS, sau đó dùng LLM chạy qua Ollama để trả lời câu hỏi dựa trên nội dung tài liệu. Ứng dụng cũng lưu lịch sử hội thoại theo từng tài liệu và hiển thị nguồn tham chiếu cho từng câu trả lời.

## Tính năng chính

- Tải lên một hoặc nhiều file PDF/DOCX cùng lúc.
- Trích xuất nội dung văn bản từ tài liệu.
- Chia nhỏ văn bản thành các đoạn phù hợp để truy hồi ngữ cảnh.
- Tạo vector store bằng FAISS với embedding `sentence-transformers/all-MiniLM-L6-v2`.
- Trả lời câu hỏi bằng mô hình `qwen2.5:7b` thông qua Ollama.
- Hiển thị nguồn tham chiếu, trang và dòng văn bản được dùng để trả lời.
- Lưu lịch sử chat theo từng tài liệu trong thư mục `chat_history/`.
- Có bộ lọc theo tên file và loại tài liệu.

## Cấu trúc thư mục

- `app.py`: giao diện Streamlit chính.
- `DocumentPipeline/document_readers.py`: đọc nội dung PDF và DOCX.
- `DocumentPipeline/text_chunking.py`: chia nhỏ văn bản theo chunk.
- `DocumentPipeline/pdf_chunk_metadata.py`: chia chunk PDF kèm metadata trang/dòng.
- `DocumentPipeline/rag_service.py`: tạo vector store và xử lý truy vấn RAG.
- `DocumentPipeline/model_llm.py`: lớp trung gian xuất các hàm xử lý chính.
- `chat_history/`: nơi lưu lịch sử hỏi đáp dưới dạng JSON.
- `doccument/`: thư mục chứa tài liệu mẫu hoặc tài liệu phục vụ thử nghiệm.

## Yêu cầu hệ thống

- Windows 10/11.
- Python 3.10 hoặc mới hơn.
- Ollama đã cài đặt và đang chạy.
- Có kết nối Internet ở lần chạy đầu để tải mô hình embedding và LLM.

## Cài đặt

### 1. Tạo môi trường ảo

Mở PowerShell tại thư mục dự án và chạy:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn việc kích hoạt môi trường ảo, chạy thêm:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Cài đặt thư viện

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Cài Ollama và tải mô hình

Ứng dụng đang cấu hình dùng các mô hình sau:

- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- LLM: `qwen2.5:7b`

Cài Ollama từ trang chính thức, sau đó tải mô hình:

```powershell
ollama pull qwen2.5:7b
```

Nếu Ollama chưa tự chạy service, hãy mở Ollama lên trước khi chạy ứng dụng.

## Chạy ứng dụng

Từ thư mục gốc của dự án, chạy:

```powershell
streamlit run app.py
```

Sau đó mở trình duyệt tại địa chỉ mà Streamlit hiển thị, thường là `http://localhost:8501`.

## Cách sử dụng

1. Mở ứng dụng Streamlit.
2. Tải lên một hoặc nhiều file PDF/DOCX.
3. Chờ hệ thống đọc tài liệu, chia chunk và tạo vector store.
4. Chọn kích thước chunk và độ chồng lấn nếu muốn tinh chỉnh kết quả.
5. Chọn bộ lọc theo tên file hoặc loại tài liệu nếu cần giới hạn phạm vi truy vấn.
6. Nhập câu hỏi vào ô chat và gửi.
7. Xem câu trả lời, nguồn trích dẫn, trang và dòng liên quan.
8. Lịch sử hỏi đáp sẽ được lưu theo từng tài liệu và có thể xem lại ở sidebar.

## Cơ chế hoạt động

Ứng dụng hoạt động theo quy trình RAG như sau:

1. Tài liệu được tải lên từ giao diện Streamlit.
2. Hệ thống trích xuất văn bản từ PDF/DOCX.
3. Văn bản được chia nhỏ thành các chunk.
4. Mỗi chunk được chuyển thành embedding bằng mô hình `all-MiniLM-L6-v2`.
5. FAISS lưu các vector để phục vụ truy hồi ngữ cảnh.
6. Khi người dùng đặt câu hỏi, hệ thống truy hồi các chunk liên quan nhất.
7. Prompt được tạo với ngữ cảnh, lịch sử hội thoại gần đây và câu hỏi hiện tại.
8. Ollama sinh câu trả lời dựa trên ngữ cảnh đã truy hồi.
9. Ứng dụng hiển thị câu trả lời kèm nguồn tham chiếu và lưu lịch sử vào JSON.

## Lưu ý quan trọng

- Với file PDF là ảnh scan, ứng dụng hiện chưa có OCR nên có thể không trích xuất được văn bản.
- Lần chạy đầu tiên có thể chậm vì phải tải mô hình embedding và mô hình LLM.
- Nếu máy yếu, bạn có thể đổi sang model Ollama nhẹ hơn trong `DocumentPipeline/rag_service.py`.
- File lịch sử chat được lưu trong `chat_history/` theo tên tài liệu đã chuẩn hóa.

## Tùy chỉnh nhanh

Nếu muốn thay đổi mô hình hoặc tham số chunk, chỉnh trong `DocumentPipeline/rag_service.py` và `app.py`:

- `EMBEDDING_MODEL_NAME`
- `OLLAMA_MODEL_NAME`
- `DEFAULT_TEMPERATURE`
- `DEFAULT_TOP_K`
- `chunk_size`
- `chunk_overlap`

## Xử lý sự cố

- Nếu không mở được ứng dụng, kiểm tra lại đã kích hoạt đúng virtual environment và đã cài đủ thư viện.
- Nếu báo lỗi không kết nối được Ollama, hãy đảm bảo Ollama đang chạy và model `qwen2.5:7b` đã được tải.
- Nếu câu trả lời không tốt, thử điều chỉnh `chunk_size` và `chunk_overlap` để cải thiện phần ngữ cảnh truy hồi.
- Nếu tài liệu không ra nội dung, hãy kiểm tra file gốc có phải là PDF scan ảnh hay không.

## Gợi ý chạy nhanh

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen2.5:7b
streamlit run app.py
```
