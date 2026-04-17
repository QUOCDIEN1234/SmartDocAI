from DocumentPipeline.document_readers import load_docx, load_pdf, load_pdf_pages
from DocumentPipeline.pdf_chunk_metadata import split_pdf_pages_with_metadata
from DocumentPipeline.text_chunking import split_text_data
import DocumentPipeline.model_llm as model
 

if __name__ == "__main__": 
    pdf_path = "../doccument/a.pdf"
    # print(load_pdf(pdf_path))
    text = load_pdf(pdf_path)
    chunks = split_text_data(text)
    vector_store = model.create_vector_store(chunks)
    query = input("Enter your query: ")
    
    # Thực hiện truy vấn tìm kiếm tương tự
    result = model.answer_query(vector_store, query)
    print(f"Answer: {result}")