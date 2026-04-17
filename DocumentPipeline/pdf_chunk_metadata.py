from DocumentPipeline.text_chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    create_text_splitter,
)


def _find_chunk_bounds(page_text: str, chunk: str, search_pos: int) -> tuple:
    start_char = page_text.find(chunk, search_pos)
    if start_char == -1:
        start_char = page_text.find(chunk)

    end_char = start_char + len(chunk) if start_char >= 0 else -1
    next_search_pos = start_char + 1 if start_char >= 0 else search_pos
    return start_char, end_char, next_search_pos


def _compute_line_range(page_lines: list, start_char: int, end_char: int) -> tuple:
    start_line = 1
    end_line = 1
    char_count = 0

    for line_idx, line in enumerate(page_lines, start=1):
        line_end = char_count + len(line) + 1
        if char_count <= start_char < line_end:
            start_line = line_idx
        if char_count < end_char <= line_end:
            end_line = line_idx
        char_count = line_end

    return start_line, end_line


def split_pdf_pages_with_metadata(
    pdf_pages: list,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple:
    """Split PDF text page-by-page and attach line-level metadata for tracing."""
    text_splitter = create_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = []
    metadatas = []

    for item in pdf_pages or []:
        page_num = item.get("page")
        page_text = (item.get("text") or "").strip()
        if not page_text:
            continue

        page_chunks = text_splitter.split_text(page_text)
        page_lines = page_text.split("\n")
        search_pos = 0

        for chunk in page_chunks:
            start_char, end_char, search_pos = _find_chunk_bounds(page_text, chunk, search_pos)
            start_line, end_line = _compute_line_range(page_lines, start_char, end_char)

            chunks.append(chunk)
            metadatas.append(
                {
                    "page": page_num,
                    "start_line": start_line,
                    "end_line": end_line,
                }
            )

    print(f"Number of chunks: {len(chunks)}")
    if chunks:
        print(f"First chunk preview: {chunks[0]}")

    return chunks, metadatas
