import os
import re
import pandas as pd
import pickle  
from docx import Document as DocxDocument
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "./Data"  # 相對於項目根目錄

# ---------- 清理與 chunking ----------
def clean_text(text):
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text, source_path, chunk_size=800, overlap=80):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.create_documents([text], metadatas=[{"source": source_path}])

def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(docs)

# ---------- 各格式載入 ----------
def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return clean_text(f.read())

def load_excel_file(path):
    df = pd.read_excel(path)
    texts = []
    for _, row in df.iterrows():
        row_text = ' '.join(str(cell) for cell in row if pd.notnull(cell))
        texts.append(clean_text(row_text))
    return " ".join(texts)

def load_docx_file(path):
    doc = DocxDocument(path)
    full_text = [para.text for para in doc.paragraphs]
    return clean_text("\n".join(full_text))

def load_pdf_files(directory) -> list[Document]:
    loader = PyPDFDirectoryLoader(directory)
    return loader.load()

# ---------- 檔案處理 ----------
def load_and_chunk_file(file_path) -> list[Document]:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        text = load_text_file(file_path)
        return chunk_text(text, source_path=file_path)
    elif ext in [".xlsx", ".xls"]:
        text = load_excel_file(file_path)
        return chunk_text(text, source_path=file_path)
    elif ext == ".docx":
        text = load_docx_file(file_path)
        return chunk_text(text, source_path=file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        print(f" 跳過圖片（尚未支援 OCR) {file_path}")
        return []
    else:
        print(f"不支援的檔案格式：{file_path}")
        return []

# ---------- 唯一 ID & 去重 ----------
def assign_unique_chunk_ids(chunks: list[Document]) -> list[Document]:
    output = []
    seen_ids = set()
    chunk_counter_per_source_page = {}

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", None)
        key = f"{source}:{page}" if page is not None else f"{source}"

        if key not in chunk_counter_per_source_page:
            chunk_counter_per_source_page[key] = 0
        else:
            chunk_counter_per_source_page[key] += 1

        chunk_index = chunk_counter_per_source_page[key]
        chunk_id = f"{key}:{chunk_index}"
        chunk.metadata["id"] = chunk_id

        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            output.append(chunk)

    return output

# ---------- 主程式 ----------
def main():
    all_chunks = []

    # Step 1: 處理 PDF
    pdf_chunks = chunk_documents(load_pdf_files(DATA_PATH))
    all_chunks.extend(pdf_chunks)

    # Step 2: 處理其他格式
    # for fname in os.listdir(DATA_PATH):
    #     fpath = os.path.join(DATA_PATH, fname)
    #     if not os.path.isfile(fpath) or fname.lower().endswith(".pdf"):
    #         continue
    #     try:
    #         chunks = load_and_chunk_file(fpath)
    #         all_chunks.extend(chunks)
    #         print(f"✅ {fname} 處理成功，共 {len(chunks)} chunks")
    #     except Exception as e:
    #         print(f"❌ {fname} 處理失敗：{e}")
            
    # 加上唯一 ID（為後續檢索與追蹤用途）
    all_chunks = assign_unique_chunk_ids(all_chunks)
    print(f"\n📦 總共處理完成的 chunks 數量：{len(all_chunks)}")

    for i, chunk in enumerate(all_chunks[:2]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk.page_content[:200])
        print("ID:", chunk.metadata["id"])
        print("Metadata:", chunk.metadata)
    
    with open("chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print("\n📝 chunks 已儲存到 chunks.pkl，可供 get_data_embedding.py 使用")

if __name__ == "__main__":
    main()