import pickle
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings

CHROMA_PATH = "chroma"
CHUNK_FILE = "chunks.pkl"

# ✅ 支援切換式設計
def get_embedding_function():
    # 預設：HuggingFace 本地模型（免 API）
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 可切換使用 AWS Bedrock（需憑證）
    # return BedrockEmbeddings(credentials_profile_name="default", region_name="us-east-1")

    # 可切換使用本地 Ollama 模型
    # return OllamaEmbeddings(model="nomic-embed-text")


def load_chunks():
    with open(CHUNK_FILE, "rb") as f:
        return pickle.load(f)

def filter_new_chunks(chunks, db):
    existing_ids = set(db.get(include=[])["ids"])
    return [c for c in chunks if c.metadata["id"] not in existing_ids]

def main():
    print("📥 載入 chunks...")
    chunks = load_chunks()

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    new_chunks = filter_new_chunks(chunks, db)
    print(f"📊 現有資料：{len(db.get()['ids'])} 筆，將新增：{len(new_chunks)} 筆")

    if new_chunks:
        ids = [c.metadata["id"] for c in new_chunks]
        db.add_documents(new_chunks, ids=ids)
        db.persist()
        print("✅ 新資料已寫入 Chroma")
    else:
        print("✅ 無需新增資料")

if __name__ == "__main__":
    main()