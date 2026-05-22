from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ✅ 建立 embedding function（與 get_data_embedding.py 相同）
def get_embedding_function():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ✅ 初始化向量資料庫
def load_vectorstore():
    embedding_fn = get_embedding_function()
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_fn)

# ✅ 主函數：查詢並回傳 top-k 且低於相似度門檻的 chunks
def retrieve_relevant_chunks(query: str, top_k: int = 5, threshold: float = 0.8):
    db = load_vectorstore()
    results = db.similarity_search_with_score(query, k=top_k)

    # 只保留低於 threshold（即較相似）的 chunks
    filtered = [(doc, score) for doc, score in results if score < threshold]
    return filtered

# ✅ 可選：直接用 CLI 測試向量查詢
if __name__ == "__main__":
    while True:
        try:
            query = input("\n📝 請輸入查詢問題（Ctrl+C 離開）：\n> ")
            results = retrieve_relevant_chunks(query)
            if not results:
                print("⚠️ 沒有找到符合相似度門檻的段落")
            else:
                print(f"\n🔍 找到 {len(results)} 筆符合相似度門檻的段落：")
                for i, (doc, score) in enumerate(results):
                    print(f"\n--- 結果 {i+1} ---")
                    print(doc.page_content[:300])
                    print("分數:", round(score, 3))
                    print("ID:", doc.metadata.get("id", "N/A"))
        except KeyboardInterrupt:
            print("\n👋 離開向量查詢測試")
            break