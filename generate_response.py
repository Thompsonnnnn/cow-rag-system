import requests

# ✅ 呼叫本地 Ollama 模型
def call_ollama(prompt: str, model: str = "mistral:instruct") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

# ✅ 組裝 Prompt（從相關段落 + 問題）
def build_prompt(context_chunks: list[str], question: str) -> str:
    context_block = "\n\n".join(context_chunks)
    return (
        "Answer the question based only on the following context:\n"
        f"{context_block}\n\n"
        f"Answer the question: {question}"
    )

# ✅ 回答問題主流程
def answer_question_with_context(chunks: list, question: str, threshold: float = 0.8) -> dict:
    """
    chunks: List of (Document, similarity_score)
    similarity_score 越低越相關
    """
    # 印出每個 chunk 的分數（debug 用）
    for i, (doc, score) in enumerate(chunks):
        print(f"🔎 Chunk {i+1} score: {score:.4f} | ID: {doc.metadata.get('id')}")

    if not chunks:
        # 沒有檢索結果 → 直接回答
        prompt = f"Answer the following question:\n\n{question}"
        answer = call_ollama(prompt)
        return {
            "answer": answer,
            "sources": [],
            "note": "No retrieved documents. Answered directly by LLM."
        }

    # 所有 chunks 都不符合相似度門檻 → 無意義檢索
    if all(score > threshold for _, score in chunks):
        prompt = f"Answer the following question:\n\n{question}"
        answer = call_ollama(prompt)
        return {
            "answer": answer,
            "sources": [],
            "note": "Retrieved chunks not relevant. Answered directly by LLM."
        }

    # 留下相似度高（score ≤ threshold）的 chunks
    filtered_chunks = [(doc, score) for doc, score in chunks if score <= threshold]
    context_texts = [doc.page_content for doc, _ in filtered_chunks]
    source_ids = [doc.metadata.get("id", "unknown") for doc, _ in filtered_chunks]

    #context_texts = [doc.page_content for doc, _ in chunks]
    #source_ids = [doc.metadata.get("id", "unknown") for doc, _ in chunks]

    # 產生 Prompt 並呼叫 LLM
    prompt = build_prompt(context_texts, question)
    answer = call_ollama(prompt)

    return {
    "answer": answer,
    "sources": source_ids,
    "note": " Answer generated based on relevant retrieved context.",
    "used_chunks": context_texts  # <--- 這句很重要
    }