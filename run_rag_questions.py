from question_set_data import QUESTION_SET
from retrieval import load_vectorstore
from generate_response import answer_question_with_context
import json

def main():
    vectorstore = load_vectorstore()
    results = []

    for idx, item in enumerate(QUESTION_SET, 1):
        question = item["question"]
        ground_truth = item["answer"]
        print(f"\n 第 {idx} 題：{question}")

        #  不再進行 refine，直接使用原始問題
        refined = question
        print(f"（Refine 關閉）使用原始問題：{refined}")

        # 1. 檢索
        top_chunks = vectorstore.similarity_search_with_score(refined, k=5)

        # 2. 回答
        result = answer_question_with_context(top_chunks, refined)
        generated_answer = result["answer"]

        # 3. 收集結果
        results.append({
            "theme": item["theme"],
            "source": item["source"],
            "question": question,
            #"refined_question": None,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "retrieved_chunks": [
                {
                    "id": doc.metadata.get("id", "unknown"),
                    "score": score,
                    #"content": doc.page_content
                }
                for doc, score in top_chunks
            ],
            "score_note": result["note"],
            #"chunk_ids": [doc.metadata.get("id") for doc, _ in top_chunks],
        })

    # 4. 儲存結果
    with open("rag_outputs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n 全部題目完成，結果已儲存到 rag_outputs.json")

if __name__ == "__main__":
    main()