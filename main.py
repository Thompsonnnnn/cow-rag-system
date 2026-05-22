from extraction_agent import refine_question, extract_keywords
from retrieval import load_vectorstore
from generate_response import answer_question_with_context

def main():
    print("🐄 Cow Expert Chatbot 啟動")
    print("輸入你的問題（輸入 exit 結束）\n")

    # 預先載入向量資料庫（避免每次都重載）
    vectorstore = load_vectorstore()

    while True:
        user_input = input("👤 你：")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 再見！")
            break

        # Step 1: 精煉問題
        refined_question = refine_question(user_input)
        print(f"🔍 Refined Question: {refined_question}")

        # Step 2: 關鍵字抽取（示意用）
        keywords = extract_keywords(refined_question)
        print(f"🔑 Extracted Keywords: {keywords}")

        # Step 3: 向量檢索（附加相似度分數）
        top_chunks = vectorstore.similarity_search_with_score(refined_question, k=5)

        # Step 4: 回答問題（自動處理無結果或不相關情況）
        result = answer_question_with_context(top_chunks, refined_question)

        # Step 5: 顯示回應
        print("\n💬 回答：", result["answer"])
        #print("📄 來源：", result["sources"])
        print("📝 備註：", result["note"])

        # 🔍 額外：印出實際使用的 chunks 內容
        print("📚 引用內容：")
        for doc, score in top_chunks:
            if doc.metadata.get("id") in result["sources"]:
                print(f"\n🧩 Chunk ID: {doc.metadata.get('id')}")
                #print(f"🔢 相似度分數: {round(score, 4)}")
                print(f"📄 內容:\n{doc.page_content[:500]}...")  # 最多印前500字
                
        print("-" * 50)

if __name__ == "__main__":
    main()