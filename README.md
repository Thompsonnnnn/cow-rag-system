# 🐄 Cow RAG Expert System

基於RAG的乳牛健康管理專家系統，整合LLM與向量檢索提供農民專業建議。

## 🎯 核心功能

- 問題精煉：用LLM轉換模糊問題為明確英文
- 關鍵詞提取：優化向量檢索
- 語義搜尋：使用ChromaDB進行相似度搜尋
- 智能回答：基於檢索結果生成答案

## 🛠️ 技術棧

- **LLM**: LangChain + Ollama (Mistral)
- **向量DB**: ChromaDB
- **Web UI**: Gradio
- **嵌入模型**: HuggingFace Embeddings

## 🚀 快速開始

### 環境需求
- Python 3.9+
- Ollama + Mistral 模型 (`ollama pull mistral`)

### 安裝

```bash
git clone https://github.com/Thompsonnnnn/cow-rag-system.git
cd cow-rag-system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python process_data.py  # 準備向量數據庫
```

### 運行

```bash
# CLI
python main.py

# Web UI
python UI.py
# 訪問 http://localhost:7860

# 批量評估
python run_rag_questions.py
```

## 📁 結構

```
.
├── extraction_agent.py    # LLM Agent (問題精煉、關鍵詞提取)
├── retrieval.py           # 向量檢索
├── generate_response.py   # 答案生成
├── main.py               # CLI 入口
├── UI.py                 # Gradio 界面
└── process_data.py       # 數據處理
```

## 📚 知識庫

涵蓋乳牛健康各方面：飼料管理、熱緊迫、跛行、乳房炎、乳產量、繁殖管理

## 📝 使用例

```python
from extraction_agent import refine_question, extract_keywords
from retrieval import load_vectorstore
from generate_response import answer_question_with_context

vectorstore = load_vectorstore()
refined = refine_question("乳牛熱緊迫怎麼辦")
chunks = vectorstore.similarity_search_with_score(refined, k=5)
result = answer_question_with_context(chunks, refined)
print(result["answer"])
```

## ⚙️ 配置

- **Ollama 端點**: `http://localhost:11434`
- **向量數據庫**: `./chroma`
- **嵌入模型**: `sentence-transformers/all-MiniLM-L6-v2`

## 📄 License

MIT

## 👤 Author

Thompson | 國立台灣大學 BBlab
