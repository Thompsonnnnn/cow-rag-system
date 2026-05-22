# 🐄 Cow RAG Expert System

一個專為乳牛健康管理設計的 Retrieval-Augmented Generation (RAG) 系統，整合LLM與向量檢索，為農民提供基於知識庫的專業建議。

## 🎯 核心功能

1. **問題理解** - 用LLM精煉和優化用戶問題
2. **關鍵詞提取** - 從問題中抽取關鍵詞用於檢索
3. **語義搜索** - 使用向量相似度搜尋相關文檔
4. **智能回答** - 基於檢索結果生成專業答案

## 🛠️ 技術棧

- **LLM框架**: LangChain
- **本地模型**: Ollama (Mistral)
- **向量數據庫**: ChromaDB
- **Web界面**: Gradio
- **嵌入模型**: Nomic Embed Text

## 📊 系統架構

```
用戶問題
    ↓
[問題精煉 Agent] → 轉換為清晰的英文問題
    ↓
[關鍵詞提取 Agent] → 提取檢索用的關鍵詞
    ↓
[向量檢索] → 從ChromaDB中找相似文檔
    ↓
[QA Agent] → 基於上下文生成答案
    ↓
返回結果 + 來源文檔
```

## 🚀 快速開始

### 前置條件
- Python 3.9+
- Ollama (https://ollama.ai)
- 已下載的Mistral模型: `ollama pull mistral`

### 安裝步驟

1. **複製倉庫**
```bash
git clone https://github.com/yourusername/cow-rag-system.git
cd cow-rag-system
```

2. **創建虛擬環境**
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows
```

3. **安裝依賴**
```bash
pip install -r requirements.txt
```

4. **準備數據**
```bash
python scripts/prepare_data.py
```

### 運行方式

**命令行界面:**
```bash
python scripts/main.py
```

**Web界面 (Gradio):**
```bash
python ui/gradio_ui.py
```

然後打開 `http://localhost:7860`

**批量評估:**
```bash
python scripts/run_rag_evaluation.py --questions data/sample_questions.json
```

## 📁 項目結構

```
.
├── src/                    # 核心模塊
│   ├── extraction_agent.py   # LLM Agent (問題精煉、關鍵詞提取)
│   ├── retrieval.py          # 向量檢索邏輯
│   ├── generate_response.py  # 答案生成
│   └── utils.py              # 輔助函數
├── ui/                     # Web界面
│   └── gradio_ui.py          # Gradio應用
├── scripts/                # 運行腳本
│   ├── main.py               # CLI版本
│   └── run_rag_evaluation.py # 評估腳本
├── data/                   # 示例數據
│   └── sample_questions.json # 示例問題
└── outputs/                # 示例輸出
    └── rag_results_sample.json
```

## 🔧 主要組件說明

### Extraction Agent (extraction_agent.py)
- **LLMAgent 類**: 通用LLM代理框架
- **question_refinement_agent**: 精煉問題
- **keyword_extraction_agent**: 提取關鍵詞
- **qa_agent**: 基於上下文回答

### Retrieval (retrieval.py)
- 使用ChromaDB作為向量數據庫
- 支持語義相似度搜索
- 返回相關度分數

### Response Generation (generate_response.py)
- 根據檢索結果生成答案
- 處理無相關文檔的情況
- 標記引用來源

## 📈 評估指標

系統使用以下指標評估性能：
- **BLEU 分數**: 語義相似度
- **精準度 (Precision)**: 回答的相關度
- **召回率 (Recall)**: 覆蓋程度

## 📝 使用示例

```python
from src.retrieval import load_vectorstore
from src.extraction_agent import refine_question, extract_keywords
from src.generate_response import answer_question_with_context

# 初始化
vectorstore = load_vectorstore()

# 用戶輸入
query = "乳牛熱緊迫怎麼辦"

# 執行RAG流程
refined = refine_question(query)
keywords = extract_keywords(refined)
chunks = vectorstore.similarity_search_with_score(refined, k=5)
result = answer_question_with_context(chunks, refined)

print(f"回答: {result['answer']}")
print(f"來源: {result['sources']}")
```

## 🔌 配置

在代碼中修改以下配置：

```python
# extraction_agent.py
LLM_MODEL = "mistral:instruct"  # 使用的模型
OLLAMA_ENDPOINT = "http://localhost:11434"  # Ollama服務地址

# retrieval.py
CHROMA_PATH = "./chroma"        # 向量數據庫路徑
EMBEDDING_MODEL = "nomic-embed-text"
```

## 📚 知識庫類別

系統涵蓋以下主題：
- 飼料管理 (Feeding Management)
- 熱緊迫 (Heat Stress)
- 跛行 (Lameness)
- 乳房炎 (Mastitis)
- 乳產量 (Milk Production)
- 繁殖管理 (Reproduction Management)

## 💡 開發者註記

- 確保Ollama服務運行在 `localhost:11434`
- 第一次運行會創建向量嵌入，可能耗時較長
- ChromaDB文件約占用數百MB空間
- 建議用GPU加速LLM推理

## 🐛 常見問題

**Q: Ollama連接失敗？**
A: 確保Ollama已啟動: `ollama serve`

**Q: 向量數據庫無法加載？**
A: 刪除 `chroma/` 目錄，重新運行 `prepare_data.py`

**Q: 回答質量不佳？**
A: 檢查檢索到的文檔是否相關，考慮調整 `similarity_search` 的 `k` 值

## 📄 許可證

MIT License

## 👤 作者

Thompson  
國立台灣大學 / BBlab

## 🙏 致謝

感謝提供乳牛健康知識的資源與指導。
