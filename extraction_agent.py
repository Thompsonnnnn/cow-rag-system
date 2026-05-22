import requests

# ✅ 呼叫 Ollama 的模型 API
def generate_response(messages: list[dict], model: str = "mistral:instruct") -> str:
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

# ✅ 泛用 LLM Agent 類別
class LLMAgent:
    def __init__(self, role_description: str, task_description: str, llm: str = "mistral:instruct"):
        self.role_description = role_description
        self.task_description = task_description
        self.llm = llm

    def inference(self, message: str) -> str:
        messages = [
            {"role": "system", "content": self.role_description},
            {"role": "user", "content": f"{self.task_description}\n{message}"}
        ]
        return generate_response(messages, model=self.llm)

# ✅ Question Refinement Agent：轉換模糊問題為明確英文問題
question_refinement_agent = LLMAgent(
    role_description=(
        "You are a helpful assistant specialized in dairy cow health and farm management. "
        "Your job is to rewrite vague or colloquial questions from farmers into clear, specific English questions"
    ),
    task_description="Rewrite the following input into a clear and specific English question . Do not explain or add anything. Just return the refined question."
)

# ✅ Keyword Extraction Agent：從問題中萃取向量搜尋用的關鍵詞
keyword_extraction_agent = LLMAgent(
    role_description=(
        "You are an assistant that extracts the most relevant technical keywords from questions about dairy cow health, feeding, reproduction, and milk production. "
        "Your goal is to help a retrieval system find relevant documents."
    ),
    task_description="Extract the most important keywords from the question. Return them as a comma-separated list. No explanations."
)

# ✅ QA Agent：以上下文回答與乳牛相關問題
qa_agent = LLMAgent(
    role_description=(
        "You are a knowledgeable assistant specialized in dairy cow health, welfare, and farm management. "
        "Your expertise includes feeding systems, heat stress prevention, lameness detection, mastitis diagnosis, milk production optimization, and reproduction management. "
        "You help answer technical questions from dairy farmers using scientific knowledge and best practices. "
        "You only use the context provided and avoid hallucinations. Your goal is to assist farmers by providing reliable and practical information extracted from academic research and industry reports."
    ),
    task_description="Answer the following question using only the context provided. Do not guess if the answer is not in the context."
)

# ✅ 對外介面
def refine_question(text: str) -> str:
    return question_refinement_agent.inference(text)

def extract_keywords(text: str) -> list[str]:
    keywords_str = keyword_extraction_agent.inference(text)
    return [kw.strip() for kw in keywords_str.split(",") if kw.strip()]

def answer_question(question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": qa_agent.role_description},
        {"role": "user", "content": f"{qa_agent.task_description}\n\nContext:\n{context}\n\nQuestion:\n{question}"}
    ]
    return generate_response(messages, model=qa_agent.llm)