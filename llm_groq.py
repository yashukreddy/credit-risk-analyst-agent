import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def get_llm(model: str = "qwen/qwen3-32b", temperature: float = 0.2) -> ChatGroq:
    """
    Returns a LangChain chat model backed by Groq API.
    
    Args:
        model: Groq model name (default: llama-3.1-8b-instant)
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
    
    Returns:
        ChatGroq: LangChain-compatible Groq chat model
    """
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("❌ Missing GROQ_API_KEY in .env file")

    return ChatGroq(
        model=model,
        temperature=temperature,
        max_retries=2,
        timeout=60,
    )


if __name__ == "__main__":
    # Quick smoke test
    print("🧪 Testing Groq LLM connection...")
    
    try:
        llm = get_llm()
        response = llm.invoke([("human", "tell me about mahesh babu")])
        print(f"✅ Groq Response: {response.content}")
        print(f"✅ LangChain integration working!")
    except Exception as e:
        print(f"❌ Error: {e}")
