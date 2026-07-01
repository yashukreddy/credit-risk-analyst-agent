from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Returns HuggingFace embeddings model for LangChain.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        HuggingFaceEmbeddings: Ready-to-use embeddings model
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},  # Change to 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True}  # Important for cosine similarity
    )


if __name__ == "__main__":
    # Quick test
    print("🧪 Testing embeddings model...")
    
    try:
        embeddings = get_embeddings()
        test_text = "This is a test document for credit risk assessment."
        vector = embeddings.embed_query(test_text)
        
        print(f"✅ Embedding dimension: {len(vector)}")
        print(f"✅ Model: sentence-transformers/all-MiniLM-L6-v2")
        print(f"✅ First 5 values: {vector[:5]}")
    except Exception as e:
        print(f"❌ Error: {e}")
