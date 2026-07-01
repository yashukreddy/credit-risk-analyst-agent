import os
import logging
import hashlib
from typing import List
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from vector_store import get_vector_store
from s3_utils import upload_file_to_s3
from parsers import parse_document_local

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_chunk_id(customer_id: str, filename: str, chunk_index: int) -> str:
    """
    Generate deterministic chunk ID to prevent duplicates.
    Format: customer_id_filename_chunk_index
    
    Args:
        customer_id: Customer identifier
        filename: Original filename
        chunk_index: Sequential chunk number (0, 1, 2, ...)
    
    Returns:
        str: Deterministic ID like "C001_bankstatement.pdf_0"
    """
    # Sanitize filename to remove special characters
    safe_filename = filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return f"{customer_id}_{safe_filename}_{chunk_index}"


def process_and_index_files(uploaded_files, customer_id: str) -> bool:
    """
    Full ingestion pipeline:
    1. Upload each file to S3 (archival)
    2. Parse text content (PyMuPDF)
    3. Chunk into smaller pieces
    4. Embed and store in Pinecone with deterministic IDs
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        customer_id: Customer ID to tag all chunks with
        
    Returns:
        bool: True if at least one document was successfully indexed
    """
    if not uploaded_files:
        logger.warning("No files provided for ingestion")
        return False

    # Initialize text splitter (chunk size optimized for all-MiniLM-L6-v2)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,        # Safe chunk size for embeddings
        chunk_overlap=50,      # Small overlap to preserve context
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    documents_to_index: List[Document] = []
    chunk_ids: List[str] = []

    for uploaded_file in uploaded_files:
        try:
            filename = uploaded_file.name
            logger.info(f"📂 Processing {filename}...")

            # --- STEP 1: Archive to S3 ---
            uploaded_file.seek(0)
            s3_key = upload_file_to_s3(uploaded_file, customer_id)
            
            if not s3_key:
                logger.warning(f"⚠️ Skipping {filename}: S3 upload failed")
                continue

            s3_path = f"s3://{os.getenv('AWS_BUCKET_NAME')}/{s3_key}"
            logger.info(f"✅ Uploaded to S3: {s3_path}")

            # --- STEP 2: Parse document text ---
            uploaded_file.seek(0)
            raw_text = parse_document_local(uploaded_file)

            if not raw_text or len(raw_text.strip()) < 10:
                logger.warning(f"⚠️ Skipping {filename}: No text extracted")
                continue

            logger.info(f"✅ Extracted {len(raw_text)} characters from {filename}")

            # --- STEP 3: Create a single Document object with metadata ---
            doc = Document(
                page_content=raw_text,
                metadata={
                    "customer_id": customer_id,
                    "filename": filename,
                    "s3_path": s3_path,
                    "type": "financial_doc",
                    "ingested_at": datetime.utcnow().isoformat()
                }
            )

            # --- STEP 4: Chunk the document ---
            chunks = text_splitter.split_documents([doc])
            logger.info(f"✅ Split {filename} into {len(chunks)} chunks")

            # --- STEP 5: Generate deterministic IDs for each chunk ---
            for chunk_index, chunk in enumerate(chunks):
                chunk_id = generate_chunk_id(customer_id, filename, chunk_index)
                chunk_ids.append(chunk_id)
                documents_to_index.append(chunk)

            logger.info(f"✅ Generated {len(chunks)} deterministic IDs for {filename}")

        except Exception as e:
            logger.error(f"❌ Error processing {filename}: {e}")
            continue

    # --- STEP 6: Batch upload to Pinecone with IDs ---
    if documents_to_index:
        logger.info(f"🚀 Indexing {len(documents_to_index)} chunks to Pinecone...")
        logger.info(f"📌 Sample IDs: {chunk_ids[:3]}...")
        
        try:
            vector_store = get_vector_store()
            
            # add_documents with explicit IDs (overwrites existing chunks with same ID)
            vector_store.add_documents(
                documents=documents_to_index,
                ids=chunk_ids
            )
            
            logger.info(f"✅ Successfully indexed {len(documents_to_index)} chunks!")
            logger.info(f"✅ Re-uploading same files will UPDATE existing chunks (no duplicates)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pinecone indexing error: {e}")
            return False
    else:
        logger.warning("⚠️ No documents to index")
        return False


if __name__ == "__main__":
    # Manual test
    print("🧪 Ingestion module loaded")
    print("Example deterministic ID:", generate_chunk_id("C001", "bank_statement.pdf", 0))
    print("Expected: C001_bank_statement.pdf_0")
