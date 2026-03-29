from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import tempfile
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import mlflow
import uuid
from dotenv import load_dotenv

load_dotenv()

# --- MLflow Configuration ---
# Fallback to local ./mlruns folder if not running in Docker
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("rag_document_search")

# --- Initialize FastAPI ---
app = FastAPI(title="Local Corporate RAG API", version="1.0")

# --- Initialize Global Resources ---
# 1. Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Persistent Vector Database
# Use a relative directory so it works natively on Windows or mounted in Docker
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)

# 3. Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# 4. Groq Configuration
# Ensure GROQ_API_KEY is available in the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set. Chat features will error out.")

class ChatRequest(BaseModel):
    query: str
    model: str = "llama-3.1-8b-instant"

class ChatResponse(BaseModel):
    answer: str

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Parse PDF
        pdf_reader = PdfReader(tmp_path)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
                
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")

        # Chunk text
        chunks = text_splitter.split_text(text)
        
        # Add to ChromaDB (Automatically persists to disk)
        vectorstore.add_texts(chunks)
        
        return {"message": f"Successfully processed {file.filename} and indexed {len(chunks)} chunks."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Ensure there's something in the DB
    if vectorstore._collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Please upload a PDF first.")
        
    try:
        # Start MLflow Trace
        with mlflow.start_run(run_name=f"chat_{uuid.uuid4().hex[:8]}"):
            # Log inputs
            mlflow.log_param("query", request.query)
            mlflow.log_param("model", request.model)
            
            # Initialize Groq LLM
            llm = ChatGroq(model_name=request.model, temperature=0, groq_api_key=GROQ_API_KEY)
            
            prompt_template = """Use the following pieces of context to answer the user's question. 
Answer the question comprehensively. If the answer is not in the context, just say you don't know based on the document.

Context:
{context}

Question: {question}
Helpful Answer:"""
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            
            # Setup RetrievalQA chain
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True # get the chunks back to log them
            )
            
            # Execute Chain
            result = qa_chain.invoke({"query": request.query})
            answer = result["result"]
            source_docs = result.get("source_documents", [])
            
            # Log the retrieved context chunks to MLflow
            for i, doc in enumerate(source_docs):
                mlflow.log_text(doc.page_content, f"retrieved_chunk_{i}.txt")
                
            # Log output
            mlflow.log_text(answer, "final_answer.txt")
            
            return ChatResponse(answer=answer)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
