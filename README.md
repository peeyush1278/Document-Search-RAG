# 🏢 Enterprise RAG Architecture (The "Star Project")

A completely **free, offline, and private** Retrieval-Augmented Generation (RAG) application utilizing an Enterprise Microservice Architecture. 

This project proves proficiency in modern Generative AI, MLOps, backend APIs, Containerization, and persistent Vector Databases.

---

## 🌟 Features
- **100% Private & Free**: No API keys required. Your data never leaves your computer.
- **Microservices Architecture**: Separated FastAPI backend and Streamlit frontend.
- **Persistent Vector DB**: Uses `ChromaDB` to save document embeddings instead of keeping them in memory.
- **Local Models**: Leverages `Ollama` for fast, private LLM execution.
- **MLOps Tracking**: Uses `MLflow` to natively trace every user query and context retrieval operation for debugging and analytics.
- **Dockerized**: Fully containerized using `docker-compose`.

## 🛠️ Tech Stack
* **Frontend:** Streamlit, `requests`
* **API Backend:** FastAPI, Uvicorn
* **Data Parsing & Chunking:** PyPDF, LangChain
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (`sentence-transformers`)
* **Vector Database:** ChromaDB
* **LLM Engine:** Ollama (Host-Machine)
* **MLOps / Tracing:** MLflow
* **Containerization:** Docker Desktop

---

## 🚀 Getting Started

### 1. Install & Run Ollama (Host Machine)
Because models are heavy and hardware acceleration (GPU) is easier to utilize on your native machine, you will run Ollama on your computer, not inside Docker.

Install [Ollama](https://ollama.com/) on your machine.
Open a terminal and pull a lightweight model:
```bash
ollama run llama3.2:1b
```

### 2. Start the Docker Services
Ensure you have **Docker Desktop** installed and running on your PC.

Open a PowerShell terminal in the root folder of this project (where `docker-compose.yml` is) and run:
```powershell
docker-compose up --build -d
```
*(This builds the python environments and starts the services in detached mode. It may take 1-3 minutes the first time).*

### 3. Access the Application
Once the containers are running, you can access your enterprise layout:

* **Frontend UI (Streamlit):** [http://localhost:8501](http://localhost:8501)
* **MLOps Tracking (MLflow):** [http://localhost:5000](http://localhost:5000)
* **API Swagger Docs (FastAPI):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 💡 Usage Workflow
1. Open the [Streamlit Frontend](http://localhost:8501).
2. Select your `Ollama Model` (make sure you downloaded it in step 1).
3. Upload a PDF document. (The frontend will send an API request to the backend, which will parse, chunk, and save it permanently to ChromaDB).
4. Chat with your document!
5. Open the [MLflow Tracking Server](http://localhost:5000) to see complete logs of your conversation, including latency metrics and retrieved document chunks!

## 🛑 Shutting Down
To clean up and shut down the containers:
```powershell
docker-compose down
```
*(Your vector embeddings are saved locally in `backend/chroma_db/` and will survive standard shutdown bounds).*
