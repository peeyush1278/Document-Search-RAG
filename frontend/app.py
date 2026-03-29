import streamlit as st
import requests
import os

# Define the backend URL depending on environment
# In docker, the frontend reaches the backend at "http://backend:8000"
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Enterprise Local RAG", page_icon="🏢", layout="wide")

# Custom CSS for UI Polishing
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 900px;}
    
    /* Soft hover transitions for buttons */
    .stButton>button {
        border-radius: 6px;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

st.title("🏢 Enterprise Knowledge Base (Powered by Groq)")
st.markdown("Upload a PDF document and ask questions. Lightning-fast responses provided by the Groq API!")

# Ensure session states exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
if "uploading" not in st.session_state:
    st.session_state.uploading = False

with st.sidebar:
    st.header("⚙️ Configuration")
    
    groq_model = st.selectbox("🧠 Select Model Engine", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"], index=0)
    
    with st.expander("🛠️ System Architecture"):
        st.markdown("""
        **Frontend:** Streamlit  
        **Backend:** FastAPI  
        **VectorDB:** Persistent ChromaDB  
        **Models:** Groq API  
        **Tracking:** MLflow  
        """)
        
    st.divider()
    
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF to the Vector Database", type=["pdf"])

    if uploaded_file is not None and not st.session_state.document_uploaded:
        # Start the file upload request
        if st.button("Process & Index Document", use_container_width=True, type="primary"):
            st.session_state.uploading = True
            
            with st.status("Processing Document 📄", expanded=True) as status:
                st.write("Extracting text and generating vectors...")
                # Prepare file for requests
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    res = requests.post(f"{BACKEND_URL}/api/upload", files=files)
                    if res.status_code == 200:
                        status.update(label="Document Indexed Successfully!", state="complete", expanded=False)
                        st.toast("✅ Document embedded securely!", icon="🚀")
                        st.session_state.document_uploaded = True
                    else:
                        status.update(label="Error processing document.", state="error", expanded=True)
                        st.error(f"Error: {res.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    status.update(label="Backend Offline", state="error", expanded=True)
                    st.error("Failed to connect to the backend server. Is it running?")
                finally:
                    st.session_state.uploading = False
                    
    elif st.session_state.document_uploaded:
        st.success(f"{uploaded_file.name} is ready for Q&A!", icon="✅")

# Display chat history or Hero Section
if len(st.session_state.messages) == 0:
    st.info("👋 Welcome! Please upload a PDF in the sidebar to begin. Then ask any question below and Groq will retrieve the answer from your document.", icon="💡")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to state and UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Request response from the backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "query": prompt,
                    "model": groq_model
                }
                res = requests.post(f"{BACKEND_URL}/api/chat", json=payload)
                
                if res.status_code == 200:
                    answer = res.json().get("answer", "")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                elif res.status_code == 400: # Could be no documents in DB
                    st.warning(res.json().get("detail", "Error 400 from server"))
                else:
                    st.error(f"Backend Error: {res.json().get('detail', 'Unknown error')}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 Failed to connect to the FastAPI backend. Make sure the backend service is running.")

