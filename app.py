import os
import time
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from io import StringIO, BytesIO
import random
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import pypdf.errors

st.set_page_config(page_title="PageMaster Chatbot", page_icon="⚖️", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fa;
        color: #333333;
        font-family: Arial, sans-serif;
    }
    .header {
        text-align: center;
        font-size: 3em;
        color: #0d47a1;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    .caption {
        text-align: center;
        font-size: 1.2em;
        color: #555555;
        font-style: italic;
        margin-bottom: 25px;
    }
    .full-width-image {
        margin: 0 auto;
        width: 100%;
        height: auto;
        display: block;
        margin-top: 20px;
    }
    div.stButton > button:first-child {
        background-color: #0d47a1;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px 20px;
    }
    div.stButton > button:active {
        background-color: #0b3954;
    }
    .stChatMessageUser {
        background-color: #bbdefb;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .stChatMessageAssistant {
        background-color: #e1f5fe;
        color: #333333;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .sidebar-heading {
        font-size: 1.5em;
        color: #0d47a1;
        font-weight: bold;
        margin-bottom: 10px;
        padding-top: 20px;
    }
    div[data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 2px solid #0d47a1;
        border-radius: 12px;
        padding: 10px;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #1565c0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    input[type="file"] {
        color: #333333;
    }
    .sidebar-hr {
        border: 0;
        height: 2px;
        background: #0d47a1;
        margin: 10px 0;
        width: 90%;
        margin-left: auto;
        margin-right: auto;
    }
    .file-upload-section {
        background-color: #eceff1;
        padding: 5px 15px;
        border-radius: 8px;
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        color: #555555;
        padding: 20px 0;
        margin-top: 20px;
        border-top: 1px solid #0d47a1;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDecoration {display:none;}
    button[title="View fullscreen"] {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="header">PageMaster Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="caption">Unleash the Power of Your PDFs with PageMaster</div>', unsafe_allow_html=True)

image_path = "bot1.png"
if os.path.exists(image_path):
    st.image(image_path, use_container_width=True, output_format="JPEG")
else:
    st.warning("Image 'bot1.png' not found. Skipping image display.")

def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

def export_chat_logs():
    chat_log = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.messages])
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("PageMaster Chat Log", styles['Title']), Spacer(1, 12)]
    for line in chat_log.split('\n'):
        story.append(Paragraph(line, styles['Normal']))
        story.append(Spacer(1, 6))
    doc.build(story)
    buffer.seek(0)
    return buffer

def preprocess_and_sample_content(texts, max_input_tokens=4000):
    cleaned_texts = [re.sub(r'\s+', ' ', doc.page_content).strip() for doc in texts]
    total_chars = sum(len(text) for text in cleaned_texts)
    approx_tokens = total_chars // 4
    if approx_tokens <= max_input_tokens:
        return " ".join(cleaned_texts)
    num_chunks = len(cleaned_texts)
    sample_size = min(num_chunks, max_input_tokens // (total_chars // num_chunks))
    sampled_indices = sorted(random.sample(range(num_chunks), sample_size))
    return " ".join(cleaned_texts[i] for i in sampled_indices)

def filter_response(response, context, question):
    question_terms = set(question.lower().split())
    context_lower = context.lower()
    if not any(term in context_lower for term in question_terms if len(term) > 3):
        if not any(term in response.lower() for term in question_terms if len(term) > 3):
            return "I don’t have sufficient information from the uploaded document to answer this question precisely."
    return response

def suggest_questions(vector_db):
    if not vector_db:
        return []
    all_docs = list(vector_db.docstore._dict.values())
    full_content = " ".join([doc.page_content.lower() for doc in all_docs])
    suggestions = []
    if any(term in full_content for term in ["revenue", "sales", "profit", "income", "loss", "earnings"]):
        suggestions.append("What financial performance metrics are mentioned in the document?")
    if any(term in full_content for term in ["purpose", "goal", "objective", "mission", "aim", "intent"]):
        suggestions.append("What is the stated purpose or objective in the document?")
    if any(term in full_content for term in ["date", "year", "quarter", "period", "month", "timeline"]):
        suggestions.append("What time period or dates are referenced in the document?")
    if any(term in full_content for term in ["key", "main", "important", "significant", "critical", "major"]):
        suggestions.append("What are the most significant points highlighted in the document?")
    while len(suggestions) < 3:
        if "what" not in [q.lower() for q in suggestions]:
            suggestions.append("What is the main topic of the document?")
        elif "who" not in [q.lower() for q in suggestions]:
            suggestions.append("Who is the intended audience of the document?")
        else:
            suggestions.append("What details stand out in the document?")
    return suggestions[:4]

def process_question(qa, question, db_retriever):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("assistant"):
        with st.spinner("Thinking 💡..."):
            result = qa.invoke(input=question)
            context = " ".join([doc.page_content for doc in db_retriever.get_relevant_documents(question)])
            filtered_response = filter_response(result["answer"], context, question)
            message_placeholder = st.empty()
            message_placeholder.markdown(filtered_response)
        st.session_state.messages.append({"role": "assistant", "content": filtered_response})
    return filtered_response

if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=2)
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "summary_length" not in st.session_state:
    st.session_state.summary_length = 200
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

st.sidebar.markdown('<div class="sidebar-heading">Settings</div>', unsafe_allow_html=True)
summary_length = st.sidebar.slider(
    "Select Summary Length (tokens)", 
    min_value=100, 
    max_value=500, 
    value=st.session_state.summary_length, 
    step=50
)
st.session_state.summary_length = summary_length

st.sidebar.markdown('<hr class="sidebar-hr">', unsafe_allow_html=True)
st.sidebar.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-heading">File Upload</div>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None and st.session_state.processed_file != uploaded_file.name:
    temp_dir = "temp_uploaded_files"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Processing uploaded file and generating summary..."):
        st.session_state.vector_db = None
        st.session_state.memory.clear()
        st.session_state.messages = []
        st.session_state.suggested_questions = []

        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        except pypdf.errors.DependencyError as e:
            st.error("This PDF requires encryption support. Please ensure 'pycryptodome' is installed.")
            raise
        except pypdf.errors.PdfReadError as e:
            st.error("This PDF might be encrypted or corrupted. Try uploading a different file.")
            raise
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            raise

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="nomic-ai/nomic-embed-text-v1",
                model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"}
            )
        except ImportError as e:
            st.error(f"Failed to initialize embeddings due to missing dependency: {str(e)}")
            raise
        except Exception as e:
            st.error(f"Error initializing embeddings: {str(e)}")
            raise
        st.session_state.vector_db = FAISS.from_documents(texts, embeddings)
        st.session_state.vector_db.save_local("10Q_vector_db")

        summary_prompt_template = """
        <s>[INST] Summarize the following document content in a concise paragraph. Focus on the key points and overall purpose of the document. Limit the summary to approximately {max_tokens} tokens. Return only the summary text:

        {document_content}
        </s>[INST]
        """
        document_content = preprocess_and_sample_content(texts, max_input_tokens=4000)
        summary_prompt = PromptTemplate(template=summary_prompt_template, input_variables=["document_content", "max_tokens"])
        llm = Together(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            temperature=0.3,
            max_tokens=st.session_state.summary_length,
            together_api_key='d5b19df5283427f0301b6a6e23d463cebcbe8ba62ca2079a61a14d1897d147ae'
        )
        summary_chain = summary_prompt | llm
        raw_summary = summary_chain.invoke({"document_content": document_content, "max_tokens": st.session_state.summary_length})
        cleaned_summary = re.sub(r'<[^>]+>|□|\s+', ' ', raw_summary).strip()
        st.session_state.summary = cleaned_summary if cleaned_summary else "Unable to generate a summary."
        st.session_state.suggested_questions = suggest_questions(st.session_state.vector_db)
        os.remove(file_path)
        st.session_state.processed_file = uploaded_file.name
    st.success(f"File '{uploaded_file.name}' processed and added to the database!")

if st.session_state.summary is not None:
    st.markdown("### Document Summary")
    st.markdown(f'{st.session_state.summary}', unsafe_allow_html=True)
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("Document Summary", styles['Title']), Spacer(1, 12), Paragraph(st.session_state.summary, styles['Normal'])]
    doc.build(story)
    buffer.seek(0)
    st.download_button(label="Export Summary", data=buffer, file_name="summary.pdf", mime="application/pdf", key="export_summary")

if st.session_state.vector_db is not None:
    db_retriever = st.session_state.vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
else:
    db_retriever = None

prompt_template = """
<s>[INST] You are PageMaster, a chatbot trained on the content of uploaded PDF files. Your purpose is to answer questions accurately based on the document while providing helpful insights when direct answers aren’t fully available.

### Instructions:
- Base your answers primarily on the document content ({context}).
- If the exact answer isn’t explicitly stated, use the context to provide a reasoned response that aligns with the document’s intent or information.
- Only say "I don’t have sufficient information from the uploaded document to answer this question" if the question is completely unrelated to the document content.
- Keep responses concise, clear, and professional.
- If the query is unclear, ask for clarification (e.g., "Could you please specify...?").

### Inputs:
- **CONTEXT**: {context}
- **CHAT HISTORY**: {chat_history}
- **QUESTION**: {question}

### Response:
Provide a direct answer or a reasoned response based on the document content. Use the fallback statement only when the question is entirely outside the document’s scope.
</s>[INST]
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

TOGETHER_AI_API = 'd5b19df5283427f0301b6a6e23d463cebcbe8ba62ca2079a61a14d1897d147ae'
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.3,
    max_tokens=512,
    together_api_key=TOGETHER_AI_API
)

if st.session_state.vector_db is not None:
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        retriever=db_retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
else:
    qa = None

if st.session_state.vector_db is not None:
    st.markdown("### Chat with PageMaster")
    
    if st.session_state.suggested_questions:
        st.markdown("#### Suggested Questions")
        for i, suggestion in enumerate(st.session_state.suggested_questions):
            if st.button(suggestion, key=f"suggest_{i}"):
                process_question(qa, suggestion, db_retriever)

    for message in st.session_state.messages:
        if message.get("role") == "user":
            st.chat_message("user").write(message.get("content"))
        else:
            st.chat_message("assistant").write(message.get("content"))

    input_prompt = st.chat_input("Ask a question about the uploaded document...")
    if input_prompt:
        if qa is not None:
            process_question(qa, input_prompt, db_retriever)
        else:
            st.error("Please upload a file to enable the chatbot.")
    
    # Buttons moved below the chat interface
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button('Reset All Chat 🗑️', on_click=reset_conversation, key="reset_chat")
    with col2:
        chat_log_buffer = export_chat_logs()
        st.download_button(
            label="Export Chat Log",
            data=chat_log_buffer,
            file_name="chat_log.pdf",
            mime="application/pdf",
            key="export_chat_log"
        )

# Footer section
st.markdown('<div class="footer">PageMaster © 2025 All Rights Reserved Srujan Kothuri</div>', unsafe_allow_html=True)
