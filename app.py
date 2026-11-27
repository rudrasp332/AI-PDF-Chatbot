
import os
import hashlib

import streamlit as st
import numpy as np

from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================
# 1. STREAMLIT CONFIGURATION
# =========================================

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI PDF Chatbot")
st.caption("Upload your PDF and chat with its content.")


# =========================================
# 2. SESSION STATE
# =========================================

defaults = {
    "chat_history": [],
    "pdf_id": None,
    "chunks": [],
    "chunk_pages": [],
    "vectorizer": None,
    "tfidf_matrix": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================
# 3. GROQ CLIENT
# =========================================

@st.cache_resource
def get_groq_client(api_key):
    return Groq(api_key=api_key)


def get_groq_client_safely():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error(
            "Groq API key not found. "
            "Please configure GROQ_API_KEY."
        )
        return None

    return get_groq_client(api_key)


# =========================================
# 4. TEXT CHUNKING
# =========================================

def create_chunks(text, page_number, chunk_size=500, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "text": chunk,
                "page": page_number
            })

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# =========================================
# 5. PDF PROCESSING
# =========================================

def process_pdf(pdf_bytes):

    reader = PdfReader(
        __import__("io").BytesIO(pdf_bytes)
    )

    all_chunks = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if not page_text or not page_text.strip():
            continue

        page_chunks = create_chunks(
            page_text,
            page_number
        )

        all_chunks.extend(page_chunks)

    if not all_chunks:
        raise ValueError(
            "No readable text found in this PDF."
        )

    chunk_texts = [
        item["text"] for item in all_chunks
    ]

    chunk_pages = [
        item["page"] for item in all_chunks
    ]

    # Build lightweight TF-IDF index
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        dtype=np.float32
    )

    tfidf_matrix = vectorizer.fit_transform(
        chunk_texts
    )

    return (
        chunk_texts,
        chunk_pages,
        vectorizer,
        tfidf_matrix
    )


# =========================================
# 6. RETRIEVAL
# =========================================

def retrieve_context(question, top_k=3):

    vectorizer = st.session_state.vectorizer
    matrix = st.session_state.tfidf_matrix

    query_vector = vectorizer.transform(
        [question]
    )

    scores = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    best_indices = np.argsort(scores)[::-1][:top_k]

    retrieved_chunks = []
    source_pages = set()

    for idx in best_indices:

        # Ignore chunks with no keyword overlap
        if scores[idx] <= 0:
            continue

        retrieved_chunks.append(
            st.session_state.chunks[idx]
        )

        source_pages.add(
            st.session_state.chunk_pages[idx]
        )

    return (
        "\n\n".join(retrieved_chunks),
        sorted(source_pages)
    )


# =========================================
# 7. SIDEBAR
# =========================================

with st.sidebar:

    st.header("⚙️ Settings")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.subheader("📄 PDF Information")

    if st.session_state.pdf_id:

        st.success("PDF Loaded")

        st.write(
            "Total Chunks:",
            len(st.session_state.chunks)
        )

        st.write("Vector Database: TF-IDF")

        st.write("LLM: Groq Cloud")

    else:

        st.info("Upload a PDF to get started.")

    st.divider()

    st.caption(
        "Built with Streamlit, Scikit-learn and Groq"
    )


# =========================================
# 8. PDF UPLOAD
# =========================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    pdf_hash = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    if st.session_state.pdf_id != pdf_hash:

        with st.spinner("Processing PDF..."):

            try:

                (
                    chunks,
                    chunk_pages,
                    vectorizer,
                    tfidf_matrix
                ) = process_pdf(pdf_bytes)

                st.session_state.pdf_id = pdf_hash

                st.session_state.chunks = chunks

                st.session_state.chunk_pages = chunk_pages

                st.session_state.vectorizer = vectorizer

                st.session_state.tfidf_matrix = tfidf_matrix

                st.session_state.chat_history = []

                st.success(
                    f"PDF processed successfully! "
                    f"{len(chunks)} chunks created."
                )

            except Exception as e:

                st.session_state.pdf_id = None

                st.error(f"PDF Processing Error: {e}")


# =========================================
# 9. CHATBOT
# =========================================

if st.session_state.pdf_id is not None:

    st.divider()

    st.subheader("💬 Chat with your PDF")

    # Display previous messages
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.write(message["content"])

            if message["role"] == "assistant":

                sources = message.get("sources", [])

                if sources:

                    st.caption(
                        "📚 Sources: " +
                        ", ".join(
                            f"PDF Page {p}"
                            for p in sources
                        )
                    )

    # Chat input
    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        try:

            context, source_pages = retrieve_context(
                question,
                top_k=3
            )

            if not context:

                answer = (
                    "I could not find relevant information "
                    "for this question in the PDF."
                )

            else:

                client = get_groq_client_safely()

                if client is None:
                    st.stop()

                system_prompt = f"""
You are an AI PDF assistant.

Answer using only the provided PDF context.

Use the previous conversation to understand
follow-up questions.

Do not invent information.

If the answer is not available in the PDF,
clearly say so.

PDF Context:
{context}
"""

                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ]

                messages.extend(
                    st.session_state.chat_history[-10:]
                )

                with st.chat_message("assistant"):

                    st.write("🤖 **AI Assistant**")

                    with st.spinner("Thinking..."):

                        stream = client.chat.completions.create(
                            model=os.getenv(
                                "GROQ_MODEL",
                                "openai/gpt-oss-20b"
                            ),
                            messages=messages,
                            temperature=0.3,
                            max_completion_tokens=800,
                            stream=True
                        )

                        def generate_response():

                            for chunk in stream:

                                content = (
                                    chunk.choices[0].delta.content
                                )

                                if content:
                                    yield content

                        answer = st.write_stream(
                            generate_response()
                        )

            # Display sources
            with st.chat_message("assistant"):

                if source_pages:

                    st.caption(
                        "📚 Sources: " +
                        ", ".join(
                            f"PDF Page {p}"
                            for p in source_pages
                        )
                    )

            # Save assistant response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": source_pages
            })

        except Exception as e:

            st.error(f"Something went wrong: {e}")

            if (
                st.session_state.chat_history
                and st.session_state.chat_history[-1]["role"] == "user"
            ):
                st.session_state.chat_history.pop()

else:

    st.info("📄 Please upload a PDF to start chatting.")
