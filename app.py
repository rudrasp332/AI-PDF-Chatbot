
import os
from pathlib import Path

import streamlit as st
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# 1. CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📚",
    layout="wide"
)

# Load .env from the same folder as app.py
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


# -----------------------------------
# 2. SESSION STATE
# -----------------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "chunk_pages" not in st.session_state:
    st.session_state.chunk_pages = []

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# -----------------------------------
# 3. PDF TEXT EXTRACTION
# -----------------------------------

def extract_pdf_text(pdf_file):

    reader = PdfReader(pdf_file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text and text.strip():
            pages.append((page_number, text))

    return pages


# -----------------------------------
# 4. TEXT CHUNKING
# -----------------------------------

def create_chunks(pages, chunk_size=500, overlap=50):

    chunks = []
    chunk_pages = []

    for page_number, text in pages:

        words = text.split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk = " ".join(words[start:end])

            if chunk.strip():
                chunks.append(chunk)
                chunk_pages.append(page_number)

            if end >= len(words):
                break

            start = end - overlap

    return chunks, chunk_pages


# -----------------------------------
# 5. TF-IDF VECTOR DATABASE
# -----------------------------------

def create_vector_database(chunks):

    vectorizer = TfidfVectorizer(
        max_features=20000,
        stop_words="english",
        dtype=np.float32
    )

    tfidf_matrix = vectorizer.fit_transform(chunks)

    return vectorizer, tfidf_matrix


# -----------------------------------
# 6. RETRIEVE RELEVANT CHUNKS
# -----------------------------------

def retrieve_context(question, top_k=4):

    vectorizer = st.session_state.vectorizer
    tfidf_matrix = st.session_state.tfidf_matrix

    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        question_vector,
        tfidf_matrix
    ).flatten()

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in top_indices:

        if similarities[index] > 0:

            results.append({
                "text": st.session_state.chunks[index],
                "page": st.session_state.chunk_pages[index],
                "score": float(similarities[index])
            })

    return results


# -----------------------------------
# 7. GROQ CLIENT
# -----------------------------------

def get_groq_client():

    if not GROQ_API_KEY:
        return None

    return Groq(api_key=GROQ_API_KEY)


# -----------------------------------
# 8. GENERATE ANSWER
# -----------------------------------

def generate_answer(question, context, chat_history):

    client = get_groq_client()

    if client is None:
        raise ValueError(
            "Groq API key not found. Check your .env file."
        )

    system_prompt = """
    You are an AI PDF assistant.

    Answer the user's questions using the provided PDF context.

    Rules:
    1. Give simple and clear answers.
    2. Use only the information available in the PDF context.
    3. If the answer is not available in the PDF, clearly say:
       "I could not find this information in the uploaded PDF."
    4. Do not invent information.
    5. Explain technical topics in easy words when possible.
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Include recent conversation history
    for message in chat_history[-6:]:

        messages.append({
            "role": message["role"],
            "content": message["content"]
        })

    messages.append({
        "role": "user",
        "content": f"""
        PDF CONTEXT:

        {context}

        QUESTION:

        {question}

        Answer using the PDF context.
        """
    })

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_completion_tokens=800,
        stream=True
    )

    for chunk in response:

        content = chunk.choices[0].delta.content

        if content:
            yield content


# -----------------------------------
# 9. APPLICATION UI
# -----------------------------------

st.title("📚 AI PDF Chatbot")

st.write(
    "Upload a PDF and ask questions about its content."
)

with st.sidebar:

    st.header("Upload Your PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    process_button = st.button(
        "Process PDF",
        use_container_width=True
    )

    if st.button("Clear Chat", use_container_width=True):

        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.caption("Powered by Groq Cloud AI")


# -----------------------------------
# 10. PROCESS PDF
# -----------------------------------

if process_button:

    if uploaded_file is None:

        st.warning("Please upload a PDF first.")

    else:

        with st.spinner("Reading and processing your PDF..."):

            try:

                pages = extract_pdf_text(uploaded_file)

                if not pages:
                    st.error(
                        "No readable text found. "
                        "This PDF may contain scanned images."
                    )

                else:

                    chunks, chunk_pages = create_chunks(pages)

                    if not chunks:
                        st.error("Could not create text chunks.")

                    else:

                        vectorizer, tfidf_matrix = (
                            create_vector_database(chunks)
                        )

                        st.session_state.chunks = chunks
                        st.session_state.chunk_pages = chunk_pages
                        st.session_state.vectorizer = vectorizer
                        st.session_state.tfidf_matrix = tfidf_matrix
                        st.session_state.pdf_name = uploaded_file.name
                        st.session_state.chat_history = []

                        st.success(
                            f"PDF processed successfully! "
                            f"{len(chunks)} text chunks created."
                        )

            except Exception as e:

                st.error(f"PDF processing error: {str(e)}")


# -----------------------------------
# 11. DISPLAY PDF STATUS
# -----------------------------------

if st.session_state.pdf_name:

    st.success(
        f"Current PDF: {st.session_state.pdf_name}"
    )


# -----------------------------------
# 12. DISPLAY CHAT HISTORY
# -----------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):

            with st.expander("View PDF Sources"):

                for source in message["sources"]:

                    st.write(
                        f"Page {source['page']}"
                    )


# -----------------------------------
# 13. USER QUESTION
# -----------------------------------

question = st.chat_input(
    "Ask a question about your PDF..."
)

if question:

    if not st.session_state.chunks:

        st.warning("Please upload and process a PDF first.")

    elif not GROQ_API_KEY:

        st.error(
            "Groq API key not found. "
            "Please check your .env file."
        )

    else:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            try:

                relevant_chunks = retrieve_context(question)

                if not relevant_chunks:

                    answer = (
                        "I could not find relevant information "
                        "in the uploaded PDF."
                    )

                    st.markdown(answer)

                else:

                    context = "\n\n".join([
                        f"Page {item['page']}:\n{item['text']}"
                        for item in relevant_chunks
                    ])

                    # Exclude the current question from prior history
                    previous_history = (
                        st.session_state.chat_history[:-1]
                    )

                    answer = st.write_stream(
                        generate_answer(
                            question,
                            context,
                            previous_history
                        )
                    )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": relevant_chunks
                    if relevant_chunks else []
                })

            except Exception as e:

                st.error(f"Error: {str(e)}")

                # Remove unanswered user message so it can be retried
                st.session_state.chat_history.pop()

