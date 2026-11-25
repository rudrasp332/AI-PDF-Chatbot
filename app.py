
import streamlit as st
import ollama
import faiss
import numpy as np

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# =====================================
# 1. STREAMLIT CONFIGURATION
# =====================================

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI PDF Chatbot")
st.caption("Upload your PDF and chat with its content.")


# =====================================
# 2. SESSION STATE
# =====================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_id" not in st.session_state:
    st.session_state.pdf_id = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "chunk_pages" not in st.session_state:
    st.session_state.chunk_pages = []

if "index" not in st.session_state:
    st.session_state.index = None


# =====================================
# 3. LOAD EMBEDDING MODEL
# =====================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# =====================================
# 4. SIDEBAR
# =====================================

with st.sidebar:

    st.header("⚙️ Settings")

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.subheader("📄 PDF Information")

    if st.session_state.index is not None:

        st.success("PDF Loaded")

        st.write(
            "Total Chunks:",
            len(st.session_state.chunks)
        )

        st.write("Vector Database: FAISS")

        st.write("LLM: Mimo (Ollama)")

    else:

        st.info("Upload a PDF to get started.")

    st.divider()

    st.caption("Built with Streamlit, FAISS and Ollama")


# =====================================
# 5. PDF UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# =====================================
# 6. PDF PROCESSING
# =====================================

if uploaded_file is not None:

    # Identify uploaded PDF
    pdf_id = (
        uploaded_file.name,
        uploaded_file.size,
        uploaded_file.getvalue()
    )

    if st.session_state.pdf_id != pdf_id:

        st.session_state.pdf_id = pdf_id

        st.session_state.chat_history = []
        st.session_state.chunks = []
        st.session_state.chunk_pages = []
        st.session_state.index = None

        # ---------------------------------
        # Extract PDF text page by page
        # ---------------------------------

        try:

            reader = PdfReader(uploaded_file)

            pages_text = []

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                text = page.extract_text()

                if text and text.strip():

                    pages_text.append({
                        "page": page_number,
                        "text": text
                    })

            if not pages_text:

                st.warning(
                    "No readable text found in this PDF."
                )

            else:

                # ---------------------------------
                # 7. TEXT SPLITTING
                # ---------------------------------

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )

                chunks = []
                chunk_pages = []

                for page_data in pages_text:

                    page_chunks = splitter.split_text(
                        page_data["text"]
                    )

                    for chunk in page_chunks:

                        chunks.append(chunk)

                        chunk_pages.append(
                            page_data["page"]
                        )

                st.session_state.chunks = chunks

                st.session_state.chunk_pages = chunk_pages

                # ---------------------------------
                # 8. GENERATE EMBEDDINGS
                # ---------------------------------

                with st.spinner("Creating embeddings..."):

                    model = load_embedding_model()

                    embeddings = model.encode(chunks)

                    embeddings = np.array(
                        embeddings,
                        dtype="float32"
                    )

                # ---------------------------------
                # 9. STORE IN FAISS
                # ---------------------------------

                faiss.normalize_L2(embeddings)

                dimension = embeddings.shape[1]

                index = faiss.IndexFlatIP(dimension)

                index.add(embeddings)

                st.session_state.index = index

                st.success("PDF processed successfully!")

        except Exception as e:

            st.error(f"PDF Processing Error: {e}")


# =====================================
# 10. CHATBOT
# =====================================

if st.session_state.index is not None:

    st.divider()

    st.subheader("💬 Chat with your PDF")

    # ---------------------------------
    # Display Previous Messages
    # ---------------------------------

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.write(message["content"])

            if message["role"] == "assistant":

                sources = message.get("sources", [])

                if sources:

                    st.caption("📚 Sources")

                    st.write(
                        ", ".join(
                            f"PDF Page {page}"
                            for page in sources
                        )
                    )

    # ---------------------------------
    # Chat Input
    # ---------------------------------

    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        # Save user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        try:

            # ---------------------------------
            # 11. QUESTION EMBEDDING
            # ---------------------------------

            model = load_embedding_model()

            query_embedding = model.encode([question])

            query_embedding = np.array(
                query_embedding,
                dtype="float32"
            )

            faiss.normalize_L2(query_embedding)

            # ---------------------------------
            # 12. FAISS SEARCH
            # ---------------------------------

            scores, indices = st.session_state.index.search(
                query_embedding,
                k=min(3, st.session_state.index.ntotal)
            )

            retrieved_chunks = []

            source_pages = set()

            for idx in indices[0]:

                if idx != -1:

                    retrieved_chunks.append(
                        st.session_state.chunks[idx]
                    )

                    source_pages.add(
                        st.session_state.chunk_pages[idx]
                    )

            context = "\n\n".join(retrieved_chunks)

            # ---------------------------------
            # 13. CREATE OLLAMA MESSAGES
            # ---------------------------------

            messages = [
                {
                    "role": "system",
                    "content": f"""
                    You are an AI PDF assistant.

                    Answer using only the PDF context.

                    Use previous conversation to understand
                    follow-up questions.

                    Do not invent information.

                    If the answer is not available in the
                    PDF context, say so.

                    PDF Context:
                    {context}
                    """
                }
            ]

            # Include recent conversation
            messages.extend(
                st.session_state.chat_history[-10:]
            )

            # ---------------------------------
            # 14. STREAMING RESPONSE
            # ---------------------------------

            with st.chat_message("assistant"):

                st.write("🤖 **AI Assistant**")

                with st.spinner("Thinking..."):

                    stream = ollama.chat(
                        model="mimo-v2.6:latest",
                        messages=messages,
                        stream=True
                    )

                    answer = st.write_stream(
                        chunk["message"]["content"]
                        for chunk in stream
                    )

                # ---------------------------------
                # 15. DISPLAY SOURCES
                # ---------------------------------

                st.caption("📚 Sources")

                if source_pages:

                    st.write(
                        ", ".join(
                            f"PDF Page {page}"
                            for page in sorted(source_pages)
                        )
                    )

                else:

                    st.write("No sources found.")

            # ---------------------------------
            # 16. SAVE CHAT HISTORY
            # ---------------------------------

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sorted(source_pages)
            })

        except Exception as e:

            st.error(f"Something went wrong: {e}")

            # Remove failed user question
            st.session_state.chat_history.pop()


else:

    st.info("📄 Please upload a PDF to start chatting.")
