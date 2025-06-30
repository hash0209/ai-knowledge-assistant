import streamlit as st

from memory import get_memory
from query_chain import get_llm, create_qa_chain, ask_a_question
from retriever import load_vector_store

st.set_page_config(page_title = "AI Knowledge Assistant", layout="wide")


#load vector store
#load memory
#load qa_chain
#load llm

if "vectordb" not in st.session_state:
    vectordb=load_vector_store()
    st.session_state.vectordb = vectordb

if "memory" not in st.session_state:
    st.session_state.memory = get_memory()

if "llm" not in st.session_state:
    st.session_state.llm =get_llm()

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = create_qa_chain(
        st.session_state.llm,
        st.session_state.vectordb.as_retriever(),
        st.session_state.memory
    )

query=st.text_input("Ask a Question")

if "qa_log" not in st.session_state:
    st.session_state.qa_log = []

if st.button("Submit") and query:
    answer = ask_a_question(st.session_state.qa_chain,query)
    st.session_state.qa_log.append((query,answer))
    st.success(answer)

# Show conversation history
if st.session_state.qa_log:
    st.markdown("### 💬 Conversation History")
    for q, a in st.session_state.qa_log:
        st.markdown(f"**Q:** {q}\n\n**A:** {a}")

# Download Q&A log
if st.session_state.qa_log:
    log_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.qa_log])
    st.download_button("Download Q&A Log", log_text, file_name="qa_log.txt")
