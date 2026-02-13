import streamlit as st
import time
from src.document_loader import load_all_pdfs
from src.chunking import split_documents
from src.llm import get_embeddings, get_llm
from src.faiss_retriever import hybrid_search
from src.prompts import prompt_template
from src.reranker import rerank_hits

st.set_page_config(page_title="Cross-Border Compliance AI",layout="wide")

#display chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Kenya Legal AI Assistant")
st.caption("Cross-border data transfer laws in Kenya")

@st.cache_resource
def load_embeddings():
    return get_embeddings()

@st.cache_resource
def load_llm():
    print("Loading LLM Model...")
    return get_llm()

# sidebar for indexing documents
with st.sidebar:
    st.header("Ingest Legal Documents")
    if st.button("Ingest PDFs"):
        with st.spinner("Loading and indexing documents..."):
            documents = load_all_pdfs(["rag_docs"])
            split_docs = split_documents(documents)
            embeddings = load_embeddings()
            index_document(split_docs, embedder=embeddings, index_name="legal_docs")
            st.success("Ingestion and Indexing Complete!")

# chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# user input
if prompt := st.chat_input("Ask about cross-border data transfer in Kenya"):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Legal analysis in progress...") as status:
            start_time = time.time()

            status.write("Searching legal documents...")
            embeddings = load_embeddings()

            raw_hits = hybrid_search(query_text=prompt, embedder=embeddings, top_k=20)

            hits = rerank_hits(prompt, raw_hits, top_k=10)

            if not hits:
                status.update(label="No relevant legal documents found.", state="error")
                st.stop()

            #context = "\n\n".join(hit["content"] for hit in hits)
            context = "\n\n".join(f"Source: {hit['source']}, Section: {hit['section']}\nContent: {hit['content']}" for hit in hits)


            full_prompt = prompt_template.format(context=context, query=prompt)

            status.write("Generating response using LLM...")
            llm = load_llm()
            status.update(label="Streaming response...", state="running")

            response_holder = st.empty()
            full_response = ""

            try:
                for chunk in llm.stream(full_prompt):
                    content = chunk.content if hasattr(chunk, "content") else chunk
                    full_response += content
                    response_holder.markdown(full_response + "▌")

                latency = time.time() - start_time
                answer = full_response + f"\n\n*Retrieved {len(hits)} documents in {latency:.2f} seconds.*"
                response_holder.markdown(answer)

                status.update(label="Completed", state="complete")
            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"An error occurred: {e}")

        with st.expander("View Legal sources"):
            for i, hit in enumerate(hits, start=1):
                st.markdown(f"**Source {i}:** {hit['source']} | **Section:** {hit['section']}")
                st.markdown(hit["content"][:500] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
                