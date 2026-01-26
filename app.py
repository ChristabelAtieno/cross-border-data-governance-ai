import streamlit as st
import time
from src.ingestion import load_all_pdfs
from src.chunking import split_documents
from src.llm import get_embeddings, get_llm
from src.ingest import create_index, index_document
from src.retriever import hybrid_search

st.set_page_config(page_title="Kenya Legal AI",layout="wide")

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
        with st.spinner("Searching legal database"):
            start_time = time.time()
            embeddings = load_embeddings()
            hits = hybrid_search(
                query_text=prompt,
                embedder=embeddings,
                top_k=6)
            
            if hits:
                context = "\n\n.".join(hit["content"] for hit in hits)

                prompt_template = """
                You are a legal assistant specializing in cross-border data transfers.
                Use ONLY the information provided in the context to answer the questions.
                If the answer is not contained in the context, clearly state what is known and what is missing
                Context:
                {context}
                Question:
                {query}
                Answer:
                """

                llm = load_llm()

                full_prompt = prompt_template.format(context=context, query=prompt)
                response = llm.invoke(full_prompt)

                answer = response.content if hasattr(response, 'content') else response

                latency = time.time() - start_time
                answer += f"\n\n*Retrieved {len(hits)} documents in {latency:.2f} seconds.*"
                st.markdown(answer)

                # Display sources
                with st.expander("View Legal sources"):
                    for i, hit in enumerate(hits):
                        st.markdown(f"**Source {i}:** {hit['source']} | **Section:** {hit['section']}")
                        st.markdown(hit["content"][:500] + "...")
                
                st.session_state.messages.append({"role":"assistant","content":answer})
            else:
                st.error("No relevant legal documents found in the database.")
                st.session_state.messages.append({"role":"assistant","content":"No relevant legal documents found in the database."})
