import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import tempfile

@st.cache_resource #runs the function once and caches the result
def load_models():
    embedder=SentenceTransformer("all-MiniLM-L6-v2")
    qa_model=pipeline("question-answering",model="deepset/roberta-base-squad2")
    return embedder,qa_model
embedder,qa_model=load_models()


st.title("LLM-Powered knowledge assistant for enterprises")
st.write("Upload a company PDF and ask questions about it.")

uploaded_file=st.file_uploader("Upload Company Policy PDF",type="pdf")


if uploaded_file:
    #save pdf temporarily
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path=tmp_file.name
    
    #Extract & clean text
    reader=PdfReader(pdf_path)
    raw_text="".join([page.extract_text() for page in reader.pages])
    clean_lines=[line.strip()for line in raw_text.splitlines() if line.strip()]
    text="\n".join(clean_lines)

    def split_text(text,chunk_size=500,overlap=50):
        chunks=[]
        for i in range(0,len(text),chunk_size-overlap):
            chunk=text[i:i+chunk_size]
            if len(chunk.strip())>100:
                chunks.append(chunk)
        return chunks
    chunks=split_text(text)

    chunk_embeddings=embedder.encode(chunks)
    dimension=chunk_embeddings.shape[1]
    index=faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_embeddings))

    question=st.text_input("💬 Ask a question about the document:")
    if question:
        question_embedding=embedder.encode([question])
        D,I=index.search(np.array(question_embedding),k=5)
        top_chunks=[chunks[i] for i in I[0]]
        context="\n\n".join(top_chunks)

        response=qa_model(question=question,context=context)
        st.success(f"🧠 **Answer:** {response['answer']}")

        with st.expander("📚 Show retrieved document chunks"):
            for i,chunk in enumerate(top_chunks):
                st.markdown(f"**Chunk {i+1}:**\n```\n{chunk}\n```")