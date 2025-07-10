from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# 1. Load PDF
pdf_path = "data/employee_handbook.pdf"
reader = PdfReader(pdf_path)
raw_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

# 2. Clean and split
def split_text(text, chunk_size=800, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) > 100:  # avoid tiny chunks
            chunks.append(chunk.strip())
    return chunks

chunks = split_text(raw_text)

# 3. Embed chunks
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = model.encode(chunks)

# 4. Build FAISS index
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(chunk_embeddings))

# 5. Ask question
question = input("💬 Question: ").strip()

if question:
    question_embedding = model.encode([question])
    D, I = index.search(np.array(question_embedding), k=6)
    top_chunks = [chunks[i] for i in I[0]]
    
    # Show chunks used for debugging
    print("\n📚 Top Retrieved Chunks:")
    for idx, c in enumerate(top_chunks):
        print(f"\n--- Chunk {idx+1} ---\n{c}\n")

    context = "\n\n".join(top_chunks)

    # 6. Run QA
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    response = qa_model(question=question, context=context)
    print("🧠 Answer:", response["answer"])
else:
    print("❗ Please enter a valid question.")
