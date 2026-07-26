from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def load_pdf(path):
    reader = PdfReader(path)

    text = ''
    for page in reader.pages:
        text+=page.extract_text()
    return text  

def chunk_split(text , chunk_size= 500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i+chunk_size]
        chunks.append(chunk)

    return chunks

def build_index(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, model, embeddings

def search(query, model, index, chunks,top_k=3):
    query_embeddings = model.encode([query])
    query_embeddings = np.array(query_embeddings).astype("float32")

    distance , indices = index.search(query_embeddings , top_k)

    results= []
    for i in indices[0]:
        results.append(chunks[i])

    return results    

if __name__ == "__main__":

    text= load_pdf(r"D:\BOOKS\CS102 LAB B2\CS LAB MANUAL\CS101L - Lab 01.pdf")
    chunks = chunk_split(text)
    index, model, embeddings = build_index(chunks)

    query = input("Serach Engine..")

    results = search(query, model, index, chunks,top_k=3)

    for result in results:
        print("\n..............")
        print(result)
