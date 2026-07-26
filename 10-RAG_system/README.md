# 📚 PDF RAG System

## Overview

This project builds a simple Retrieval-Augmented Generation (RAG) system.

It reads a PDF file, converts the text into embeddings, and stores those embeddings in a FAISS vector database. When the user asks a question, the system converts the question into an embedding, searches for the most similar text chunks in FAISS, and returns the most relevant results from the PDF.

---

## Features

- Load a PDF file
- Extract text from every page
- Split the text into chunks
- Convert each chunk into an embedding
- Store embeddings in a FAISS index
- Search the PDF using natural language questions

---

## Project Pipeline

```
PDF
 │
 ▼
Extract text using PdfReader
 │
 ▼
Split text into chunks
 │
 ▼
Convert chunks into embeddings
 │
 ▼
Store embeddings in FAISS
 │
 ▼
User asks a question
 │
 ▼
Convert question into an embedding
 │
 ▼
FAISS finds the nearest chunks
 │
 ▼
Return the most relevant text
```

---

## Technologies Used

- Python
- PyPDF
- Sentence Transformers
- NumPy
- FAISS

---

## How to Run

```bash
python rag_system.py
```

---

## Example

### PDF

Computer Science Notes

### User Question

```
What is Generative AI?
```

### Output

The system returns the most relevant text chunks from the PDF that answer the question.

---

## Limitations

- PDF quality affects the results.
  - Scanned PDFs may not extract text correctly.
  - OCR is not supported.

- Fixed chunk size.
  - A chunk may split a sentence in the middle.
  - Sentence-based chunking would improve retrieval.

- No overlap between chunks.
  - Some context may be lost at chunk boundaries.
  - Production RAG systems usually use overlapping chunks.

---

## Future Improvements

- Add OCR support for scanned PDFs.
- Use sentence-based chunking.
- Add overlapping chunks.
- Return similarity scores.
- Connect the retriever to a Large Language Model (LLM) to generate complete answers.

---

## Author

Muhammad Yahya