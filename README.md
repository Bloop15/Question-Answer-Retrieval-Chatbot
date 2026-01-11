# NexusAI – Textbook Question Answering System

NexusAI is a simple AI-powered system that answers questions from textbooks using a retrieval-based approach.  
It does not guess or hallucinate answers — it only uses the content present in the provided study material.

The system also supports image input, where a photo of a question can be uploaded and converted into text before answering.

---

## How it works

1. Textbooks or notes are stored as `.txt` files inside the `data/` folder.
2. The content is split into small chunks.
3. A TF-IDF vectorizer is used to find the most relevant chunks for a given question.
4. These chunks are sent to the Gemini model.
5. Gemini generates an answer using only the retrieved context.

---

## Features

- Ask questions from your textbook notes  
- Retrieves the most relevant content using TF-IDF  
- Gemini AI generates exam-friendly answers  
- Upload an image of a question and extract text using Gemini Vision  
- Debug mode to view which text chunks were used  

---

## Notes

- The system only answers based on the `.txt` files inside the `data` folder.
- If the answer is not found in the data, it will say:
  
  **"I don't have enough information in the knowledge base to answer this."**

---

## Purpose

This project was built to learn:
- Information retrieval (TF-IDF + cosine similarity)
- Prompt-controlled LLM generation
- Image-based question input
- End-to-end AI application development using Streamlit

