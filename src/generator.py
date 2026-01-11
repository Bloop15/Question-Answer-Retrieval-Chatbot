# src/generator.py
import os
from dotenv import load_dotenv
from typing import List, Dict
import logging

import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        try:
            genai.api_key = GEMINI_API_KEY
        except Exception:
            logger.warning("Could not set API key via genai.configure or genai.api_key; calls may fail.")

CLASSIC_INSUFFICIENT_MSG = "I don't have enough information in the knowledge base to answer this."


class GeminiGenerator:
    """
    Wraps a Gemini model. It gets:
      - the user's question
      - retrieved context chunks from the KB
    and returns a short, clear, exam-friendly answer using ONLY that context.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", max_chars: int = 2200):
        self.max_chars = max_chars
        try:
            self.model = genai.GenerativeModel(model_name)
            self.model_name = model_name
        except Exception:
            self.model = None
            self.model_name = model_name

    def _build_prompt(self, question: str, context_chunks: List[Dict]) -> str:
        # Build a clean context block
        blocks = []
        for i, c in enumerate(context_chunks, start=1):
            header = f"[CONTEXT {i} — {c.get('subject','unknown')} | page {c.get('page','?')}]"
            blocks.append(header + "\n" + c.get("text", ""))

        combined_context = "\n\n".join(blocks).strip()

        prompt = f"""
You are a helpful tutor for college students.

You are given:
1. A student's question.
2. Several context passages retrieved from their textbook-based knowledge base.

Your rules:
- Use ONLY the information inside the context passages to answer.
- Do NOT invent new facts or use outside knowledge.
- Write a clear, concise explanation that a student can understand in an exam.
- Avoid repeating whole paragraphs from the context; summarize and explain.
- If the context does not contain enough information to answer, reply exactly:

{CLASSIC_INSUFFICIENT_MSG}

====================
STUDENT QUESTION:
{question}

====================
RETRIEVED CONTEXT PASSAGES:
{combined_context}
====================

Now answer the student's question using ONLY the information in the context above.
Keep the answer focused and avoid unnecessary extra details.
"""
        return prompt

    def generate(self, question: str, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return CLASSIC_INSUFFICIENT_MSG

        prompt = self._build_prompt(question, context_chunks)

        try:
            if self.model:
                response = self.model.generate_content(prompt)
                text = getattr(response, "text", None)
                if not text:
                    try:
                        text = response.candidates[0].content
                    except Exception:
                        text = str(response)
                text = text.strip()
            else:
                resp = genai.generate_text(model=self.model_name, prompt=prompt)
                text = getattr(resp, "text", "") or str(resp)
                text = text.strip()
        except Exception as e:
            logger.exception("Primary generation failed: %s", e)
            text = ""

        if not text:
            return CLASSIC_INSUFFICIENT_MSG

        if CLASSIC_INSUFFICIENT_MSG in text:
            return CLASSIC_INSUFFICIENT_MSG

        # basic length control
        return text[: self.max_chars]
