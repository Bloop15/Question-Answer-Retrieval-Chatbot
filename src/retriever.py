import os
import re
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u200b", "")
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u0080-\uFFFF]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class Retriever:
    def __init__(
        self,
        data_dir: str = DATA_DIR,
        chunk_size: int = 900,
        chunk_overlap: int = 100,
    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.chunks: List[Dict] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        self._load_and_index()

    def _list_txt_files(self) -> List[str]:
        """Recursively list all .txt files under data_dir."""
        txt_files = []
        if not os.path.exists(self.data_dir):
            return []
        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                if f.lower().endswith(".txt"):
                    txt_files.append(os.path.join(root, f))
        txt_files.sort()
        return txt_files

    def _read_txt(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            return clean_text(text)
        except Exception:
            try:
                with open(path, "r", encoding="latin-1") as f:
                    text = f.read()
                return clean_text(text)
            except Exception:
                return ""

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs using blank lines; fallback to sentence-based packing."""
        if not text:
            return []
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

        # If document doesn't have blank-lines, split by sentences and pack into chunk_size
        if len(paragraphs) <= 1:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            paragraphs = []
            acc = ""
            for s in sentences:
                if len(acc) + len(s) + 1 <= self.chunk_size:
                    acc = (acc + " " + s).strip()
                else:
                    if acc:
                        paragraphs.append(acc)
                    acc = s
            if acc:
                paragraphs.append(acc)
        return paragraphs

    def _split_long_paragraph(self, p: str) -> List[str]:
        """Split long paragraph using sliding-window with overlap."""
        if len(p) <= self.chunk_size:
            return [p]
        chunks = []
        start = 0
        L = len(p)
        while start < L:
            end = start + self.chunk_size
            chunk = p[start:end].strip()
            chunks.append(chunk)
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
            if start >= L - 1:
                break
        return chunks

    def _chunk_file(self, text: str) -> List[str]:
        paragraphs = self._split_paragraphs(text)
        final_chunks = []

        for p in paragraphs:
            if len(p) <= self.chunk_size:
                final_chunks.append(p)
            else:
                parts = self._split_long_paragraph(p)
                final_chunks.extend(parts)

        return final_chunks

    def _load_and_index(self):
        """Load all txt files, chunk them, and build TF-IDF index."""
        self.chunks = []
        txt_paths = self._list_txt_files()

        for path in txt_paths:
            rel = os.path.relpath(path, self.data_dir)
            folder = os.path.dirname(rel)
            filename = os.path.splitext(os.path.basename(path))[0]
            subject = f"{folder}/{filename}" if folder and folder != "." else filename

            full_text = self._read_txt(path)
            paragraphs = self._split_paragraphs(full_text)

            for page_num, para in enumerate(paragraphs, start=1):
                para_chunks = self._chunk_file(para)

                for cid, ctext in enumerate(para_chunks, start=1):
                    self.chunks.append(
                        {
                            "subject": subject,
                            "page": page_num,
                            "chunk_id": cid,
                            "text": ctext,
                        }
                    )

        texts = [c["text"] for c in self.chunks]

        if texts:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        else:
            self.vectorizer = None
            self.tfidf_matrix = None

    def reload(self):
        """Public: reload and rebuild the index."""
        self._load_and_index()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Return top_k chunks ranked by cosine similarity to query."""
        if not query or self.tfidf_matrix is None or self.vectorizer is None:
            return []

        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix)[0]

        idx_sorted = np.argsort(scores)[::-1]
        top_idx = idx_sorted[: max(1, top_k)]

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            ch = self.chunks[idx]
            results.append(
                {
                    "subject": ch["subject"],
                    "page": ch["page"],
                    "chunk_id": ch["chunk_id"],
                    "text": ch["text"],
                    "score": float(scores[idx]),
                    "rank": rank,
                }
            )
        return results

    def get_subjects(self) -> List[str]:
        return sorted({c["subject"] for c in self.chunks})

    def get_chunk_count(self) -> int:
        return len(self.chunks)
