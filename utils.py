import os
import re
from pathlib import Path
from typing import List, Dict, Any


import nltk

# Download the standard Punkt tokenizer
nltk.download("punkt")

# Also download punkt_tab (needed in NLTK >= 3.9)
nltk.download("punkt_tab")

nltk.download('wordnet')

from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download as nltk_download
from rake_nltk import Rake
from langdetect import detect, DetectorFactory
import pytesseract
from PyPDF2 import PdfReader
from docx import Document
from pdf2image import convert_from_path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ Setup ------------------
DetectorFactory.seed = 0

# Ensure NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk_download("punkt")

try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk_download("vader_lexicon")

# Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Optional spaCy
SPACY_AVAILABLE = False
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

# Tesseract Path (adjust if needed)
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ------------------ Core Functions ------------------

def extract_text(path: str, ocr: bool = False, ocr_pages: int = 3) -> str:
    """Extract text from PDF, DOCX, TXT, CSV. Fallback: OCR for PDFs."""
    ext = Path(path).suffix.lower()
    text = ""

    try:
        if ext == ".pdf":
            try:
                reader = PdfReader(path)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            except Exception:
                text = ""

            if ocr and len(text.strip()) < 50:
                try:
                    images = convert_from_path(path, first_page=1, last_page=ocr_pages)
                    for img in images:
                        text += pytesseract.image_to_string(img) + "\n"
                except Exception:
                    pass

        elif ext == ".docx":
            doc = Document(path)
            text = "\n".join([p.text for p in doc.paragraphs])

        elif ext in [".txt", ".md", ".log"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        elif ext == ".csv":
            df = pd.read_csv(path, encoding="utf-8", dtype=str, errors="ignore")
            text = df.to_string(index=False)

        else:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception:
                text = ""
    except Exception:
        text = ""

    return text.strip()


def summarize_text(text: str, max_sentences: int = 5) -> List[str]:
    sents = sent_tokenize(text)
    return sents[:max_sentences]


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:max_keywords]


def language_of(text: str) -> str:
    if not text or len(text.strip()) < 5:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


def word_count_and_freq(text: str, top_n: int = 30) -> Dict[str, Any]:
    words = re.findall(r"\b\w+\b", text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return {"count": len(words), "top": top}


def sentiment_scores(text: str) -> Dict[str, float]:
    if not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    return sia.polarity_scores(text)


def get_entities_spacy(text: str) -> List[Dict[str, str]]:
    if not SPACY_AVAILABLE:
        return []
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return []
    return [{"text": ent.text, "label": ent.label_} for ent in nlp(text).ents]


def find_matches_with_context(text: str, term: str, context_chars: int = 80) -> List[str]:
    matches = []
    if not term:
        return matches
    for m in re.finditer(re.escape(term), text, re.IGNORECASE):
        start = max(0, m.start() - context_chars)
        end = min(len(text), m.end() + context_chars)
        matches.append(text[start:end].replace("\n", " ").strip())
    return matches


def compute_similarity(doc_texts: List[str]) -> List[Dict[str, Any]]:
    if len(doc_texts) < 2:
        return []
    vectorizer = TfidfVectorizer(stop_words="english").fit_transform(doc_texts)
    sim_matrix = cosine_similarity(vectorizer)
    results = []
    n = len(doc_texts)
    for i in range(n):
        for j in range(i + 1, n):
            results.append({"i": i, "j": j, "score": float(sim_matrix[i, j])})
    return sorted(results, key=lambda x: x["score"], reverse=True)


def analyze_single(path: str, ocr: bool, ocr_pages: int, max_sum: int, max_kw: int) -> Dict[str, Any]:
    text = extract_text(path, ocr=ocr, ocr_pages=ocr_pages)
    lang = language_of(text)
    wc_info = word_count_and_freq(text, top_n=30)
    summary = summarize_text(text, max_sentences=max_sum)
    keywords = extract_keywords(text, max_keywords=max_kw)
    sentiment = sentiment_scores(text)
    entities = get_entities_spacy(text) if SPACY_AVAILABLE else []

    return {
        "file_name": os.path.basename(path),
        "file_type": Path(path).suffix,
        "size_bytes": os.path.getsize(path) if os.path.exists(path) else None,
        "language": lang,
        "word_count": wc_info["count"],
        "top_words": wc_info["top"],
        "sentence_count": len(sent_tokenize(text)) if text else 0,
        "reading_time_min": round(wc_info["count"] / 200.0, 2) if wc_info["count"] else 0,
        "keywords": keywords,
        "summary": summary,
        "preview": text[:1000],
        "sentiment": sentiment,
        "entities": entities,
        "full_text": text,
    }
