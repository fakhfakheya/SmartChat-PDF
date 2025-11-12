import os
from pdf2image import convert_from_path
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\wÀ-ÿ,.:;!?()\- ]', '', text)
    return text.strip()

def pdf_to_text(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Le fichier {pdf_path} n'existe pas")
    pages = convert_from_path(pdf_path)
    full_text = ""
    for page in pages:
        text = pytesseract.image_to_string(page, lang='fra')
        text = clean_text(text)
        full_text += text + "\n"
    return full_text

def split_documents(text: str, chunk_size: int = 300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks
