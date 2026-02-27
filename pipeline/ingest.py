import pdfplumber
from docx import Document
import os, re

def _read_pdf(path):
    text = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            text.append(p.extract_text() or "")
    return "\n".join(text)

def _read_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def load_resume(path:str)->dict:
    ext = os.path.splitext(path)[1].lower()
    if ext==".pdf":
        raw = _read_pdf(path)
    elif ext==".docx":
        raw = _read_docx(path)
    else:
        # Try multiple encodings for text files
        raw = None
        for encoding in ['utf-8', 'utf-8-sig', 'windows-1252', 'iso-8859-1', 'ascii']:
            try:
                raw = open(path, "r", encoding=encoding).read()
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if raw is None:
            # If all encodings fail, read as binary and decode with errors='ignore'
            with open(path, 'rb') as f:
                raw = f.read().decode('utf-8', errors='ignore')
    text = re.sub(r'\s+', ' ', raw).strip()
    bullets = [b.strip(" •-") for b in re.split(r'[•\u2022\-]\s+', raw) if len(b.strip())>0]
    return {"raw_text": raw, "full_text": text, "bullets": bullets}

def load_jd(path:str)->dict:
    ext = os.path.splitext(path)[1].lower()
    if ext==".pdf":
        raw = _read_pdf(path)
    elif ext==".docx":
        raw = _read_docx(path)
    else:
        # Try multiple encodings for text files
        raw = None
        for encoding in ['utf-8', 'utf-8-sig', 'windows-1252', 'iso-8859-1', 'ascii']:
            try:
                raw = open(path, "r", encoding=encoding).read()
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if raw is None:
            # If all encodings fail, read as binary and decode with errors='ignore'
            with open(path, 'rb') as f:
                raw = f.read().decode('utf-8', errors='ignore')
    text = re.sub(r'\s+', ' ', raw).strip()
    reqs = re.split(r'[\n;\.]+', raw)
    reqs = [r.strip(" -•\n\t") for r in reqs if len(r.strip())>0]
    return {"raw_text": raw, "text": text, "requirements": reqs, "title": ""}
