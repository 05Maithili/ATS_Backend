import spacy, re
nlp = spacy.load("en_core_web_sm")

def _clean(s:str)->str:
    s = re.sub(r'[^\x00-\x7F]+',' ',s)
    s = re.sub(r'\s+',' ',s).strip()
    return s

def normalize_resume(resume:dict)->dict:
    doc = nlp(resume["full_text"])
    toks = [t.lemma_.lower() for t in doc if t.is_alpha]
    bullets = []
    for b in resume["bullets"]:
        b2 = _clean(b)
        if b2 and len(b2.split())>=3:
            bullets.append(b2)
    return {"full_text":" ".join(toks), "bullets": bullets}

def normalize_jd(jd:dict)->dict:
    doc = nlp(jd["text"])
    toks = [t.lemma_.lower() for t in doc if t.is_alpha]
    reqs = []
    for r in jd["requirements"]:
        r2 = _clean(r)
        if r2 and len(r2.split())>=3:
            reqs.append(r2)
    return {"text":" ".join(toks), "requirements": reqs, "title": jd.get("title","")}
