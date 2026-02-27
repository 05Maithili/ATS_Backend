import json
from rapidfuzz import process, fuzz

def load_taxonomy(path:str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _normalize_one(surface:str, taxonomy:list[str]):
    cand, score, _ = process.extractOne(surface, taxonomy, scorer=fuzz.WRatio)
    return cand, score/100.0

def extract_and_normalize_skills(jd_norm:dict, resume_norm:dict, taxonomy:list[str]):
    jd_tokens = list(set(jd_norm["text"].split()))
    rs_tokens = list(set(resume_norm["full_text"].split()))
    jd_sk, rs_sk = set(), set()
    for t in jd_tokens:
        cand, conf = _normalize_one(t, taxonomy)
        if conf >= 0.9: jd_sk.add(cand)
    for t in rs_tokens:
        cand, conf = _normalize_one(t, taxonomy)
        if conf >= 0.9: rs_sk.add(cand)
    return jd_sk, rs_sk
