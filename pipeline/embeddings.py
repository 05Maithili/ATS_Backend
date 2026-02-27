from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

_sbert = None
_xenc = None

def build_embeddings(cfg:dict):
    global _sbert, _xenc
    if _sbert is None:
        _sbert = SentenceTransformer(cfg["embeddings"]["sbert_model"])
    if cfg["embeddings"].get("use_cross_encoder", True) and _xenc is None:
        _xenc = CrossEncoder(cfg["embeddings"]["cross_encoder_model"])
    return {"sbert": _sbert, "xenc": _xenc}

def pairwise_match(jd_reqs, resume_bullets, cfg, embs):
    if not jd_reqs or not resume_bullets:
        return []
    sbert = embs["sbert"]
    xenc = embs["xenc"]
    jr = sbert.encode(jd_reqs, normalize_embeddings=True, convert_to_numpy=True)
    rb = sbert.encode(resume_bullets, normalize_embeddings=True, convert_to_numpy=True)
    sim = jr @ rb.T
    matches = []
    k = int(cfg["embeddings"]["topk_per_requirement"])
    for i, row in enumerate(sim):
        idx = np.argsort(-row)[:k]
        pairs = [(jd_reqs[i], resume_bullets[j], float(row[j])) for j in idx]
        ce_scores = None
        if xenc is not None and pairs:
            ce_scores = xenc.predict([(p[0], p[1]) for p in pairs])
        for j, (jr_text, rb_text, cos) in enumerate(pairs):
            ce = float(ce_scores[j]) if ce_scores is not None else cos
            matches.append({"jd_req": jr_text, "bullet": rb_text, "cos": float(cos), "cross": ce})
    return matches
