from typing import List, Dict, Any, Set
import numpy as np

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def coverage(jd_terms: Set[str], res_terms: Set[str]) -> float:
    if not jd_terms:
        return 0.0
    return len(jd_terms & res_terms) / len(jd_terms)

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _semantic_from_matches(matches: List[Dict[str, Any]], topk: int = 10) -> float:
    """
    Use cross-encoder scores if present; else cosine from matches.
    Returns percentage [0..100].
    """
    if not matches:
        return 0.0
    scores = []
    for m in matches:
        # prefer cross-encoder if provided; else fallback to cosine
        val = float(m.get("cross", m.get("cos", 0.0)))
        scores.append(val)
    if not scores:
        return 0.0
    # Normalize heuristically to [0..1] if scores look like logits
    # If scores already in [0..1], this keeps them; if raw ~[-10,10], squish via sigmoid-ish scale
    arr = np.array(scores, dtype=float)
    # If any score > 1, assume raw logits and map to (0,1) via 1/(1+exp(-x))
    if np.any(arr > 1.0) or np.any(arr < 0.0):
        arr = 1.0 / (1.0 + np.exp(-arr))
    # average top-k
    k = min(topk, len(arr))
    sem = float(np.mean(np.sort(arr)[-k:]))
    return _clamp01(sem) * 100.0

def _semantic_from_sbert(jd_reqs: List[str], res_bullets: List[str], embs: Dict[str, Any], topk: int = 5) -> float:
    """
    SBERT fallback: mean of top-k cosine similarities (per JD requirement),
    averaged across requirements. Returns percentage [0..100].
    """
    if not embs or "sbert" not in embs or embs["sbert"] is None:
        return 0.0
    if not jd_reqs or not res_bullets:
        return 0.0

    sbert = embs["sbert"]
    # Use normalized embeddings so dot == cosine
    jr = sbert.encode(jd_reqs, normalize_embeddings=True, convert_to_numpy=True)
    rb = sbert.encode(res_bullets, normalize_embeddings=True, convert_to_numpy=True)

    if jr.size == 0 or rb.size == 0:
        return 0.0

    sim = jr @ rb.T  # cosine similarities in [-1,1], but typically [0,1] for relevant text
    per_req = []
    for i in range(sim.shape[0]):
        row = sim[i]
        k = min(topk, row.shape[0])
        if k == 0:
            continue
        top = np.sort(row)[-k:]
        # clamp to [0,1] then average
        top = np.clip(top, 0.0, 1.0)
        per_req.append(float(np.mean(top)))
    if not per_req:
        return 0.0
    return _clamp01(float(np.mean(per_req))) * 100.0

def compute_features(
    jd_norm: Dict[str, Any],
    res_norm: Dict[str, Any],
    matches: List[Dict[str, Any]],
    jd_skills: Set[str],
    res_skills: Set[str],
    jd_kp: List[str],
    res_kp: List[str],
    embs: Dict[str, Any] = None,   # <-- NEW: pass embeddings for semantic fallback
) -> Dict[str, Any]:
    # Semantic alignment: prefer cross-encoder/cos scores from matches; else SBERT fallback
    sem = _semantic_from_matches(matches, topk=10)
    if sem == 0.0:  # fallback if matches empty or uninformative
        sem = _semantic_from_sbert(jd_norm.get("requirements", []), res_norm.get("bullets", []), embs, topk=5)

    feat = {
        "keyword_coverage": coverage(set(jd_kp), set(res_kp)) * 100.0,
        "skill_overlap": jaccard(set(jd_skills), set(res_skills)) * 100.0,
        "semantic_alignment": sem,
        "formatting": 85.0,
    }
    subs = {
        "Keyword": round(feat["keyword_coverage"], 2),
        "Skills": round(feat["skill_overlap"], 2),
        "Semantic": round(feat["semantic_alignment"], 2),
        "Format": round(feat["formatting"], 2),
    }
    return {"features": feat, "subscores": subs}

def combined_score(feat_state: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    w = cfg["scoring_weights"]
    f = feat_state["features"]
    score = (
        w["keyword_coverage"] * f["keyword_coverage"] +
        w["skill_overlap"]     * f["skill_overlap"] +
        w["semantic_alignment"]* f["semantic_alignment"] +
        w["formatting"]        * f["formatting"]
    )
    return round(score, 2)
