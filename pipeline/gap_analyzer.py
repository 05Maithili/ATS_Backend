def rank_missing_by_impact(feat_state, jd_skills:set, res_skills:set, jd_kp:list[str], res_kp:list[str], cfg)->list[dict]:
    missing_skills = list(set(jd_skills) - set(res_skills))
    missing_kp = [k for k in jd_kp if k not in set(res_kp)]
    ranked = []
    for s in missing_skills:
        ranked.append({"term": s, "estimated_gain": 1.5})
    for k in missing_kp:
        ranked.append({"term": k, "estimated_gain": 1.0})
    ranked.sort(key=lambda x: -x["estimated_gain"])
    return ranked
