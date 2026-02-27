import sys, os
import argparse, json, copy, datetime
# Ensure relative imports work regardless of run location
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from pipeline.utils import read_config, ensure_dir, write_log
from pipeline.ingest import load_resume, load_jd
from pipeline.normalize import normalize_resume, normalize_jd
from pipeline.embeddings import pairwise_match, build_embeddings
from pipeline.skills_taxonomy import load_taxonomy, extract_and_normalize_skills
from pipeline.keyphrases import extract_keyphrases_hybrid
from pipeline.scoring import compute_features, combined_score
from pipeline.gap_analyzer import rank_missing_by_impact
from pipeline.docx_writer import render_docx_from_state
from pipeline.rewrite_llm import propose_rewrites_guarded

OUTPUT_DIR = "data/output"


def analyze_once(cfg, resume_path, jd_path):
    """Run ATS analysis once."""
    resume = load_resume(resume_path)
    jd = load_jd(jd_path)
    resume_n = normalize_resume(resume)
    jd_n = normalize_jd(jd)

    embs = build_embeddings(cfg)
    matches = pairwise_match(jd_n["requirements"], resume_n["bullets"], cfg, embs)
    taxonomy = load_taxonomy(cfg["paths"]["skills_taxonomy"])
    jd_skills, res_skills = extract_and_normalize_skills(jd_n, resume_n, taxonomy)
    jd_kp = extract_keyphrases_hybrid(jd_n["text"], jd_n["requirements"])
    res_kp = extract_keyphrases_hybrid(resume_n["full_text"], resume_n["bullets"])

    feats = compute_features(jd_n, resume_n, matches, jd_skills, res_skills, jd_kp, res_kp)
    score = combined_score(feats, cfg)
    gaps = rank_missing_by_impact(feats, jd_skills, res_skills, jd_kp, res_kp, cfg)

    return {
        "config": cfg,
        "resume": resume_n,
        "jd": jd_n,
        "matches": matches,
        "features": feats,
        "score": score,
        "gaps": gaps,
        "rewrites": [],
        "meta": {
            "resume_path": resume_path,
            "jd_path": jd_path,
            "timestamp": datetime.datetime.now().isoformat()
        }
    }


def print_scores(state, header="ATS Analysis"):
    subs = state["features"]["subscores"]
    print(f"\n=== {header} ===")
    for k, v in subs.items():
        print(f"{k:9s}: {v}%")
    print("-" * 28)
    print(f"Final ATS: {state['score']}%")


def cmd_full_run(args):
    cfg = read_config(args.config)
    ensure_dir(cfg["paths"]["output_dir"])

    # STEP 1: Initial analysis
    state = analyze_once(cfg, args.resume, args.jd)
    print_scores(state, "Initial ATS Analysis")
    write_log("data/output/log.txt", f"Initial score: {state['score']}% | subs: {state['features']['subscores']}")

    # STEP 2: Ask for optimization
    if input("\nDo you want to optimize your resume to improve the score? (y/n): ").strip().lower() != "y":
        print("Okay, exiting without changes.")
        return

    # STEP 3: Show recommendations
    print("\n🔍 Recommended Additions (ranked by estimated impact):")
    for i, g in enumerate(state["gaps"][:20], 1):
        print(f"{i:2d}. {g['term']:<30} (+{g['estimated_gain']}%)")

    if input("\nAuto-rewrite relevant bullets using your LLM model? (y/n): ").strip().lower() != "y":
        print("No rewrites applied. Exiting after showing recommendations.")
        return

    # STEP 4: Run LLM rewrites
    proposals = propose_rewrites_guarded(state, max_items=args.max_rewrites)
    state["rewrites"] = proposals

    applied = 0
    for rw in proposals:
        if rw.get("status") == "ok" and rw.get("bullet"):
            ev = rw.get("evidence", "")
            try:
                idx = state["resume"]["bullets"].index(ev)
                state["resume"]["bullets"][idx] = rw["bullet"]
            except ValueError:
                state["resume"]["bullets"].append(rw["bullet"])
            applied += 1

    print(f"\nApplied rewrites: {applied}")

    # STEP 5: Recompute ATS on updated resume (without reloading from disk)
    embs = build_embeddings(cfg)
    matches = pairwise_match(state["jd"]["requirements"], state["resume"]["bullets"], cfg, embs)
    taxonomy = load_taxonomy(cfg["paths"]["skills_taxonomy"])
    jd_skills, res_skills = extract_and_normalize_skills(state["jd"], state["resume"], taxonomy)
    jd_kp = extract_keyphrases_hybrid(state["jd"]["text"], state["jd"]["requirements"])
    res_kp = extract_keyphrases_hybrid(state["resume"]["full_text"], state["resume"]["bullets"])
    feats = compute_features(state["jd"], state["resume"], matches, jd_skills, res_skills, jd_kp, res_kp, embs)
    state["features"] = feats
    state["score"] = combined_score(feats, cfg)

    print_scores(state, "Post-Update ATS Analysis")

    # STEP 6: Save outputs
    if input("\nDo you want to save the updated resume and report? (y/n): ").strip().lower() != "y":
        print("Discarded saving. Exiting.")
        return

    report = {
        "before_score": state["meta"].get("score_before", 0),
        "after_score": state["score"],
        "before_subscores": state["features"]["subscores"],
        "after_subscores": state["features"]["subscores"],
        "proposals": state["rewrites"],
        "gaps": state["gaps"],
        "timestamp": datetime.datetime.now().isoformat(),
        "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4-turbo")
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    render_docx_from_state(state, out_path=os.path.join(OUTPUT_DIR, "updated_resume.docx"))
    write_log("data/output/log.txt", f"Saved results. Before: {state.get('meta', {}).get('score_before', 0)} After: {state['score']}%")

    print("\n✅ Saved:")
    print(" - data/output/updated_resume.docx")
    print(" - data/output/report.json")
    print(" - data/output/log.txt")


def main():
    ap = argparse.ArgumentParser(description="AI ATS Optimizer — Interactive")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max-rewrites", type=int, default=8)
    sub = ap.add_subparsers(dest="cmd")

    fr = sub.add_parser("full-run", help="Analyze → Recommend → Rewrite → Rescore → Save (interactive)")
    fr.set_defaults(func=cmd_full_run)
    fr.add_argument("--resume", required=True, help="Path to resume (PDF/DOCX/TXT)")
    fr.add_argument("--jd", required=True, help="Path to job description (TXT/PDF/DOCX)")

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
