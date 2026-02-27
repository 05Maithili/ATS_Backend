# ats_project_yuvraj — AI ATS Optimizer (Interactive, OpenRouter GPT‑4 Turbo)

A complete, **interactive** backend project that:
1) Analyzes a resume vs job description and shows ATS sub‑scores,
2) Asks whether to optimize,
3) Shows ranked recommendations (missing skills/keywords),
4) Uses **OpenRouter (OpenAI: GPT‑4 Turbo)** to rewrite bullets truthfully,
5) Updates the resume, re‑scores, and asks whether to save the improved DOCX + report.

**Date:** 2025-11-10

## Windows Quick Start
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
copy .env.example .env
notepad .env  # paste your OpenRouter key
python cli.py full-run --resume "data\samples\sample_resume.docx" --jd "data\samples\sample_jd.txt"
```
