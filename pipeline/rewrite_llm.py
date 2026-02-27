# Compact rewrite version for limited tokens (≤600)
import os, json, re, requests
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER", "http://localhost")
OPENROUTER_APP     = os.getenv("OPENROUTER_APP_NAME", "ai-ats-optimizer")

def _validate_bullet(text, target_kw, min_w=6, max_w=25):
    if not text: return False
    if not (min_w <= len(text.split()) <= max_w): return False
    if re.search(r'\b(I|my|me)\b', text, flags=re.I): return False
    if target_kw.lower() not in text.lower(): return False
    return True

def _llm_rewrite(evidence_bullet, target_kw):
    if not OPENROUTER_API_KEY:
        return {"status": "not_applicable", "bullet": "", "reason": "Missing OPENROUTER_API_KEY"}

    sys_prompt = "You are a resume optimization AI. Rewrite the given bullet truthfully using the target keyword naturally. Keep it short (under 25 words) and action-oriented."
    user_prompt = f"Keyword: {target_kw}\nBullet: {evidence_bullet}\nReturn JSON: {{'status':'ok|skip','bullet':'rewritten sentence'}}"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": OPENROUTER_REFERER,
        "X-Title": OPENROUTER_APP,
        "Content-Type": "application/json"
    }
    body = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 500   # ✅ comfortably below your 630 limit
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=45)
        if resp.status_code == 402:
            return {"status": "not_applicable", "bullet": "", "reason": "OpenRouter: Not enough credits (402)"}
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        # attempt to parse JSON output from model
        obj = None
        parse_error = None
        try:
            obj = json.loads(content)
        except Exception as e1:
            parse_error = e1
            # try to extract a JSON-like substring
            m = re.search(r"\{.*\}", content, flags=re.S)
            if m:
                txt = m.group(0)
                try:
                    obj = json.loads(txt)
                except Exception:
                    try:
                        obj = json.loads(txt.replace("'", '"'))
                    except Exception as e2:
                        parse_error = e2
                        obj = None
            else:
                # as a last resort, try replacing single quotes
                try:
                    obj = json.loads(content.replace("'", '"'))
                except Exception as e3:
                    parse_error = e3
                    obj = None

        if not obj:
            # return a clear skip reason rather than raising
            reason = f"Parse error: {parse_error}" if parse_error else "No JSON returned"
            return {"status": "skip", "bullet": "", "reason": reason}

        if obj.get("status") == "ok" and _validate_bullet(obj.get("bullet", ""), target_kw):
            print(f"✅ Rewrote bullet for '{target_kw}'")
            return {"status": "ok", "bullet": obj["bullet"], "reason": "llm_ok"}
        else:
            print(f"⚠️ Skipped '{target_kw}': {obj.get('reason', 'Invalid')}")
            return {"status": "skip", "bullet": "", "reason": obj.get("reason", "failed_validation")}
    except Exception as e:
        return {"status": "not_applicable", "bullet": "", "reason": str(e)}

def propose_rewrites_guarded(state, max_items=8):
    proposals = []
    bullets = state["resume"]["bullets"]
    if not bullets:
        return proposals
    print(f"\n🚀 Starting compact LLM rewrites (<=600 tokens per request)...")
    for gap in state["gaps"][:max_items]:
        kw = gap["term"]
        ev = bullets[0]
        res = _llm_rewrite(ev, kw)
        proposals.append({
            "target": kw,
            "evidence": ev,
            "bullet": res.get("bullet", ""),
            "status": res.get("status", "skip"),
            "reason": res.get("reason", "")
        })
    return proposals
