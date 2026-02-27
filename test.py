import requests, os
from dotenv import load_dotenv
load_dotenv()

r = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    },
    json={
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Say hi"}]
    }
)

print(r.status_code)
print(r.text[:400])
