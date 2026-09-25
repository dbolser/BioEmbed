"""One-prompt Vertex smoke test through the vendored client (ADC in-process)."""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
for line in (HERE / ".env.vertex").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, _, v = line.partition("="); os.environ.setdefault(k.strip(), v.strip())
sys.path.insert(0, str(HERE.parent / "vendor" / "embedders-dilemma"))
from llm_judge.llm_client import get_vertex_credentials, settings
from openai import OpenAI
tok, proj = get_vertex_credentials()
print("ADC project:", proj, "| token:", "ok" if tok else "MISSING")
c = OpenAI(api_key=tok, base_url=settings.base_url)
r = c.chat.completions.create(model=settings.model, max_tokens=2000,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}])
print("reply:", (r.choices[0].message.content or "").strip()[:80], "| usage:", r.usage)
