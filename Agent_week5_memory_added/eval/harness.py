import os, time, math, json
from typing import List, Dict
from openai import OpenAI
from tools.retriever import query_topk
from agent import run_agent

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED = os.getenv("EMBED_MODEL","text-embedding-3-small")

def embed(txt: str):
    return client.embeddings.create(model=EMBED, input=txt or "").data[0].embedding

def cos(a,b):
    s = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return 0.0 if na*nb==0 else s/(na*nb)

def llm_judge(question: str, context: str, answer: str) -> bool:
    prompt = (f"Given the context, judge if the answer is supported by it.\n"
              f"Question: {question}\nContext:\n{context}\nAnswer:\n{answer}\n"
              "Respond with ONLY 'SUPPORTED' or 'UNSUPPORTED'.")
    m = client.chat.completions.create(model=os.getenv("MODEL","gpt-4.1-mini"),
                                       messages=[{"role":"user","content":prompt}])
    out = (m.choices[0].message.content or "").strip().upper()
    return out.startswith("SUPPORTED")

def run_suite(cases: List[Dict], k_for_eval=3, sim_threshold=0.75) -> Dict:
    rows = []
    for c in cases:
        q = c["q"]
        expect_src = c.get("expect_src")
        expect_ans = c.get("expect_ans")
        require_json = c.get("require_json", False)
        judge_grounded = c.get("judge_grounded", False)

        t0 = time.time()
        ans = run_agent(q)
        dt = time.time()-t0

        citation_ok = (expect_src is None) or (expect_src in ans)
        json_ok = True
        if require_json:
            try: json.loads(ans)
            except Exception: json_ok = False

        sim = None; sim_ok = True
        if expect_ans:
            sim = cos(embed(ans), embed(expect_ans))
            sim_ok = sim >= sim_threshold

        grounded = True
        if judge_grounded:
            ctx_chunks = query_topk(q, k=k_for_eval)
            ctx_text = "\n\n".join(x["text"] for x in ctx_chunks)
            grounded = llm_judge(q, ctx_text, ans)

        rows.append({
            "q": q, "latency_s": round(dt,2),
            "citation_ok": citation_ok, "json_ok": json_ok,
            "sim": round(sim,3) if sim is not None else None,
            "sim_ok": sim_ok, "grounded": grounded
        })

    n = len(rows)
    passed = sum(1 for r in rows if r["citation_ok"] and r["json_ok"] and r["grounded"] and r["sim_ok"])
    return {"summary":{"n":n,"pass_rate":round(passed/max(n,1),3)}, "rows": rows}
