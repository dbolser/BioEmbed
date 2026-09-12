"""Smoke test: tiny model on two small bio tasks, end to end."""
import mteb

model = mteb.get_model("sentence-transformers/all-MiniLM-L6-v2")
tasks = mteb.get_tasks(tasks=["BIOSSES", "NanoNFCorpusRetrieval"])
results = mteb.evaluate(model, tasks=tasks, cache=mteb.ResultCache("results/mteb"))
for r in results:
    print(r.task_name, r.get_score())
