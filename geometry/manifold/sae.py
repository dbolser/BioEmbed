"""A learned dictionary of ideas: a top-k sparse autoencoder on the centred paper vectors (Gao et al. 2024,
https://arxiv.org/abs/2406.04093; on text embeddings of abstracts: O'Neill et al. 2024, https://arxiv.org/abs/2408.00657).

  x = (paper vector - corpus mean) * s     (s: so the average x has length 1)
  z = TopK_k(ReLU(W_enc (x - b_dec) + b_enc)),  x_hat = W_dec z + b_dec,  decoder columns kept unit length
Training papers: at most --cap per topic (big topics do not dominate), 2% held out. Scores: fraction of variance
explained on held-out papers, dead features. Then every feature is named by the MeSH headings / publication
types most over-represented among papers where it fires (lift, on 1M papers), and compared with the concepts:
the MeSH/GO concept text (centred) nearest its decoder direction.
Writes data/geometry/medembed_small/manifold/sae/{sae.pt, features.csv, summary.json}
CPU, root env: uv run python geometry/manifold/sae.py [--latents 4096] [--k 32] [--epochs 8] [--threads 16]
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, torch

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/geometry/medembed_small"
M = D / "manifold"; OUT = M / "sae"; OUT.mkdir(exist_ok=True)
NAME = "eom_mcs200_ms3"
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


class SAE(torch.nn.Module):
    def __init__(self, d, n, k):
        super().__init__(); self.k = k
        W = torch.randn(n, d); W /= W.norm(dim=1, keepdim=True)
        self.W_enc = torch.nn.Parameter(W.clone()); self.b_enc = torch.nn.Parameter(torch.zeros(n))
        self.W_dec = torch.nn.Parameter(W.T.clone()); self.b_dec = torch.nn.Parameter(torch.zeros(d))

    def encode(self, x):
        pre = torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)
        v, i = pre.topk(self.k, dim=1)
        return torch.zeros_like(pre).scatter_(1, i, v)

    def forward(self, x):
        z = self.encode(x); return z @ self.W_dec.T + self.b_dec, z

    @torch.no_grad()
    def norm_dec(self): self.W_dec /= self.W_dec.norm(dim=0, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", type=int, default=4096); ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--cap", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=16); ap.add_argument("--batch", type=int, default=4096)
    a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(0)
    X = np.load(D / "X.npy", mmap_mode="r"); mu = np.load(M / "mean.npy")
    lab = np.load(D / f"clusters/{NAME}_all.npy")
    rng = np.random.default_rng(0)
    rows = np.sort(np.concatenate([rng.choice(np.flatnonzero(lab == g), min(a.cap, (lab == g).sum()), replace=False)
                                   for g in range(lab.max() + 1)]))
    Xt = np.asarray(X[rows], np.float32) - mu
    s = 1 / np.linalg.norm(Xt, axis=1).mean(); Xt *= s
    hold = rng.random(len(rows)) < 0.02
    tr, te = torch.from_numpy(Xt[~hold]), torch.from_numpy(Xt[hold])
    log(f"training papers {len(tr):,}, held out {len(te):,}, scale {s:.3f}")
    model = SAE(384, a.latents, a.k)
    with torch.no_grad(): model.b_dec.copy_(tr[:200_000].median(0).values)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    var_te = ((te - te.mean(0)) ** 2).sum()
    for ep in range(a.epochs):
        perm = torch.randperm(len(tr)); fired = torch.zeros(a.latents, dtype=torch.bool)
        if ep == a.epochs - 2:
            for g in opt.param_groups: g["lr"] = 3e-4
        for b in range(0, len(tr), a.batch):
            xb = tr[perm[b:b + a.batch]]
            xh, z = model(xb); loss = ((xh - xb) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); model.norm_dec()
            fired |= (z > 0).any(0)
        with torch.no_grad():
            xh, _ = model(te); fve = 1 - float(((xh - te) ** 2).sum() / var_te)
        log(f"epoch {ep + 1}: held-out variance explained {fve:.3f}; features that fired this epoch {fired.float().mean():.1%}")
    torch.save(dict(state=model.state_dict(), scale=float(s), k=a.k), OUT / "sae.pt")  # plain types: weights_only load

    # name the features: MeSH / publication-type lift among papers where each fires (1M random papers)
    n = len(X); ev = np.sort(rng.choice(n, 1_000_000, replace=False))
    z_rows, z_cols, z_vals = [], [], []
    with torch.no_grad():
        for b in range(0, len(ev), 100_000):
            xb = torch.from_numpy((np.asarray(X[ev[b:b + 100_000]], np.float32) - mu) * s)
            z = model.encode(xb); nz = z.nonzero()
            z_rows.append(nz[:, 0].numpy() + b); z_cols.append(nz[:, 1].numpy()); z_vals.append(z[nz[:, 0], nz[:, 1]].numpy())
    Z = sp.csr_matrix((np.concatenate(z_vals), (np.concatenate(z_rows), np.concatenate(z_cols))), shape=(len(ev), a.latents))
    Zb = Z.copy(); Zb.data[:] = 1
    f = np.load(M / "features.npz", allow_pickle=True)
    F = sp.csr_matrix((np.ones(len(f["col"]), np.float32), f["col"], f["ptr"]), shape=(n, len(f["names"])))[ev]
    names = np.array(f["names"], dtype=object)
    co = (Zb.T @ F).toarray()  # (latents, features): papers where the latent fires and the heading is present
    fire = np.asarray(Zb.sum(0)).ravel(); base = np.asarray(F.sum(0)).ravel() / len(ev)
    lift = co / np.maximum(fire, 1)[:, None] / np.maximum(base, 1e-9)[None]
    share = co / np.maximum(fire, 1)[:, None]
    score = np.where(co >= 20, np.log(np.maximum(lift, 1e-9)) * np.sqrt(share), -np.inf)
    # concepts nearest each decoder direction
    cdf = pd.read_parquet(M / "concepts.parquet"); C = np.load(M / "concept_def.npy").astype(np.float32)
    C -= C.mean(0); C /= np.linalg.norm(C, axis=1, keepdims=True)
    Wd = model.W_dec.detach().numpy().T  # (latents, 384), unit rows
    cs = Wd @ C.T; ci = cs.argmax(1)
    rows_out = []
    for j in range(a.latents):
        o = np.argsort(-score[j])[:4]; o = o[np.isfinite(score[j, o])]
        rows_out.append(dict(feature=j, fires=float(fire[j] / len(ev)), mesh=" | ".join(f"{names[i]} ({lift[j, i]:.0f}x)" for i in o),
                             concept=f"{cdf.name[ci[j]]} ({cdf.source[ci[j]]})", concept_cos=float(cs[j, ci[j]])))
    df = pd.DataFrame(rows_out).sort_values("fires", ascending=False); df.to_csv(OUT / "features.csv", index=False)
    alive = fire > 0
    # does the nearest concept agree with the MeSH naming? (the concept's name among the feature's top 4 headings)
    agree = np.mean([r["concept"].rsplit(" (", 1)[0] in r["mesh"] for r in rows_out if r["mesh"]])
    summ = dict(latents=a.latents, k=a.k, heldout_fve=fve, alive=float(alive.mean()),
                median_fire_rate=float(np.median(fire[alive] / len(ev))), named=float(np.mean([bool(r["mesh"]) for r in rows_out])),
                nearest_concept_is_a_top_heading=float(agree), median_concept_cos=float(np.median(cs.max(1))))
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1)); log(summ)
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(df.iloc[::max(1, len(df) // 40)].head(40)[["feature", "fires", "mesh", "concept", "concept_cos"]].to_string(index=False))


if __name__ == "__main__":
    main()
