"""Optional stage 3: a small instruction LLM judges the pairs closest to the decision threshold.

Why: after stage 2 the remaining errors sit near the threshold. LLM matchers are strong
zero-shot and robust on entities unlike the training data (Peeters & Bizer, 2023/2024),
which matters for France (not in train). DoorDash's production linker routes exactly this
middle band to a stronger judge. An LLM is far too slow for every pair, so only the
``max_pairs_test`` test pairs nearest the threshold are judged.

Rules: rule 5 allows only MIT/Apache-2.0 models of <= 8B parameters, so ``v5.llm.model``
must match a prefix in ``v5.llm.allowed_models`` (checked from the model cards). External
services that resolve entities are prohibited, so the intended endpoint is a LOCAL Ollama
server on the GPU machine; ``base_url`` + ``OLLAMA_API_KEY`` also reach a remote Ollama API.

Method (no labels reach the LLM; it is zero-shot, so any fold may be used):
  select   the stage-2 threshold rule gives each arbitrated pair its threshold (T_first for an
           entity's best candidate, T_rest otherwise); test pairs are ranked by |p2 - T| and
           the nearest ``max_pairs_test`` are judged. The same |p2 - T| band is applied to train
           folds 0-2, where whole entities are sampled up to ``max_pairs_train`` pairs.
  score    the LLM returns {"same_business": bool, "confidence": 0-100} (JSON schema output);
           s = confidence if same else 100 - confidence, scaled to [0, 1]. Answers are cached
           (resumable; re-running only sends the missing pairs).
  apply    logistic regression on [logit p2, logit s] fitted on judged pairs of the tuning
           folds; judged pairs get the combined probability; the stage-2 decision rule is
           re-applied. Fold 0: paired bootstrap on the judged entities (the only ones that can
           change). Writes subs/<name>/ from the test pairs; upload only if the CI is > 0.

Run:  python -m ber.llm_judge check
      python -m ber.llm_judge score --split train
      python -m ber.llm_judge score --split test
      python -m ber.llm_judge apply --name sub_v5_llm
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import v5 as V
from .baseline import Decider, EntityScorer, entity_key
from .config import ensure_parent, load_config
from .eval.scorer import paired_bootstrap
from .neural.common import country_store
from .store import split_countries

SYSTEM = (
    "You are an expert in business entity resolution. Decide whether two records describe the SAME "
    "real-world business. The records come from different sources and are noisy: abbreviations "
    "(pvt/private, ltd/limited, corp/corporation, st/street, rd/road), missing or extra legal suffixes, "
    "typos, word reordering, names transliterated between Indian scripts and Latin letters, partial or "
    "missing addresses, landmark references (near ...). Text is lowercased and punctuation removed. "
    "Branches of a chain with the same name at different addresses are DIFFERENT businesses. A different "
    "house, unit or suite number, or a different postal code, usually means a different business. "
    "A missing address component alone is not evidence of a different business."
)
SCHEMA = {"type": "object",
          "properties": {"same_business": {"type": "boolean"},
                         "confidence": {"type": "integer", "minimum": 0, "maximum": 100}},
          "required": ["same_business", "confidence"]}


# ------------------------------------------------------------------ config / client
def lcfg() -> dict:
    return V.vcfg().get("llm") or {}


def enabled() -> bool:
    return bool(lcfg().get("enabled", False))


def model_card() -> str:
    """The allow-list entry of the configured model; refuses anything else (rule 5)."""
    c = lcfg()
    model = c["model"]
    for prefix, facts in (c.get("allowed_models") or {}).items():
        if model.startswith(prefix):
            return facts
    raise SystemExit(f"model {model!r} is not in v5.llm.allowed_models: only MIT/Apache-2.0 models "
                     f"of <= 8B parameters are allowed (check the model card, then add it)")


def cache_path(split: str) -> str:
    return V.run_path("llm", f"{split}{V.sfx(lcfg().get('stage2_tag', ''))}.jsonl")


def sel_path(split: str) -> str:
    return V.run_path("llm", f"sel_{split}{V.sfx(lcfg().get('stage2_tag', ''))}.npy")


def pair_prompt(a_name, a_addr, b_name, b_addr, country) -> str:
    return (f"Country: {country}\nRecord A: name: {a_name or '(empty)'} | address: {a_addr or '(empty)'}\n"
            f"Record B: name: {b_name or '(empty)'} | address: {b_addr or '(empty)'}\n"
            "Are A and B the same business? Answer in JSON.")


def ask(prompt: str) -> float:
    """One chat call; returns s in [0, 1] (NaN if every retry failed)."""
    c = lcfg()
    body = {"model": c["model"], "stream": False, "format": SCHEMA,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "options": {"temperature": 0, "num_predict": 40}}
    if c.get("think") is not None:
        body["think"] = bool(c["think"])
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OLLAMA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OLLAMA_API_KEY']}"
    req = urllib.request.Request(c["base_url"].rstrip("/") + "/api/chat",
                                 data=json.dumps(body).encode(), headers=headers, method="POST")
    for attempt in range(int(c.get("retries", 3))):
        try:
            with urllib.request.urlopen(req, timeout=c.get("timeout_s", 60)) as r:
                out = json.loads(json.loads(r.read())["message"]["content"])
            conf = min(max(float(out["confidence"]), 0.0), 100.0) / 100.0
            return conf if out["same_business"] else 1.0 - conf
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
            time.sleep(1.5 * (attempt + 1))
    return float("nan")


# ------------------------------------------------------------------ selection
def stage2_state(split: str):
    """(pruned keys, ents, p2, stage-2 result) of the configured stage-2 variant."""
    tag = lcfg().get("stage2_tag", "")
    res = json.load(open(V.run_path(f"stage2{V.sfx(tag)}.json")))
    keys, ents = V.read_keys(split)
    kept = np.load(V.run_path(f"kept_{split}{V.sfx(tag)}.npy"))
    p2 = np.load(V.run_path(f"p2_{split}{V.sfx(tag)}.npy"))
    kk = keys[kept].reset_index(drop=True)
    if len(p2) != len(kk):
        raise ValueError(f"p2_{split} has {len(p2)} rows for {len(kk)} pruned pairs: rerun stage2 / predict")
    return kk, ents, p2, res


def decision_distance(kk: pd.DataFrame, p: np.ndarray, rules: dict) -> np.ndarray:
    """|p - T| for pairs that won arbitration (T_first for an entity's best candidate, else
    T_rest); inf for pairs that lost arbitration (they are never predicted)."""
    dec = Decider(kk, p)
    t = rules["threshold"]
    thr = np.where(dec.pos == 0, t["t_first"], t["t_rest"])
    return np.where(dec.rec_best, np.abs(p - thr), np.inf)


def select(split: str) -> np.ndarray:
    """Row indices (into the pruned pairs) to judge; the test band also defines the train band."""
    c = lcfg()
    kk, ents, p2, res = stage2_state(split)
    d = decision_distance(kk, p2, res["rules"])
    band_path = V.run_path("llm", f"band{V.sfx(c.get('stage2_tag', ''))}.json")
    if split == "test":
        n = min(int(c["max_pairs_test"]), int(np.isfinite(d).sum()))
        idx = np.sort(np.argsort(d, kind="stable")[:n])
        ensure_parent(band_path)
        json.dump({"d_max": float(d[idx].max()) if n else 0.0, "n_test": int(n)}, open(band_path, "w"))
        return idx
    if not os.path.exists(band_path):
        raise SystemExit("score the test split first: its band defines the train band")
    d_max = json.load(open(band_path))["d_max"]
    fold = V.pair_folds(kk, ents)
    cand = np.flatnonzero((d <= d_max) & np.isin(fold, V.vcfg()["gbdt_folds"]))
    ek = entity_key(kk["country"].to_numpy()[cand], kk["s1_row"].to_numpy()[cand])
    order = np.random.default_rng(load_config()["seed"] + 23).permutation(np.unique(ek))
    per = pd.Series(1, index=ek).groupby(level=0).size().reindex(order).to_numpy()
    take = set(order[np.cumsum(per) <= int(c["max_pairs_train"])].tolist())
    return np.sort(cand[pd.Series(ek).isin(take).to_numpy()])


def pair_key(kk: pd.DataFrame, idx: np.ndarray) -> list[str]:
    cols = [kk[k].to_numpy()[idx] for k in ("country", "src", "doc_row", "s1_row")]
    return [f"{a}-{b}-{c}-{d}" for a, b, c, d in zip(*cols)]


def read_cache(split: str) -> dict:
    out = {}
    path = cache_path(split)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    out[r["k"]] = r["s"]
                except (json.JSONDecodeError, KeyError):
                    continue                        # a line cut by a crash
    return out


# ------------------------------------------------------------------ stages
def check() -> None:
    """Endpoint + licence check and a timed probe (ETA for the configured budgets)."""
    c = lcfg()
    print(f"model {c['model']}: {model_card()}; endpoint {c['base_url']}", flush=True)
    prompts = [pair_prompt("alpha traders pvt ltd", "12 mg road pune 411001", "alpha traders private limited",
                           "mg road 12 pune", "India"),
               pair_prompt("subway", "400 main st springfield 62701", "subway", "500 main st springfield 62701", "US"),
               pair_prompt("boulangerie paul", "5 rue de rivoli 75001 paris", "paul sarl", "5 rue rivoli paris", "France")] * 8
    t = time.time()
    with ThreadPoolExecutor(max_workers=int(c.get("concurrency", 16))) as ex:
        s = list(ex.map(ask, prompts))
    rate = len(prompts) / (time.time() - t)
    print(f"answers (same-business score): {np.round(s[:3], 2).tolist()} (expected high, low, high)")
    n = int(c["max_pairs_test"]) + int(c["max_pairs_train"])
    print(f"{rate:.1f} pairs/s at concurrency {c.get('concurrency', 16)} -> ~{n / max(rate, 1e-9) / 60:.0f} min "
          f"for {n:,} pairs", flush=True)
    if np.isnan(s).all():
        raise SystemExit("no answer from the endpoint: is `ollama serve` running and the model pulled?")


def score(split: str) -> None:
    """Judge the selected pairs of ``split`` (resumable cache)."""
    c = lcfg()
    model_card()
    kk, _, _, _ = stage2_state(split)
    idx = select(split)
    ensure_parent(sel_path(split))
    np.save(sel_path(split), idx)
    keys = pair_key(kk, idx)
    cache = read_cache(split)
    todo = [i for i, k in enumerate(keys) if k not in cache or cache[k] is None]
    print(f"[llm {split}] {len(idx):,} pairs selected, {len(idx) - len(todo):,} cached, {len(todo):,} to ask",
          flush=True)
    if not todo:
        return
    names = split_countries(split)
    country = kk["country"].to_numpy()[idx]
    prompts = [None] * len(idx)
    for code in np.unique(country):
        st = country_store(split, names[code], cols=["name_n", "addr_n"])
        m = np.flatnonzero(country == code)
        rows = idx[m]
        s1 = kk["s1_row"].to_numpy()[rows]
        src, doc = kk["src"].to_numpy()[rows], kk["doc_row"].to_numpy()[rows]
        an, aa = st.strings(1, "name_n", s1), st.strings(1, "addr_n", s1)
        bn, ba = [None] * len(m), [None] * len(m)
        for s in (2, 3):
            ms = np.flatnonzero(src == s)
            if len(ms):
                for j, x, y in zip(ms, st.strings(s, "name_n", doc[ms]), st.strings(s, "addr_n", doc[ms])):
                    bn[j], ba[j] = x, y
        for j, i in enumerate(m):
            prompts[i] = pair_prompt(an[j], aa[j], bn[j], ba[j], names[code])
    t0, done = time.time(), 0
    with open(cache_path(split), "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=int(c.get("concurrency", 16))) as ex:
        futs = {ex.submit(ask, prompts[i]): i for i in todo}
        for fut in as_completed(futs):
            s = fut.result()
            fh.write(json.dumps({"k": keys[futs[fut]], "s": None if np.isnan(s) else round(s, 4)}) + "\n")
            done += 1
            if done % 500 == 0 or done == len(todo):
                fh.flush()
                rate = done / (time.time() - t0)
                print(f"[llm {split}] {done:,}/{len(todo):,}  {rate:.1f} pairs/s  "
                      f"ETA {(len(todo) - done) / max(rate, 1e-9) / 60:.0f} min", flush=True)


def llm_scores(split: str, kk: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(selected rows, their s; NaN where the LLM gave no answer)."""
    idx = np.load(sel_path(split))
    cache = read_cache(split)
    s = np.array([cache.get(k) if cache.get(k) is not None else np.nan for k in pair_key(kk, idx)], dtype=np.float64)
    return idx, s


def features(p: np.ndarray, s: np.ndarray) -> np.ndarray:
    lp = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    ls = np.log(np.clip(s, 0.02, 0.98) / (1 - np.clip(s, 0.02, 0.98)))
    return np.column_stack([lp, ls, lp * ls])


def apply(name: str) -> None:
    """Fit the p2 + LLM combiner on tuning folds, measure on fold 0, write the test submission."""
    from sklearn.linear_model import LogisticRegression

    from .submit import finalize_submission

    v = V.vcfg()
    rep = v["report_fold"]
    folds = np.array(v["gbdt_folds"])
    kk, ents, p2, res = stage2_state("train")
    idx, s = llm_scores("train", kk)
    ok = np.isfinite(s)
    idx, s = idx[ok], s[ok]
    fold = V.pair_folds(kk, ents)[idx]
    y = kk["label"].to_numpy()[idx]
    fit = np.isin(fold, folds[folds != rep])
    if fit.sum() < int(lcfg().get("min_fit_pairs", 200)) or len(np.unique(y[fit])) < 2:
        raise SystemExit(f"only {int(fit.sum())} judged tuning-fold pairs: raise v5.llm.max_pairs_train")
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(features(p2[idx[fit]], s[fit]), y[fit])
    p_new = p2.copy()
    p_new[idx] = lr.predict_proba(features(p2[idx], s))[:, 1].astype(np.float32)
    rules = res["rules"]
    keep_old, keep_new = V.apply_rule(kk, p2, rules), V.apply_rule(kk, p_new, rules)
    # fold-0 entities that have a judged pair: the only ones whose prediction can change
    ek_all = entity_key(ents["country"], ents["s1_row"])
    judged = set(entity_key(kk["country"].to_numpy()[idx[fold == rep]], kk["s1_row"].to_numpy()[idx[fold == rep]]).tolist())
    emask = (ents["fold"] == rep).to_numpy() & pd.Series(ek_all).isin(judged).to_numpy()
    if not emask.any():
        raise SystemExit(f"no judged fold-{rep} entity: the effect cannot be measured, so no submission is "
                         f"written (raise v5.llm.max_pairs_train)")
    sc = EntityScorer(ents, kk, emask)
    f_old, f_new = sc.per_entity(keep_old), sc.per_entity(keep_new)
    boot = load_config()["bootstrap"]
    bs = paired_bootstrap(f_old, f_new, boot["n_resamples"], boot["alpha"])
    n_rep = int((ents["fold"] == rep).sum())
    kt, _, p2t, _ = stage2_state("test")
    idx_t, s_t = llm_scores("test", kt)
    band_share_test = len(idx_t) / max(np.isfinite(decision_distance(kt, p2t, rules)).sum(), 1)
    out = {"model": lcfg()["model"], "model_card": model_card(), "coef": lr.coef_.tolist(),
           "intercept": lr.intercept_.tolist(), "fit_pairs": int(fit.sum()),
           "fold0_judged_entities": int(emask.sum()), "fold0_judged_delta": bs,
           "fold0_macro_delta_equivalent": bs["delta"] * emask.sum() / n_rep,
           "note": "delta is measured on the sampled judged entities; on test every band pair is judged, "
                   "so the full effect scales with the band coverage",
           "test_band_share_of_arbitrated_pairs": float(band_share_test)}
    verdict = "KEEP" if bs["ci_low"] > 0 else ("WORSE" if bs["ci_high"] < 0 else "NO CLEAR DIFFERENCE")
    out["verdict"] = verdict
    print(f"[llm apply] fold-{rep} judged entities {int(emask.sum()):,}: F0.5 {f_old.mean():.5f} -> "
          f"{f_new.mean():.5f}  delta {bs['delta']:+.5f}  CI [{bs['ci_low']:+.5f}, {bs['ci_high']:+.5f}]  "
          f"-> {verdict}", flush=True)
    json.dump(out, open(V.run_path(f"llm{V.sfx(lcfg().get('stage2_tag', ''))}.json"), "w"), indent=2, default=float)
    # test
    ok_t = np.isfinite(s_t)
    pt = p2t.copy()
    pt[idx_t[ok_t]] = lr.predict_proba(features(p2t[idx_t[ok_t]], s_t[ok_t]))[:, 1].astype(np.float32)
    keep_t = V.apply_rule(kt, pt, rules)
    mpath, cpath, n_m, n_c = V.write_outputs(kt, keep_t, "llm predict")
    finalize_submission(name, mpath, cpath, offline_metrics={"stage2": res["report"], "llm": out},
                        n_match_pairs=n_m, n_candidate_pairs=n_c,
                        notes=f"v5 stage2 + LLM judge ({lcfg()['model']}) on {len(idx_t):,} threshold-band test pairs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    s = sub.add_parser("score")
    s.add_argument("--split", required=True, choices=["train", "test"])
    a = sub.add_parser("apply")
    a.add_argument("--name", required=True)
    args = ap.parse_args()
    if not enabled():
        print("LLM judge disabled (v5.llm.enabled: false): nothing to do", flush=True)
        return
    V.setup_logging()
    if args.cmd == "check":
        check()
    elif args.cmd == "score":
        score(args.split)
    else:
        apply(args.name)


if __name__ == "__main__":
    main()
