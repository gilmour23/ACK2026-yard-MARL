from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    root=Path(a.root); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)

    summaries=[]
    cases=[]
    for p in sorted(root.rglob("summary.json")):
        s=json.loads(p.read_text(encoding="utf-8"))
        if "training_seed" in s and "cases" in s:
            summaries.append(s)
    for p in sorted(root.rglob("cases.json")):
        cases.extend(json.loads(p.read_text(encoding="utf-8")))
    if len(summaries)!=3 or {int(s["training_seed"]) for s in summaries}!={51,52,53}:
        raise RuntimeError(("expected seed summaries 51/52/53",summaries))
    if not cases:
        raise RuntimeError("no cases")

    pooled={
        "source_training_seeds":[51,52,53],
        "cases":len(cases),
        "mean_destination_count":float(np.mean([x["destination_count"] for x in cases])),
        "best_destination_rank_median":float(np.median([x["best_destination_heuristic_rank"] for x in cases])),
        "best_destination_rank_p90":float(np.quantile([x["best_destination_heuristic_rank"] for x in cases],0.90)),
    }
    for k in (1,2,3,5):
        pooled[f"top{k}_coverage"]=float(np.mean([x["best_destination_heuristic_rank"]<=k for x in cases]))
        pooled[f"mean_top{k}_regret"]=float(np.mean([x["topk_regret"][str(k)] for x in cases]))
        pooled[f"median_top{k}_regret"]=float(np.median([x["topk_regret"][str(k)] for x in cases]))
        pooled[f"p90_top{k}_regret"]=float(np.quantile([x["topk_regret"][str(k)] for x in cases],0.90))

    result={
        "per_seed":{str(int(s["training_seed"])):s for s in sorted(summaries,key=lambda z:int(z["training_seed"]))},
        "pooled":pooled,
        "interpretation_note":"No training and no new counterfactual continuation. Heuristic ranks are replayed on the frozen probe states and joined to previously computed exhaustive destination J values."
    }
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")

    lines=[
        "# Destination heuristic-rank coverage",
        "",
        f"Cases: {pooled['cases']}",
        f"Mean feasible destinations per tested target: {pooled['mean_destination_count']:.2f}",
        f"Best-destination heuristic rank: median {pooled['best_destination_rank_median']:.1f}, p90 {pooled['best_destination_rank_p90']:.1f}",
        "",
    ]
    for k in (1,2,3,5):
        lines.append(
            f"- Top-{k}: coverage {pooled[f'top{k}_coverage']:.3f}, "
            f"mean regret {pooled[f'mean_top{k}_regret']:.4f} J, "
            f"p90 regret {pooled[f'p90_top{k}_regret']:.4f} J"
        )
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(pooled,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
