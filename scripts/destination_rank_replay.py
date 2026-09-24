from __future__ import annotations

"""Replay the frozen probe states and rank destinations by the current heuristic.

No counterfactual branches and no training are run here.  The script joins the
heuristic ranking reconstructed from the exact probe state with the already
computed exhaustive destination counterfactual J values.
"""

from pathlib import Path
import argparse
import json

import numpy as np

from target_destination_counterfactual import load_model, find_probe, target_descriptor


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--tdcf-dir",required=True)
    ap.add_argument("--training-seed",type=int,required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    seed=int(a.training_seed)
    if seed not in {51,52,53}:
        raise ValueError(seed)
    root=Path(a.tdcf_dir)
    rows=json.loads((root/"destinations.json").read_text(encoding="utf-8"))
    model=load_model(Path(a.checkpoint))

    existing={}
    for r in rows:
        key=(int(r["scenario"]),int(r["training_seed"]),int(r["heuristic_rank"]),int(r["destination"]))
        if key in existing:
            raise RuntimeError(("duplicate destination row",key))
        existing[key]=r

    cases=[]
    for scenario in range(1101,1111):
        snapshot,meta=find_probe(model,scenario,seed,3)
        if snapshot is None:
            continue
        sim=snapshot.sim
        assert sim is not None
        block=int(meta["block"])
        candidates=list(sim.proactive_candidates(block))
        descriptors=[
            target_descriptor(snapshot,block,cid,rank)
            for rank,cid in enumerate(candidates,start=1)
        ]
        for desc in descriptors[:min(5,len(descriptors))]:
            moving=sim.containers[desc["moving_cid"]]
            scored=[]
            for dest in desc["feasible_destinations"]:
                stack=sim.yard.stacks[block][dest]
                inversion=sum(1 for lower_id in stack if sim.containers[lower_id].eta < moving.eta)
                height=len(stack)/sim.yard.max_tier
                score=float(inversion+0.35*height)
                scored.append((score,int(dest),int(inversion),float(height)))
            scored.sort(key=lambda x:(x[0],x[1]))
            ranks={dest:i+1 for i,(_,dest,_,_) in enumerate(scored)}
            if scored[0][1] != int(desc["heuristic_destination"]):
                raise RuntimeError(("heuristic ranking replay mismatch",scenario,seed,desc["heuristic_rank"]))

            j_by_dest={}
            for _,dest,_,_ in scored:
                key=(scenario,seed,int(desc["heuristic_rank"]),dest)
                if key not in existing:
                    raise RuntimeError(("missing counterfactual destination row",key))
                j_by_dest[dest]=float(existing[key]["J"])
            best_dest=min(j_by_dest,key=lambda d:(j_by_dest[d],d))
            best_j=j_by_dest[best_dest]
            best_rank=int(ranks[best_dest])

            topk_regret={}
            for k in (1,2,3,5):
                eligible=[d for d,rk in ranks.items() if rk<=min(k,len(ranks))]
                topk_regret[str(k)]=float(min(j_by_dest[d] for d in eligible)-best_j)

            heur_dest=int(desc["heuristic_destination"])
            cases.append({
                "scenario":scenario,
                "training_seed":seed,
                "target_heuristic_rank":int(desc["heuristic_rank"]),
                "target_cid":desc["target_cid"],
                "destination_count":len(scored),
                "heuristic_destination":heur_dest,
                "heuristic_destination_J":float(j_by_dest[heur_dest]),
                "counterfactual_best_destination":int(best_dest),
                "counterfactual_best_J":float(best_j),
                "best_destination_heuristic_rank":best_rank,
                "heuristic_destination_regret":float(j_by_dest[heur_dest]-best_j),
                "topk_regret":topk_regret,
            })

    if not cases:
        raise RuntimeError("no destination cases")

    summary={
        "training_seed":seed,
        "cases":len(cases),
        "mean_destination_count":float(np.mean([x["destination_count"] for x in cases])),
        "best_destination_rank_median":float(np.median([x["best_destination_heuristic_rank"] for x in cases])),
        "best_destination_rank_p90":float(np.quantile([x["best_destination_heuristic_rank"] for x in cases],0.90)),
        "top1_coverage":float(np.mean([x["best_destination_heuristic_rank"]<=1 for x in cases])),
        "top2_coverage":float(np.mean([x["best_destination_heuristic_rank"]<=2 for x in cases])),
        "top3_coverage":float(np.mean([x["best_destination_heuristic_rank"]<=3 for x in cases])),
        "top5_coverage":float(np.mean([x["best_destination_heuristic_rank"]<=5 for x in cases])),
        "mean_top1_regret":float(np.mean([x["topk_regret"]["1"] for x in cases])),
        "mean_top2_regret":float(np.mean([x["topk_regret"]["2"] for x in cases])),
        "mean_top3_regret":float(np.mean([x["topk_regret"]["3"] for x in cases])),
        "mean_top5_regret":float(np.mean([x["topk_regret"]["5"] for x in cases])),
        "p90_top5_regret":float(np.quantile([x["topk_regret"]["5"] for x in cases],0.90)),
    }

    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/"cases.json").write_text(json.dumps(cases,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(summary,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
