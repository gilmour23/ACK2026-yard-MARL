from __future__ import annotations

"""Fit grouped-CV offline Target/Destination critics for the V5 credit diagnostic."""

from pathlib import Path
import argparse
import json
import math
from collections import defaultdict

import numpy as np
import torch
from torch import nn


torch.set_num_threads(1)

FOLDS=[
    {1221,1222},
    {1223,1224},
    {1225,1226},
    {1227,1228},
    {1229,1230},
]
MODEL_SEEDS=[20260924,20260925,20260926]
EPOCHS=400
BATCH_SIZE=256
LR=1e-3
WEIGHT_DECAY=1e-5


class CriticMLP(nn.Module):
    def __init__(self,in_dim:int):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim,128),nn.Tanh(),
            nn.Linear(128,64),nn.Tanh(),
            nn.Linear(64,1),
        )
    def forward(self,x):
        return self.net(x).squeeze(-1)


def load_rows(root:Path,name:str)->list[dict]:
    rows=[]
    for p in sorted(root.rglob(name)):
        obj=json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj,list):
            raise RuntimeError((p,type(obj)))
        rows.extend(obj)
    return rows


def fit_predict_ensemble(
    x_train:np.ndarray,y_train:np.ndarray,x_test:np.ndarray,
)->np.ndarray:
    x_mean=x_train.mean(axis=0)
    x_std=x_train.std(axis=0)
    x_std=np.where(x_std<1e-8,1.0,x_std)
    y_mean=float(y_train.mean())
    y_std=float(y_train.std())
    if y_std<1e-8:
        y_std=1.0
    xt=((x_train-x_mean)/x_std).astype(np.float32)
    yt=((y_train-y_mean)/y_std).astype(np.float32)
    xv=((x_test-x_mean)/x_std).astype(np.float32)

    preds=[]
    for seed in MODEL_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model=CriticMLP(xt.shape[1])
        opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
        rng=np.random.default_rng(seed)
        n=len(xt)
        for _ in range(EPOCHS):
            order=rng.permutation(n)
            for st in range(0,n,BATCH_SIZE):
                ids=order[st:st+BATCH_SIZE]
                xb=torch.as_tensor(xt[ids],dtype=torch.float32)
                yb=torch.as_tensor(yt[ids],dtype=torch.float32)
                pred=model(xb)
                loss=nn.functional.mse_loss(pred,yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            p=model(torch.as_tensor(xv,dtype=torch.float32)).cpu().numpy()
        preds.append(p*y_std+y_mean)
    return np.mean(np.stack(preds,axis=0),axis=0)


def rankdata_simple(x:np.ndarray)->np.ndarray:
    """Average ranks for ties; 0-based ranks are sufficient for correlation."""
    x=np.asarray(x)
    order=np.argsort(x,kind="mergesort")
    ranks=np.empty(len(x),dtype=np.float64)
    i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:
            j+=1
        r=(i+j-1)/2.0
        ranks[order[i:j]]=r
        i=j
    return ranks


def spearman(x:np.ndarray,y:np.ndarray)->float:
    if len(x)<2:
        return float("nan")
    rx=rankdata_simple(np.asarray(x,dtype=np.float64))
    ry=rankdata_simple(np.asarray(y,dtype=np.float64))
    if rx.std()<1e-12 or ry.std()<1e-12:
        return float("nan")
    return float(np.corrcoef(rx,ry)[0,1])


def pairwise_counts(pred:np.ndarray,label:np.ndarray)->tuple[int,int]:
    correct=0
    total=0
    n=len(pred)
    for i in range(n):
        for j in range(i+1,n):
            dy=float(label[i]-label[j])
            if abs(dy)<1e-12:
                continue
            dp=float(pred[i]-pred[j])
            total+=1
            if dp*dy>0:
                correct+=1
            elif abs(dp)<1e-12:
                correct+=0.5
    return correct,total


def evaluate_groups(rows:list[dict],pred:np.ndarray,branch:str)->dict:
    if len(rows)!=len(pred):
        raise ValueError((len(rows),len(pred)))
    groups=defaultdict(list)
    for i,row in enumerate(rows):
        if branch=="target":
            key=(int(row["training_seed"]),int(row["scenario"]))
            label=float(row["Q_target_label"])
            j=float(row["expected_J_under_destination_policy"])
            heur_rank=int(row["target_heuristic_rank"])
        else:
            key=(
                int(row["training_seed"]),
                int(row["scenario"]),
                int(row["target_position"]),
            )
            label=float(row["Q_destination_label"])
            j=float(row["J"])
            heur_rank=int(row["destination_heuristic_rank"])
        groups[key].append((float(pred[i]),label,j,heur_rank))

    total_correct=0.0
    total_pairs=0
    spears=[]
    critic_regrets=[]
    random_regrets=[]
    heuristic_regrets=[]

    for vals in groups.values():
        p=np.asarray([v[0] for v in vals],dtype=np.float64)
        y=np.asarray([v[1] for v in vals],dtype=np.float64)
        j=np.asarray([v[2] for v in vals],dtype=np.float64)
        h=np.asarray([v[3] for v in vals],dtype=np.int64)
        c,t=pairwise_counts(p,y)
        total_correct+=c
        total_pairs+=t
        s=spearman(p,y)
        if math.isfinite(s):
            spears.append(s)

        best_j=float(j.min())
        chosen=int(np.argmax(p))
        critic_regrets.append(float(j[chosen]-best_j))
        random_regrets.append(float(j.mean()-best_j))
        hidx=np.flatnonzero(h==1)
        if len(hidx)!=1:
            raise RuntimeError(("heuristic rank1 cardinality",len(hidx)))
        heuristic_regrets.append(float(j[int(hidx[0])]-best_j))

    if total_pairs<=0:
        raise RuntimeError("no pairwise comparisons")
    return {
        "groups":len(groups),
        "pairwise_accuracy":float(total_correct/total_pairs),
        "mean_spearman":float(np.mean(spears)) if spears else float("nan"),
        "mean_critic_top1_regret":float(np.mean(critic_regrets)),
        "mean_uniform_random_regret":float(np.mean(random_regrets)),
        "mean_heuristic_top1_regret":float(np.mean(heuristic_regrets)),
        "critic_vs_random_regret_ratio":(
            float(np.mean(critic_regrets))/float(np.mean(random_regrets))
            if float(np.mean(random_regrets))>1e-12 else 0.0
        ),
    }


def run_branch(rows:list[dict],branch:str)->dict:
    if branch=="target":
        xkey="x_target"; ykey="Q_target_label"
    elif branch=="destination":
        xkey="x_destination"; ykey="Q_destination_label"
    else:
        raise ValueError(branch)

    fold_results=[]
    all_test_rows=[]
    all_test_pred=[]

    for fold_idx,test_scenarios in enumerate(FOLDS,1):
        train=[r for r in rows if int(r["scenario"]) not in test_scenarios]
        test=[r for r in rows if int(r["scenario"]) in test_scenarios]
        if not train or not test:
            raise RuntimeError(("empty fold",fold_idx,branch))
        xtr=np.asarray([r[xkey] for r in train],dtype=np.float64)
        ytr=np.asarray([float(r[ykey]) for r in train],dtype=np.float64)
        xte=np.asarray([r[xkey] for r in test],dtype=np.float64)
        pred=fit_predict_ensemble(xtr,ytr,xte)
        metrics=evaluate_groups(test,pred,branch)
        metrics.update({
            "fold":fold_idx,
            "test_scenarios":sorted(test_scenarios),
            "train_rows":len(train),
            "test_rows":len(test),
        })
        fold_results.append(metrics)
        all_test_rows.extend(test)
        all_test_pred.extend(pred.tolist())

    pooled=evaluate_groups(
        all_test_rows,np.asarray(all_test_pred,dtype=np.float64),branch
    )
    fold_order_pass=sum(float(f["pairwise_accuracy"])>0.55 for f in fold_results)
    gate=bool(
        float(pooled["pairwise_accuracy"])>=0.60
        and float(pooled["critic_vs_random_regret_ratio"])<=0.75
        and fold_order_pass>=4
    )
    return {
        "branch":branch,
        "rows":len(rows),
        "input_dim":len(rows[0][xkey]),
        "folds":fold_results,
        "pooled":pooled,
        "folds_pairwise_gt_055":fold_order_pass,
        "gate_pass":gate,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    root=Path(a.root)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)

    target_rows=load_rows(root,"target_rows.json")
    dest_rows=load_rows(root,"destination_rows.json")
    summaries=[]
    for p in sorted(root.rglob("summary.json")):
        s=json.loads(p.read_text(encoding="utf-8"))
        if "training_seed" in s and "valid_probe_states" in s:
            summaries.append(s)
    if {int(s["training_seed"]) for s in summaries}!={61,62,63}:
        raise RuntimeError(("source seed set mismatch",summaries))

    target=run_branch(target_rows,"target")
    destination=run_branch(dest_rows,"destination")
    both=bool(target["gate_pass"] and destination["gate_pass"])
    if both:
        decision="branch_q_representation_sufficient_for_bounded_credit_pilot"
    elif target["gate_pass"] or destination["gate_pass"]:
        decision="partial_sufficiency_redesign_failed_branch_before_policy_training"
    else:
        decision="current_features_insufficient_for_branch_q_ranking"

    result={
        "protocol":"audits/V5_BRANCH_CREDIT_OFFLINE_PROTOCOL_20260924.md",
        "source_training_seeds":[61,62,63],
        "scenarios":[1221,1230],
        "source_summaries":sorted(summaries,key=lambda x:int(x["training_seed"])),
        "critic_training":{
            "architecture":"input-128-64-1_tanh",
            "epochs":EPOCHS,
            "batch_size":BATCH_SIZE,
            "learning_rate":LR,
            "weight_decay":WEIGHT_DECAY,
            "model_seeds":MODEL_SEEDS,
            "folds":[sorted(x) for x in FOLDS],
        },
        "target":target,
        "destination":destination,
        "both_branches_pass":both,
        "decision":decision,
        "note":"Offline supervised critic diagnostic only; no policy parameters were updated.",
    }
    (out/"summary.json").write_text(
        json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8"
    )

    lines=[
        "# V5 branch-credit offline sufficiency diagnostic",
        "",
        f"Decision: **{decision}**",
        "",
        "## Target critic",
        f"- held-out pairwise accuracy: **{target['pooled']['pairwise_accuracy']:.3f}**",
        f"- mean within-state Spearman: **{target['pooled']['mean_spearman']:.3f}**",
        f"- critic Top-1 regret: **{target['pooled']['mean_critic_top1_regret']:.4f} J**",
        f"- uniform-random regret: **{target['pooled']['mean_uniform_random_regret']:.4f} J**",
        f"- heuristic Top-1 regret: **{target['pooled']['mean_heuristic_top1_regret']:.4f} J**",
        f"- critic/random regret ratio: **{target['pooled']['critic_vs_random_regret_ratio']:.3f}**",
        f"- folds pairwise > 0.55: **{target['folds_pairwise_gt_055']}/5**",
        f"- gate: **{'PASS' if target['gate_pass'] else 'FAIL'}**",
        "",
        "## Destination critic",
        f"- held-out pairwise accuracy: **{destination['pooled']['pairwise_accuracy']:.3f}**",
        f"- mean within-state Spearman: **{destination['pooled']['mean_spearman']:.3f}**",
        f"- critic Top-1 regret: **{destination['pooled']['mean_critic_top1_regret']:.4f} J**",
        f"- uniform-random regret: **{destination['pooled']['mean_uniform_random_regret']:.4f} J**",
        f"- heuristic Top-1 regret: **{destination['pooled']['mean_heuristic_top1_regret']:.4f} J**",
        f"- critic/random regret ratio: **{destination['pooled']['critic_vs_random_regret_ratio']:.3f}**",
        f"- folds pairwise > 0.55: **{destination['folds_pairwise_gt_055']}/5**",
        f"- gate: **{'PASS' if destination['gate_pass'] else 'FAIL'}**",
        "",
        "No policy retraining was performed.",
    ]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({
        "target":target["pooled"],
        "target_gate":target["gate_pass"],
        "destination":destination["pooled"],
        "destination_gate":destination["gate_pass"],
        "decision":decision,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
