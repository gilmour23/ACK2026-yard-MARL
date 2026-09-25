from __future__ import annotations

"""Bounded self-contained preflight for V6 corrected hybrid MARL.

This is implementation validation only. It is intentionally tiny and does not
consume any future final evaluation bank.
"""

from pathlib import Path
import argparse
import hashlib
import json

from v6.storage_bc import train_common_storage_bc
from v6.train_hybrid import HybridPPOConfig, train_hybrid_ppo
from v6.evaluate_hybrid import evaluate_v6


KINDS = ("marl_resource", "marl_noresource", "single")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for i, kind in enumerate(KINDS):
        root = out / kind
        root.mkdir(parents=True, exist_ok=True)
        bc = root / "storage_bc.pt"
        bc_metrics = train_common_storage_bc(
            kind,
            bc,
            scenario_seeds=[1000, 1001],
            epochs=1,
            batch_size=64,
            rng_seed=20260926,
            hidden=32,
            pair_hidden=8,
        )

        include_resource = kind != "marl_noresource"
        cfg = HybridPPOConfig(
            total_episodes=2,
            episodes_per_update=2,
            update_epochs=1,
            critic_extra_epochs=1,
            minibatch_size=256,
            hidden=32,
            pair_hidden=8,
            seed=91 + i,
            include_resource_state=include_resource,
        )
        _, updates, episodes = train_hybrid_ppo(
            kind,
            cfg,
            root / "train",
            storage_bc_checkpoint=bc,
        )
        if len(updates) != 1:
            raise RuntimeError((kind, "unexpected update count", len(updates)))
        if int(updates[0]["episodes_in_batch"]) != 2:
            raise RuntimeError((kind, "multi-episode batch failed", updates[0]))
        if float(updates[0]["mc_completed_fraction"]) != 1.0:
            raise RuntimeError((kind, "nonterminal training batch"))
        if float(updates[0]["critic_extra_actor_max_abs_delta"]) != 0.0:
            raise RuntimeError((kind, "actor changed in critic-only step"))
        if len(episodes) != 2:
            raise RuntimeError((kind, "episode count mismatch", len(episodes)))

        ckpt = root / "train" / "v6_hybrid_final.pt"
        eval_summary = evaluate_v6(
            ckpt,
            root / "eval_1401.csv",
            scenarios=[1401],
            repeats=1,
            stochastic=True,
        )
        results[kind] = {
            "bc_sha256": sha256(bc),
            "bc_metrics": bc_metrics,
            "checkpoint_sha256": sha256(ckpt),
            "train_update": updates[0],
            "eval": eval_summary,
        }

    # The two MARL BC artifacts are architecture-identical but deliberately
    # differ because their declared observations differ.  Within each kind,
    # the artifact is built once and reused for all future seed fan-out.
    summary = {
        "status": "PASS",
        "purpose": "V6 implementation preflight only",
        "bc_scenarios": [1000, 1001],
        "training_episodes_per_arm": 2,
        "episodes_per_update": 2,
        "diagnostic_eval_scenario": 1401,
        "results": results,
    }
    (out / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "kinds": list(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
