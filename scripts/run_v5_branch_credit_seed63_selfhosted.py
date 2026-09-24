from __future__ import annotations

"""Run the seed63 branch-credit recovery on one self-hosted machine.

This script parallelizes the ten deterministic scenario shards locally, then
leaves the collected files in a directory consumable by the existing offline
critic fitter. No policy parameters are updated.
"""

from pathlib import Path
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys


SCENARIOS = list(range(1221, 1231))


def run_one(repo: Path, checkpoint: Path, out_root: Path, scenario: int) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo / "src"), str(repo), env.get("PYTHONPATH", "")]
    )
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    out = out_root / "seed63" / f"scenario{scenario}"
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(repo / "scripts" / "collect_v5_branch_credit_dataset.py"),
        "--checkpoint", str(checkpoint),
        "--training-seed", "63",
        "--scenario-start", str(scenario),
        "--scenario-end", str(scenario),
        "--out", str(out),
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"scenario {scenario} failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    summary_path = out / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"scenario {scenario}: missing summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("valid_probe_states", 0)) != 1:
        raise RuntimeError(
            f"scenario {scenario}: expected exactly one valid probe state, got {summary}"
        )
    return {
        "scenario": scenario,
        "destination_rows": int(summary["destination_rows"]),
        "target_rows": int(summary["target_rows"]),
        "stdout_tail": proc.stdout[-1000:],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--max-workers", type=int, default=3)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    checkpoint_arg = Path(args.checkpoint).resolve()
    out_root = Path(args.out_root).resolve()
    workers = max(1, min(int(args.max_workers), len(SCENARIOS)))

    if not checkpoint_arg.exists():
        raise FileNotFoundError(checkpoint_arg)
    if checkpoint_arg.is_dir():
        matches = sorted(checkpoint_arg.rglob("hierarchical_marl_final.pt"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one hierarchical_marl_final.pt under {checkpoint_arg}, got {matches}"
            )
        checkpoint = matches[0]
    else:
        checkpoint = checkpoint_arg

    print(
        json.dumps(
            {
                "mode": "self_hosted_local_parallel",
                "training_seed": 63,
                "scenarios": SCENARIOS,
                "max_workers": workers,
                "checkpoint": str(checkpoint),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(run_one, repo, checkpoint, out_root, scenario): scenario
            for scenario in SCENARIOS
        }
        for fut in concurrent.futures.as_completed(futs):
            scenario = futs[fut]
            result = fut.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "completed_scenario": scenario,
                        "destination_rows": result["destination_rows"],
                        "target_rows": result["target_rows"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    results.sort(key=lambda x: x["scenario"])
    if [x["scenario"] for x in results] != SCENARIOS:
        raise RuntimeError("scenario coverage mismatch")

    manifest = {
        "training_seed": 63,
        "scenarios": SCENARIOS,
        "max_workers": workers,
        "all_scenarios_complete": True,
        "total_destination_rows": sum(x["destination_rows"] for x in results),
        "total_target_rows": sum(x["target_rows"] for x in results),
        "results": results,
    }
    (out_root / "seed63_local_parallel_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
