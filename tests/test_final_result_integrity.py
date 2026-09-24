import importlib.util
from pathlib import Path

import pytest


ROOT=Path(__file__).resolve().parents[1]


def load_script(name:str):
    path=ROOT/"scripts"/name
    spec=importlib.util.spec_from_file_location(name.replace(".py",""),path)
    module=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_final_learned_eval_rejects_duplicate_policy_seed_rows():
    mod=load_script("run_final_experiment.py")
    ev={"scenarios_start":901,"scenarios_end":930,"stochastic_repeats":3}
    rows=[]
    for scenario in range(901,931):
        for repeat in range(3):
            row={
                "seed":scenario,
                "policy_seed":scenario*100+repeat,
                "evaluation_mode":"stochastic",
                "rule_resolve_proactive_pair":"True",
                "include_resource_state":"True",
                "enable_proactive":"True",
            }
            row.update({m:1.0 for m in mod.METRICS})
            rows.append(row)
    mod.validate_learned_eval_rows(rows,ev,True)

    bad=list(rows)
    bad[-1]=dict(bad[-2])
    with pytest.raises(RuntimeError,match="duplicate"):
        mod.validate_learned_eval_rows(bad,ev,True)


def test_final_aggregator_rejects_duplicate_arm_seed_records():
    mod=load_script("summarize_final_experiment.py")
    expected={(arm,seed) for arm in mod.LEARNED for seed in [51,52,53]}
    records=[{"manifest":{"arm":arm,"training_seed":seed}} for arm,seed in sorted(expected)]
    assert set(mod.validate_unique_matrix_keys(records,expected))==expected

    bad=list(records)
    bad[-1]={"manifest":dict(records[-2]["manifest"])}
    with pytest.raises(RuntimeError):
        mod.validate_unique_matrix_keys(bad,expected)
