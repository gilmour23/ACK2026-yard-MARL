# V5 branch-credit offline recovery — seed63 sharding — 2026-09-25

Status: infrastructure recovery only. Scientific protocol is unchanged.

## Reason

Original workflow run `35994074454` produced complete artifacts for training seeds 61 and 62, but seed63 hit the 240-minute job timeout during exhaustive counterfactual collection. The offline fitting job was therefore skipped.

This recovery does not change:
- source checkpoint;
- scenarios 1221–1230;
- probe-state selection rule;
- forced Target×Destination enumeration;
- frozen greedy continuation;
- labels;
- critic features;
- cross-validation folds;
- critic architecture/training;
- sufficiency gates.

## Recovery method

Re-run **seed63 only**, with scenarios 1221–1230 split into ten independent jobs, one scenario per job.

The collector is deterministic for a fixed:
- V5 seed63 checkpoint;
- scenario;
- policy probe seed rule `scenario*100`.

Therefore scenario sharding changes only execution granularity, not the diagnostic data-generating process.

After all ten seed63 shards finish:
1. reuse the completed seed61 artifact from run `35994074454`;
2. reuse the completed seed62 artifact from run `35994074454`;
3. combine them with the ten seed63 scenario shards;
4. enforce exact coverage of all 30 `(training_seed, scenario)` probe states and reject duplicate Target/Destination rows;
5. run the pre-registered 5-fold offline critic fit.

No policy retraining is performed.
