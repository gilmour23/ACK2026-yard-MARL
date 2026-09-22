# Fixed-policy critic diagnostic — preregistered protocol

Base: f6e40f53e2ac7f34bea0dcd4c8761204cc3b965f. Branch: experiment/critic-diagnostic only.
Checkpoint SHA256: 2bbd3a2e795a215d3fcbd654d58fad154d4298b55bb50ce92f54ad50b82ddb59.
Protocol written before critic-only fitting; completed 2k A/B is preserved, not extended.

## Data and endpoints

Reuse canonical fixed-policy validation data: scenarios 601–610, three stochastic policy seeds per scenario (`scenario*100+repeat`). Train 601–607; held-out diagnostic validation 608–610. These are reused diagnostic scenarios, not unseen final tests. Targets are exact terminal-complete undiscounted MC returns. Reconstruct missing metadata by replaying only the canonical episodes, verifying observations, values, and MC against the old data. Do not regenerate six training runs or the other 180 evaluations.

Baseline: ordinary least squares, numpy.linalg.lstsq, predictors [1, critic_observation[2]], no regularization, fitted on train scenarios only. Report EV, RMSE, R2, bias, prediction and target variance, and 60-minute calibration bins. EV alone does not measure calibration bias.

## Same-architecture fitting

PermutationInvariantCritic, hidden128, unchanged architecture. Adam lr3e-4, batch256, max_grad_norm0.5; loss0.25*MSE, identical to canonical vf_coef0.5 times half-MSE. No value normalization. Twenty full epochs, last epoch endpoint, no best-validation checkpoint selection. Seeds21,22,23. Primary condition: warm-start canonical critic. Secondary initialization control: fresh critic of exactly the same architecture, same data/batch order/optimizer. This tests inherited saturation, not a new architecture. Actor is not in the critic optimizer and is never updated. Log every optimizer step; evaluate full train/validation each epoch. No hyperparameter search or 4k/6k/30k policy training.

Engineering gates specified by user: held-out EV>=0.20 supports coarse representational sufficiency; >=0.50 is stronger evidence. If both neural conditions remain <=0.10 while regression is strong, examine neural optimization/normalization/implementation. These are diagnostic thresholds, not literature standards.

## Actual PPO value-update trace

Call canonical train_resource_marl with its original loss, minibatch order, GAE and joint clipping. Bound collection to one terminal-complete rollout per seed21,22,23 (no policy training). At Adam.step, instrument original gradients and update the critic only; actor parameters are never changed. Record pre/post clip actor/critic/global norms, component losses, coverage, targets/predictions, critic Adam deltas, layerwise gradients and saturation. A detached tensor shadow Adam records the counterfactual joint optimizer step without changing the actor. First minibatch exactly matches canonical online PPO; later minibatches are explicitly fixed-actor diagnostics, not an unmodified PPO run. Additional shadow critic-only clipping comparisons use the same gradients and parameter path and cannot establish end-to-end causality by themselves.

## Interpretation and scope

Compare A canonical online critic, B warm-start fit (and labeled fresh control), C time regression on identical held-out transitions. Distinguish exact-MC supervision, sample/optimizer exposure, initialization and joint clipping. Propose exactly one next pilot after results. Full30k remains No-Go irrespective of these diagnostic results; future reconsideration needs stable critic gains, held-out stochastic J improvement and a signal of nonuniform pair learning. No actor/reward/environment edits.
