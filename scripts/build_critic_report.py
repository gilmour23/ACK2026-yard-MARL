"""Render the ten requested sections from completed numeric evidence."""
from critic_diag_common import *
import argparse

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in r)+' |' for r in rows])
def f(x,n=4):return f'{x:.{n}f}'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);d=json.loads((out/'analysis.json').read_text())
    b=d['baseline'];fits=d['fits'];old=d['pilot'];pp=d['ppo'];baseurl=f'https://github.com/gilmour23/ACK2026-yard-MARL/blob/{BASE}'
    warm=[fits[f'warm{s}']['final']['validation'] for s in [21,22,23]]
    report=f'''# ACK2026 fixed-policy critic diagnostic — 2026-09-22

**판정: 현재 PermutationInvariantCritic은 주어진 observation에서 coarse MC return을 충분히 표현할 수 있다.** 기존 가중치와 같은 구조를 유지한 critic-only 학습의 held-out EV는 {min(r['ev'] for r in warm):.3f}–{max(r['ev'] for r in warm):.3f}였다. 현재 online critic의 같은 표본 EV는 {b['online_validation']['ev']:.8f}다. 따라서 coarse value failure를 Target ETA 정보 부족만으로 설명할 수 없다.

현재 증거는 **critic 업데이트 횟수·target 품질을 포함한 online 학습 과정의 underfitting**을 우선 지지한다. joint clipping이 critic 학습을 막는다는 가설은 이번 수치로 지지되지 않았다. Fine Target/pair credit의 해결, policy objective 개선은 검증하지 않았다. **Full 30k는 계속 No-Go다.**

## 1. Latest canonical commit verification

기준 main commit: `{BASE}`. canonical checkpoint SHA-256: `{CHECKPOINT_SHA}`. 두 값 모두 실제 내용과 대조했다. GitHub의 24개 blob, root tree와 commit object를 검증해 로컬 작업 사본을 만들었다. 실험·instrumentation은 `experiment/critic-diagnostic`에만 commit했고 `src/`, main, actor 구조, reward, 환경은 수정하지 않았다.

{table(['확인 항목','판정','실제 근거'],[
 ['ResourcePPOConfig','확인','src/train_yc_marl.py:294 / 298 / 299 → 512 / 2 / 256'],
 ['SinglePPOConfig','확인','src/train_yc_single.py:85 / 86 → 512 / 2 / 256'],
 ['evaluate() stochastic 기본값','확인','src/evaluate_yc_policies.py:41 → stochastic=True'],
 ['Full runner 30k 차단','확인','scripts/run_v4_experiments.py:17; 인자 없는 실행이 No-Go RuntimeError로 종료'],
 ['두 trainer 직접 실행 차단','확인','train_yc_marl.py:587; train_yc_single.py:167; 두 실행 모두 학습 전에 종료'],
 ['CURRENT_MODEL_README','확인','docs/CURRENT_MODEL_README.md: 완료된 6 run / 210 episode 중간 결과와 critic 진단 필요성 반영'],
 ['warm start 설명','확인','train_yc_marl.py:428의 state_dict load, :439의 새 Adam, :442의 step/update=0; :576은 저장만 수행'],
 ['GitHub Actions','통과','main 기준 commit: pytest 16 passed in 3.15s; 로컬 16 passed in 6.04s']])}

[main pytest 실행](https://github.com/gilmour23/ACK2026-yard-MARL/actions/runs/35697019378), [진단 코드 84be699 pytest 실행](https://github.com/gilmour23/ACK2026-yard-MARL/actions/runs/35699368496): 후자는 16 passed in 4.42s. 로그를 evidence에 보존했다. 마지막 문서·집계 commit의 CI 상태는 최종 실행 manifest에 별도로 적는다.

하위 `run_episode()`는 아직 `stochastic=False`가 기본값이다(`src/evaluate_yc_policies.py:23`). 이번 replay에서는 `stochastic=True`를 명시했다. optimizer/RNG metadata 저장은 exact resume 지원을 의미하지 않는다. `init_checkpoint`는 새 optimizer와 새 RNG 흐름을 쓰는 weights-only warm start다. simulator 진행 상태도 복원하지 않는다.

## 2. 2k A/B pilot final summary

기존 산출물 155개를 OUTPUT_MANIFEST의 크기·SHA-256과 대조해 모두 일치함을 확인했다. 6개 training run과 210개 평가 episode 전체를 다시 실행하지 않았다. 누락된 metadata를 위해 canonical 30회만 replay한 범위는 3절에 구분했다. 아래는 완료된 결과다. 기존 실행은 Drive source snapshot에서 수행되었으므로 당시 존재하지 않았던 Git commit을 실행 이력으로 소급 기재하지 않는다. GitHub 기준과 env/network의 bytes가 같고, trainer 차이는 기본값·문서·실행 차단이며 실제 pilot의 명시적 설정은 이미 512/2/256이었다.

{table(['Run','Decisions','완료 episode','미완료 tail','PPO update','Adam step','Storage / YC','학습 proactive'],[[label,t['actual_decisions'],t['episodes'],t['unfinished_episode_decisions'],t['updates'],t['optimizer_steps'],f"{t['storage_samples']} / {t['yc_samples']}",t['proactive_samples']] for label in ['A21','A22','A23','B21','B22','B23'] for t in [old[label]['training']]])}

A는 총 6,246 decisions, B는 6,624 decisions로 B가 약 6.05% 더 수집했다. B의 episode는 모두 terminal/drain까지 수집됐고 중간 tail 유실은 없었다. A의 tail도 update에는 포함됐지만 bootstrap target을 사용하며, 마지막 진행 중 episode를 exact resume할 수는 없다.

평가: scenarios 601–610, scenario별 stochastic sampling 3회, policy seed=`scenario*100+repeat`. canonical 30회 + 6 policy × 30회 = 210회, 총 248,551 decisions. 이 bank와 301–305는 최종 unseen test set이 아니다.

{table(['정책','Truck delay','Storage delay','Reactive rehandling','Proactive','Extra YC min/retrieval','J'],[[label,f(v['mean_truck_completion_delay']),f(v['mean_storage_completion_delay']),f(v['rehandling_moves'],2),f(v['proactive_moves'],2),f(v['extra_yc_minutes_per_retrieval']),f(v['objective_proxy'])] for label in old for v in [old[label]['validation']]])}

{table(['정책','MC EV','MC RMSE','P(Proactive)','Sampled pro rate','Op entropy','Conditional pair H(norm)'],[[label,f(v['mc_ev'],8),f(v['mc_rmse']),f(v['p_proactive_all_yc']),f(v['sampled_proactive_rate']),f(v['operation_entropy_all_yc']),f(v['pairs']['pair_entropy_norm'],7)] for label in old for v in [old[label]['validation']]])}

B는 3 seeds 모두 EV/RMSE가 개선됐지만 J는 seed21 +1.105%, seed22 +4.746% 악화, seed23 −0.781% 개선이었다. seed 평균 J는 A 11.43877 → B 11.63535, **1.719% 악화**다. Pair distribution은 계속 거의 uniform이다. canonical의 conditional pair H=0.999912, ESS/K=0.999127, TV(uniform)=0.011827이며, uniform 대비 expected Target ETA lead 차이는 +0.02035분에 불과했다. Flat greedy proactive=0은 credit failure의 독립 근거로 사용하지 않았다.

3 training seeds와 10 scenarios의 진단 결과다. 210회를 독립 scenario 210개처럼 취급하지 않는다. 4k/6k extension이나 30k 학습을 실행하지 않았다.

## 3. Critic representational sufficiency test

canonical 12k actor의 기존 fixed-policy data만 사용했다. 총 **35,370 transitions / 30 terminal-complete episodes**. Train=601–607의 24,913 transitions / 21 episodes, validation=608–610의 10,457 transitions / 9 episodes다. Transition random split은 사용하지 않았다. 세 validation scenario는 이미 관찰한 진단 bank이므로 최종 일반화 성능 주장에 쓰지 않는다.

기존 NPZ에 원 reward·정확한 simulation time·role·terminal·decision index가 없어서 canonical 30 episodes만 동일 stochastic seed로 replay했다. `obs`와 exact undiscounted MC return은 원본과 bitwise 일치했고 J도 일치했다. actor hash는 전후 `{d['dataset']['actor_hash_before']}`로 같았다. `fixed_policy_dataset.npz`에는 obs, scenario, policy_seed, episode_id, simulation_time, reward, terminal, mc, current_critic_prediction, actor_role(0=Storage/1=YC), decision_index, action을 저장했다. Return은 마지막 terminal부터 reward를 float64로 역누적한 값이며 bootstrap을 포함하지 않는다.

현재 critic observation은 440차원이다: global context 8 + block 4×8 + stack 100×4. `global_observation()`의 obs[2]에 `min(sim.now/480, 2)`가 이미 들어 있다(`src/yc_marl_env.py:1163`). Context/slot/block encoding, mean/max pooling, global fusion Tanh, linear value head가 현재 구조다(`src/v4_networks.py:10`, :26). Temporal signal을 입력받을 경로는 존재한다. 세부 Target ETA와 stack order aliasing은 별도 한계이며 coarse return을 표현하지 못한다는 결론으로 연결할 수 없다.

시간 회귀를 정확히 재현했다. 입력은 **상수 1과 obs[2] 두 열뿐**, train scenario에서 `numpy.linalg.lstsq(X, MC, rcond=None)`로 OLS를 수행했다. 정규화·정규항·validation fitting·다항식은 없다.

`V_hat = -12.27720743632284 + 12.108310683201031 * obs[2]`

Held-out EV={b['regression_validation']['ev']:.14f}, RMSE={b['regression_validation']['rmse']:.8f}, R²={b['regression_validation']['r2']:.6f}, bias={b['regression_validation']['bias']:.6f}. **이전 EV≈0.958은 재현됐다.** EV는 평균 오차를 제거하기 때문에 bias 약 −1.096과 RMSE를 함께 봐야 한다. 이 회귀는 진단용이며 critic/reward/Target score로 설치하지 않았다.

현재 online critic의 전체 canonical 데이터 prediction std={b['online_full']['prediction_std']:.8f}, range=[{b['online_full']['prediction_min']:.8f}, {b['online_full']['prediction_max']:.8f}]로 거의 상수다. 동일 held-out 표본에서 prediction mean={b['online_validation']['prediction_mean']:.6f}, std={b['online_validation']['prediction_std']:.8f}; target mean={b['online_validation']['target_mean']:.6f}, std={b['online_validation']['target_std']:.6f}다.

## 4. Critic-only training result

**Primary**: canonical critic weights에서 시작, 구조 변경 없음, Adam lr=3e-4, batch256, max_grad_norm=0.5, loss=0.25×MSE. 이는 online의 `vf_coef=0.5 × half-MSE`와 같다. 각 seed 20 full epochs, 1,960 optimizer steps, 마지막 epoch를 endpoint로 사전 지정했다. Validation이 가장 좋았던 checkpoint를 고르지 않았다. Actor는 optimizer에 포함되지 않았고 가중치는 한 번도 업데이트하지 않았다.

**Fresh control**: critic의 초기화만 새로 했다. 같은 구조·data·batch order·optimizer·budget을 유지했다. 이는 architecture 비교가 아니다. 모든 fit의 새 환경 decisions는 0이며 기존 transition을 재사용했다.

아래 A/B/C는 **동일한 held-out 10,457 transitions**에서 계산했다. A는 canonical online critic이며, 앞 절의 B episode-complete 2k critic과 혼동하면 안 된다.

{table(['모델','EV','RMSE','Prediction variance','Target variance','Prediction mean'],[[r['model'],f(r['ev'],8),f(r['rmse']),f(r['prediction_variance'],8),f(r['target_variance']),f(r['prediction_mean'])] for r in d['comparison']])}

{table(['Fit','Train EV','Validation EV','Train RMSE','Validation RMSE','Clip frequency','평균 grad 전 / 후','평균 parameter Δ L2'],[[label,f(r['final']['train']['ev']),f(r['final']['validation']['ev']),f(r['final']['train']['rmse']),f(r['final']['validation']['rmse']),f(r['gradient_statistics']['clipped']['mean']),f"{r['gradient_statistics']['grad_before']['mean']:.4f} / {r['gradient_statistics']['grad_after']['mean']:.4f}",f(r['gradient_statistics']['parameter_update_norm']['mean'],6)] for label,r in fits.items()])}

전체 step별 value loss, target/prediction mean·std, gradient norm 전후, clipping scale/frequency, layer별 gradient, parameter update norm은 각 `fits/*/optimizer_steps.csv`에 있다. `learning_curve.json`에는 epoch별 full train/validation EV·RMSE·분포와 fusion saturation이 있다. 실제 데이터 coverage는 epoch마다 train transition 각 1회, validation 0회였다. Raw target 분포는 train/validation별로 고정하며 optimizer batch별 통계도 보존했다.

초기 예산을 별도로 계측했다. 이 측정은 첫 결과를 본 뒤 추가했으며 primary endpoint를 바꾸지 않았다. 각 seed의 같은 first-epoch batch 순서·loss·lr를 재현했고 step98 EV가 원래 first-epoch 기록과 일치했다.

{table(['Adam steps','EV seed21','EV seed22','EV seed23'],[[step,*[f(next(r['validation']['ev'] for r in d['budgets'][str(s)]['curve'] if r['step']==step),6) for s in [21,22,23]]] for step in [10,20,40,60,98]])}

Actor와 optimizer를 분리한 offline fit에서도 **20 steps만 주면 EV≈0.024**였다. 따라서 actor를 제외했다는 사실만으로 개선됐다고 해석할 수 없다. 업데이트가 누적되면서 같은 구조가 return 차이를 학습했다. 다만 이 곡선은 21 episodes를 섞은 offline 데이터다. Online의 1–2 episodes와 데이터 다양성·target 정확도·재사용량이 다르므로, online epoch 증가만으로 결과가 그대로 재현된다고 단정할 수 없다.

20-epoch validation EV가 초기 최고점보다 낮아지는 경우도 있다. 충분한 표현력과 학습 가능성은 확인됐지만 최적 epoch나 일반화 안정성을 확정한 실험은 아니다. 사용자의 EV≥0.20 / ≥0.50 engineering diagnostic gate는 모든 primary seed에서 통과했다. 문헌상의 절대 기준으로 해석하지 않는다.

Held-out time-binned calibration (분):

{table(['시각 구간 시작','n','MC mean','Online V mean','Warm21 mean','Warm22 mean','Warm23 mean'],[[r['minute_start'],r['n'],f(r['target_mean']),f(r['online_mean']),*[f(r['warm_fit_means'][f'warm{s}']) for s in [21,22,23]]] for r in d['time_calibration']])}

## 5. Online PPO critic optimization audit

`train_resource_marl()`의 실제 함수와 loss/backward/joint clipping을 실행했다. `Adam.step`에서 actor gradient를 실제 update에서 제외해 actor 가중치를 불변으로 유지했다. Critic은 원래 joint-clipped gradient로 실제 업데이트했다. 별도의 detached tensor shadow Adam으로 원래 joint update와 critic-only clipping의 delta를 비교했다. **첫 minibatch는 canonical PPO 경로와 같지만 이후 minibatch는 actor가 고정된 진단 경로다.** 이를 무수정 PPO 재학습 결과로 부르지 않는다.

Seeds21/22/23에서 첫 terminal-complete rollout 하나씩만 수집했다. actor observations/masks/gradient 기록이 기존 산출물에 없어 필요한 첫 rollout을 재현한 것이며 2k A/B 전체 재실행이 아니다. Actor parameter updates=0. Q critic은 비활성화했다. Reward, actor, critic, hyperparameters는 그대로다.

{table(['Seed','Scenario','Decisions','Critic Adam steps','PPO collect/update','종료 후 rollout EV'],[[s,r['scenario_seeds'][0],r['actual_environment_decisions'],r['optimizer_updates'],r['ppo_updates'],f(r['after_value_fit']['ev'],7)] for s,r in pp.items()])}

`train_yc_marl.py:531`은 epoch마다 permutation 하나를 만들고 :534가 중복 없이 slice한다. Value loss는 모든 minibatch sample을 사용한다(:553). 각 epoch에서 모든 transition을 정확히 1회 사용했고, Storage/YC 수 및 ids를 `ppo_trace/seed*/minibatches.json`에 남겼다. Role별 actor PG mean을 다시 평균하는 반면 critic은 transition 평균이다. 이 차이는 존재하지만 critic sample 누락 버그는 없었다.

Minibatch 평균 (PG는 부호 있는 mean loss이며 gradient norm과 다른 지표다):

{table(['Seed','PG loss','0.5×value loss','Entropy bonus','Actor grad L2','Critic grad L2','Joint clip scale','Critic parameter Δ L2'],[[s,*[f(r['minibatch_statistics'][k]['mean'],6) for k in ['pg_loss','weighted_value_loss','entropy_bonus','actor_grad_before','critic_grad_before','clip_scale','critic_parameter_update_norm']]] for s,r in pp.items()])}

총 loss는 `PG + 0.25*MSE − entropy_bonus`이고, shared Adam은 parameter별 moment를 관리한다. Actor와 critic은 이 MARL 모델에서 parameter를 공유하지 않는다. 하나의 optimizer라는 사실만으로 critic gradient가 actor gradient와 더해져 취소되는 구조는 아니다. 상호작용 경로는 global norm clipping과 policy/data/target 변화다.

세 trace 모두 joint norm clipping이 모든 minibatch에서 작동했고, combined norm은 clipping 후 약 0.5였다. 같은 scale이 actor와 critic에 적용되지만 raw norm은 critic 쪽이 지배했다. Seed21의 첫 minibatch에서 actor=0.147874, critic=17.745916, combined=17.746532, scale=0.0281745였다. Critic encoder raw gradient=0.016872에 비해 head=17.745908이었다. 그럼에도 Adam의 encoder update L2=0.099082, head update L2=0.003407로 실제 parameter 이동이 있었다. Layer별 parameter 수가 다르므로 L2 크기 자체를 학습 기여도로 해석하지 않는다.

Critic-only clipping shadow와 joint shadow의 critic Adam delta 상대 차이는 seed별 평균 **{pp['21']['minibatch_statistics']['shadow_delta_relative_difference']['mean']:.8f}, {pp['22']['minibatch_statistics']['shadow_delta_relative_difference']['mean']:.8f}, {pp['23']['minibatch_statistics']['shadow_delta_relative_difference']['mean']:.8f}**, 전체 최대 {max(r['minibatch_statistics']['shadow_delta_relative_difference']['max'] for r in pp.values()):.8f}였다. 최대도 약 0.067% 이하다. 실제 critic update와 joint shadow는 최대 절대 차이 0으로 일치했다. **Joint clipping을 분리하면 critic failure가 해결된다는 근거는 얻지 못했다.** Actor 학습에 대한 영향이나 장기간 Adam history 효과까지 배제한 것은 아니다.

Canonical fusion Tanh는 초기 거의 완전히 포화돼 있고 value std가 매우 작았다. 첫 10 updates는 주로 상수 수준을 옮겼으며 rollout EV는 약 0.001–0.002에 머물렀다. 더 많은 exact-MC 업데이트에서는 같은 가중치에서 feature response가 회복됐다. 영구적인 dead network나 forward 경로 단절이라는 해석과는 맞지 않는다.

동일 완결 rollout의 reward를 512개씩 잘라 기존 `compute_gae(gamma=1, lambda=1)`에 넣은 보조 계산에서도 bootstrap 오차를 확인했다. Seed21의 첫 두 비terminal 구간 target은 terminal-complete MC보다 평균 −2.82546, −6.96239 낮았다. 마지막 terminal 구간 오차는 0이었다. 이것은 실제 `rollout_ready`의 role threshold까지 재현한 A/B 실험이 아니라 **같은 trajectory에서 boundary target만 비교한 수치 probe**다. 정확한 rollout A/B 성능 판단은 2절의 기존 결과를 사용한다.

## 6. Root-cause ranking

1. **Critic 최적화 노출량 부족을 포함한 online underfitting — 가장 강한 증거.** 같은 구조·초기 가중치·lr에서도 20 steps EV≈0.024, 60 steps 0.547–0.706, 98 steps 0.782–0.808, 충분한 fit 후 약 0.94다. Actor를 제외하는 것만으로 초반 failure가 즉시 사라지지 않는다. 적은 value update budget과 데이터·target 조건이 함께 중요하다.
2. **부정확한 bootstrap target과 rollout boundary의 피드백 — 직접적인 target 오차 근거.** 낮은 품질의 V를 tail에 넣으면 남은 비용을 크게 잘못 추정한다. 기존 B가 세 seeds에서 EV/RMSE를 개선한 결과와 일치하지만, B의 20 Adam steps만으로는 개선 폭이 작았다. Boundary만 해결하면 충분하다는 근거는 없다.
3. **기존 critic의 Tanh saturation과 느린 초기 feature 회복 — 관찰된 학습 상태.** 초기 예측 분산과 encoder raw gradient가 작다. 그러나 같은 weights에서 회복되므로 구조적 표현 불가능이나 초기화 교체 필요성은 입증되지 않았다. 이 상태를 만든 과거 학습 이력의 세부 원인은 현재 checkpoint만으로 복원할 수 없다.
4. **Online 데이터 다양성·분포 변화·actor interaction — 아직 분리되지 않은 후보.** Offline은 고정 정책의 21 episodes를 반복 학습한다. 이번 진단은 actor를 고정했으므로 policy nonstationarity의 인과적 기여를 측정하지 않았다. Fresh optimizer warm start도 실제 경로지만 단독 원인으로 확정할 수 없다.
5. **Fine Target ETA / stack-order observation aliasing — 세부 credit의 별도 제한.** 과거 재현 artifact의 무결성과 해당 env 코드 동일성을 확인했다. Fine state 구분 한계는 남지만 time baseline 및 same-architecture fit 성공 때문에 coarse EV≈0의 주원인으로 삼을 수 없다. 높은 pooled EV가 relocation의 marginal advantage 표현력을 보장하지도 않는다.
6. **Joint clipping / shared optimizer / minibatch 누락 — 현재 critic 실패의 우선 원인으로 지지되지 않음.** Coverage는 정확했고 actor/critic parameter 공유가 없으며 separate-clipping shadow 차이도 작았다. 큰 gradient norm 하나로 clipping failure라고 결론내리지 않는다.

Pair policy가 uniform인 이유 중 어느 비중이 critic, actor representation, exploration, 작은 action 간 return 차이에 해당하는지는 이번 실험으로 분해하지 않았다. Actor를 고정했으므로 pair learning이나 stochastic objective를 개선했다는 주장을 하지 않는다.

## 7. Exactly one minimal next pilot

**변경 하나: episode-complete PPO에서 critic의 update budget만 늘린다.** 기존 2 actor+critic epochs는 유지하고, 같은 rollout/target으로 critic-only epochs 18회를 추가해 value의 총 노출을 20 epochs로 만든다. Actor epoch, learning rate, Adam 설정, gradient threshold, architecture, observation, reward, gamma/lambda, initialization은 유지한다. Optimizer separation, clipping separation, target normalization, LR 변경을 함께 묶지 않는다.

정확한 위치: `src/train_yc_marl.py`의 `train_resource_marl()`, 기존 minibatch update :555 뒤와 `update_idx += 1` 사이. 기존 main Adam을 재사용하고 추가 value step에는 actor gradient를 None으로 둔다. 기존 joint-update 경로를 바꾸지 않는다. 추가 sampler에는 독립 RNG를 사용해 기존 actor minibatch permutation RNG를 소비하지 않도록 기록한다. 이는 아직 구현하거나 실행하지 않은 제안이다.

이유: 같은 구조가 충분한 update에서는 학습하고, joint clipping 분리의 효과는 작았기 때문이다. 예상 변화는 initial saturation에서 더 빨리 벗어나 prediction variance와 coarse return fit이 증가하는 것이다. Online의 좁은 episode 분포에서 overfit될 가능성은 남는다.

제안된 검증: canonical 12k에서 동일 weights-only 시작, seeds21/22/23, control=value2 epochs / treatment=value20 epochs, **각 2k decision threshold 후 현재 episode terminal까지만**. 기존 run의 4k/6k 연장이 아니다. 환경·reward·actor PPO 설정은 모두 같게 둔다. Validation은 이번 기록에 쓰지 않은 701–710, repeats3, stochastic mode로 사전에 고정하고 최종 unseen test로 부르지 않는다.

사전 판정: 세 seed 모두 finite loss, 정확한 terminal/coverage, 추가 step에서 actor delta=0이 필수다. Critic mechanism 성공은 세 seed 모두 control 대비 held-out MC RMSE 감소, 최소 2 seeds EV≥0.50 및 나머지 EV≥0.20으로 정의한다. 이를 충족하지 못하면 budget 단독 설명을 기각/약화한다. J와 pair entropy/TV/expected pair features는 함께 기록하되 critic gate 통과를 policy 성공으로 대신하지 않는다. 해당 pilot도 30k 실행을 허용하지 않는다.

## 8. Full 30k Go / No-Go

**No-Go. 이번 진단 중 Full30k policy training=0, actor parameter updates=0.** Same-architecture representational gate 통과는 필요한 진단 결과이며 30k 허가가 아니다.

다음 재심사에 필요한 조건을 결과 전에 고정한다: (1) 동일 held-out stochastic 평가에서 위 critic gate를 안정적으로 만족, (2) J의 seed 평균이 control보다 최소 2% 낮고 최소 2/3 seeds 개선하며 scenario 단위 paired bootstrap(10,000 resamples, RNG seed20260922)의 J 차이 95% CI 상한이 0 미만일 것, (3) feasible-pair conditional TV(uniform)≥0.05가 최소 2/3 seeds에서 나타나고, operation distribution을 유지한 conditional-uniform 평가보다 학습 pair policy의 stochastic J가 낮을 것, (4) probability/terminal/coverage/checkpoint lineage에 미해결 오류가 없을 것. 단순히 entropy가 줄었다는 사실이나 greedy proactive 비율로 (3)을 통과시키지 않는다. 이 조건 충족 후에도 별도 Go 판정 전까지 runner guard를 유지한다.

현재는 (1)의 offline 표현력만 확인했고 online 안정성, (2), (3)은 충족하지 못했다. 이번 actor가 고정된 fit을 근거로 J가 좋아졌다고 추정하지 않는다.

## 9. Remaining implementation issues

- `run_episode(stochastic=False)`와 `evaluate(stochastic=True)` 기본값 불일치. `scripts/run_reward_sensitivity.py:13`도 명시적 greedy 평가를 사용한다. 이번 진단은 명시적 stochastic이라 영향을 받지 않았다.
- `_load_model()`의 `strict=False`(`evaluate_yc_policies.py:20`)와 warm-start compatible loading은 누락 tensor를 허용할 수 있다. 이번 canonical checkpoint는 strict=True로 모두 확인했다.
- optimizer/RNG/step을 저장하지만 loader가 복원하지 않고 simulator state도 없다. exact resume을 지원한다고 표현하면 안 된다. 이번 실험은 optimizer fresh start를 명시했다.
- CLI 30k guard는 작동하지만 `ResourcePPOConfig.total_steps`와 `SinglePPOConfig.total_steps`는 30000이고 library trainer API 자체에는 30k gate가 없다. 인자 없는 두 trainer 실행은 차단돼 있다. 이번 모든 호출은 bounded explicit config다.
- Native trainer 로그에는 per-minibatch gradient/update·정확한 coverage·learning curve가 없다. 이번 실험 branch의 artifact에 이를 보완했다. main에는 instrumentation을 merge하지 않았다.
- 초기 98-step critic은 일부 held-out state에서 양의 value를 예측하기도 했다. Return은 비용의 음수이므로 EV와 함께 RMSE/bias/range/calibration을 확인해야 한다. 출력 clipping이나 새 normalization을 추가하지 않았다.
- dataset metadata replay의 첫 시도는 compressed NPZ를 transition마다 다시 읽는 audit-script 성능 문제로 첫 episode 완료 전에 중단했다. 데이터 캐싱만 수정해 replay를 완료했다. 이 중단 시도의 partial decision 수는 로그에 없고 완결 dataset/실험 결과에는 포함하지 않았다. 기존 2k 산출물은 손대지 않았다.

## 10. Reproducibility manifest

기준 코드 `{BASE}`; checkpoint `{CHECKPOINT_SHA}`. 진단 코드 도입 commit `38d8f1430d4f5a87f6115d2fb5461a647624e482`, 실제 PPO trace `42294464fe86cb14b31dae05d073f52f88cd780f`, metadata replay/early-budget code `84be6992e6ded9ca0b478152ee8dd43981fe9109`. 이후 집계·문서 commit은 manifest에 기록한다. fit script의 학습 코드는 이 commit들 사이에서 바뀌지 않았다. 각 result summary의 instrumentation_commit은 해당 파일 내용을 포함하는 commit이다.

{table(['작업','Scenario / training seed','실제 새 decisions','실제 critic Adam updates','평가 방식'],[
 ['기존 6-run 2k A/B','train21/22/23; 각 run manifest의 scenario',0,'추가 0; 보존된 116','601–610 ×3 stochastic'],
 ['canonical metadata replay','601–610 ×3 policy seeds',35370,0,'stochastic; 기존 trajectories와 일치'],
 ['6 critic-only fits','train601–607 / val608–610; fit seeds21/22/23',0,11760,'고정 stochastic-policy data의 deterministic V inference'],
 ['early-budget measurement','같은 split; seeds21/22/23',0,294,'동일 first epoch 재현'],
 ['actual PPO value trace','각 training seed21/22/23의 첫 scenario',3445,30,'canonical stochastic sampling; actor 가중치 고정']])}

PPO trace에는 위 실제 critic update 외에 두 shadow Adam 계산이 각각 30회 있다. Shadow는 detached tensor에만 적용하며 actor network parameter update는 아니다. 새로 완결한 replay/trace decisions는 총 38,815, 실제 critic optimizer steps는 12,084다. Offline fit의 transition presentation 수와 environment decisions는 구분했다. 원래 2k training decisions 12,870 및 evaluation 248,551은 보존된 과거 실행 수다.

파일: `analysis.json`, `REPRODUCIBILITY_MANIFEST.json`, `OUTPUT_MANIFEST.json`, `dataset/fixed_policy_dataset.npz`, `dataset/baseline.json`, `fits/*/{{critic_only.pt,optimizer_steps.csv,learning_curve.json,predictions.npz,summary.json}}`, `budget*.json`, `ppo_trace/seed*/{{minibatches.json,rollout.npz,summary.json}}`, workflow logs, protocol. 각 파일의 SHA-256은 OUTPUT_MANIFEST에 있다. Python {d['runtime']['python'].split()[0]}, torch {d['runtime']['torch']}, numpy {d['runtime']['numpy']}, CPU torch threads=1.

큰 artifact 및 보존한 A/B evidence: [Drive 감사 폴더](https://drive.google.com/drive/folders/19ZAep4vPuGat62k0W7krETW1DFdB9Ebf). 코드: [experiment/critic-diagnostic](https://github.com/gilmour23/ACK2026-yard-MARL/tree/experiment/critic-diagnostic). Full main merge는 하지 않았다.

재현 명령은 `REPRODUCE.md`를 따른다. 기본적으로 기존 dataset과 결과를 읽으며, trajectory 재생성과 optimizer fit은 명시적 명령일 때만 실행한다. 30k나 actor 학습 명령은 포함하지 않는다.
'''
    (out/'ACK2026_ASTRA_Critic_Diagnostic_20260922.md').write_text(report)
    print('Rendered ten-section report from completed evidence.')

if __name__=='__main__':main()
