"""Reuse audited data, reproduce regression, replay only missing metadata."""
from critic_diag_common import *
import argparse,time
from yc_marl_env import STORAGE_AGENT
import evaluate_yc_policies as evaluator

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--old',required=True);ap.add_argument('--out',required=True);ap.add_argument('--checkpoint',required=True)
    a=ap.parse_args();old=Path(a.old);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((old/'OUTPUT_MANIFEST.json').read_text());checked=[]
    for item in manifest:
        p=old/item['path'];assert p.exists() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],p
        checked.append(item['path'])
    dump(out/'old_artifact_integrity.json',dict(checked=len(checked),paths=checked,manifest_sha256=sha(old/'OUTPUT_MANIFEST.json')))
    d,eps,seeds,train=load_data(old);d={k:d[k] for k in d.files}
    x=torch.tensor(d['obs']);m=model_from_checkpoint(a.checkpoint);h=state_hash(m);p=predict(m.critic,x)
    # Batched and single inference may differ by a few fp32 ULPs.
    assert np.max(abs(p-d['values']))<1e-4
    design=np.column_stack([np.ones(len(x)),d['obs'][:,2]])
    beta=np.linalg.lstsq(design[train],d['mc'][train],rcond=None)[0];reg=design@beta
    calibration=[]
    for lo in range(0,541,60):
        take=(~train)&(d['obs'][:,2]*480>=lo)&(d['obs'][:,2]*480<lo+60)
        if take.any():calibration.append(dict(minute_start=lo,online=metrics(d['values'][take],d['mc'][take]),regression=metrics(reg[take],d['mc'][take])))
    result=dict(**provenance(a.checkpoint,None,0,0,'reuse exact canonical stochastic evaluation'),
        coefficients=beta.tolist(),feature='intercept and obs[2]=min(simulation_time/480,2)',procedure='numpy.linalg.lstsq(rcond=None), scenario split, no regularization',
        train_transitions=int(train.sum()),validation_transitions=int((~train).sum()),
        online_full=metrics(d['values'],d['mc']),online_validation=metrics(d['values'][~train],d['mc'][~train]),
        regression_validation=metrics(reg[~train],d['mc'][~train]),regression_train=metrics(reg[train],d['mc'][train]),
        batch_prediction_max_difference=float(np.max(abs(p-d['values']))),time_binned_calibration=calibration)
    dump(out/'baseline.json',result);print(json.dumps(result['regression_validation']),flush=True)
    # Replay only the 30 canonical episodes because original rewards, exact sim
    # times, terminal flags, roles and indices were not retained in old NPZ.
    original=evaluator.ResourceMARLYardEnv;records=[];episode_index=0
    class TracedEnv(original):
        def step(self,action):
            g=self.critic_observation();role=0 if self.active_agent()==STORAGE_AGENT else 1
            row=dict(scenario=self.seed,policy_seed=eps[episode_index]['policy_seed'],episode_id=episode_index,
                simulation_time=float(self.sim.now),actor_role=role,decision_index=sum(r['episode_id']==episode_index for r in records),action=int(action))
            idx=len(records);assert np.array_equal(g,d['obs'][idx]),(episode_index,idx)
            ret=super().step(action);row.update(reward=float(ret[1]),terminal=bool(ret[2]));records.append(row);return ret
    evaluator.ResourceMARLYardEnv=TracedEnv;started=time.monotonic()
    for episode_index,e in enumerate(eps):
        r=evaluator.run_episode(e['seed'],'marl',m,stochastic=True,policy_seed=e['policy_seed'])
        assert abs(r['objective_proxy']-e['objective_proxy'])<1e-10
        print(json.dumps(dict(replayed=episode_index+1,decisions=len(records),seconds=time.monotonic()-started)),flush=True)
    assert state_hash(m)==h and len(records)==len(d['mc'])
    rewards=np.array([r['reward'] for r in records]);done=np.array([r['terminal'] for r in records]);mc=np.zeros(len(records));acc=0.
    for i in reversed(range(len(records))):
        if done[i]:acc=0.
        acc=rewards[i]+acc;mc[i]=acc
    assert np.array_equal(mc,d['mc']);assert sum(done)==30
    arrays={k:np.array([r[k] for r in records]) for k in records[0]}
    np.savez_compressed(out/'fixed_policy_dataset.npz',obs=d['obs'],mc=mc,current_critic_prediction=d['values'],**arrays)
    dump(out/'dataset_manifest.json',dict(**provenance(a.checkpoint,None,len(records),0,'stochastic torch.multinomial with scenario*100+repeat; actor fixed'),
        scenarios=list(range(601,611)),policy_seeds=[e['policy_seed'] for e in eps],episodes=30,
        source_dataset_sha256=sha(old/'validation/canonical/critic_dataset.npz'),dataset_sha256=sha(out/'fixed_policy_dataset.npz'),
        replay_reason='required metadata absent from original NPZ',observations_bitwise_match=True,mc_bitwise_match=True,
        actor_hash_before=h,actor_hash_after=state_hash(m),actor_parameter_updates=0))

if __name__=='__main__':main()
