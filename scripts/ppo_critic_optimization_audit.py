"""Trace the actual canonical trainer while keeping actor weights immutable.

The original loss/backward/joint clipping execute unchanged. Adam.step removes
actor gradients only for the real parameter update. Detached shadow tensors
measure the original joint Adam and separate-clipping counterfactual deltas.
Later minibatches are fixed-actor diagnostics, not unmodified online PPO.
"""
from critic_diag_common import *
import argparse,inspect,math,time
import train_yc_marl as tr

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--seed',type=int,required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    reference=model_from_checkpoint(a.checkpoint);actor_hash=state_hash(reference)
    original_clip=torch.nn.utils.clip_grad_norm_;original_adam=torch.optim.Adam;original_gae=tr.compute_gae
    rows=[];raw_grads={};shadow_joint=None;shadow_separate=None;coverage=[];buffer={}

    def find_frame():
        frame=inspect.currentframe().f_back
        while frame and frame.f_code is not tr.train_resource_marl.__code__:frame=frame.f_back
        assert frame is not None;return frame.f_locals

    def gae(rewards,values,next_values,dones,gamma,lam):
        loc=find_frame();result=original_gae(rewards,values,next_values,dones,gamma,lam)
        buffer.update(obs=np.asarray(loc['global_obs_buf']),reward=rewards,value=values,next_value=next_values,done=dones,target=result[1],role=np.asarray(loc['roles_buf']))
        return result

    def clip(params,max_norm,*args,**kwargs):
        loc=find_frame();m=loc['model'];ps=list(params);named=list(m.named_parameters());raw_grads.clear()
        for name,p in named:
            if p.grad is not None:raw_grads[name]=p.grad.detach().clone()
        actors=[p for n,p in named if n.startswith(('storage_actor.','yc_actor.'))];critics=list(m.critic.parameters())
        row=dict(step=len(rows)+1,ids=loc['ids'].tolist(),rollout_samples=int(loc['n']),
            storage_samples=int(sum(loc['roles_buf'][int(i)]==0 for i in loc['ids'])),yc_samples=int(sum(loc['roles_buf'][int(i)]==1 for i in loc['ids'])),
            pg_loss=float(loc['pg_loss'].detach()),value_half_mse=float(loc['v_loss'].detach()),weighted_value_loss=float((loc['config'].vf_coef*loc['v_loss']).detach()),entropy_bonus=float(loc['entropy_bonus'].detach()),total_loss=float(loc['loss'].detach()),
            actor_grad_before=norm(actors,True),critic_grad_before=norm(critics,True),combined_grad_before=norm(ps,True),
            critic_encoder_grad_before=norm(list(m.critic.encoder.parameters()),True),critic_head_grad_before=norm(list(m.critic.head.parameters()),True),
            target_mean=float(loc['ret_t'][loc['mb']].mean()),target_std=float(loc['ret_t'][loc['mb']].std(unbiased=False)),
            prediction_mean=float(loc['value'].detach().mean()),prediction_std=float(loc['value'].detach().std(unbiased=False)))
        with torch.no_grad():
            z,_,_=m.critic.encoder(loc['global_obs_t'][loc['mb']]);row['fusion_saturation_fraction']=float((z.abs()>.99).float().mean())
        ret=original_clip(ps,max_norm,*args,**kwargs)
        row.update(actor_grad_after=norm(actors,True),critic_grad_after=norm(critics,True),combined_grad_after=norm(ps,True))
        row['clip_scale']=row['combined_grad_after']/row['combined_grad_before'];row['critic_independent_clip_scale']=min(1.,.5/(row['critic_grad_before']+1e-6))
        rows.append(row);coverage.extend(row['ids']);return ret

    class AuditedAdam(original_adam):
        def step(self,closure=None):
            nonlocal shadow_joint,shadow_separate
            loc=find_frame();m=loc['model'];named=[(n,p) for n,p in m.named_parameters() if not n.startswith('yc_q_critic.')]
            cp=[(n,p) for n,p in named if n.startswith('critic.')];row=rows[-1]
            if shadow_joint is None:
                shadow_joint=([p.detach().clone().requires_grad_() for _,p in named],None)
                shadow_joint=(shadow_joint[0],original_adam(shadow_joint[0],lr=loc['config'].learning_rate))
                shadow_separate=([p.detach().clone().requires_grad_() for _,p in cp],None)
                shadow_separate=(shadow_separate[0],original_adam(shadow_separate[0],lr=loc['config'].learning_rate))
            prev={n:p.detach().clone() for n,p in named}
            jps,jopt=shadow_joint;sps,sopt=shadow_separate
            with torch.no_grad():
                for (n,p),q in zip(named,jps):q.copy_(p);q.grad=None if p.grad is None else p.grad.clone()
                for (n,p),q in zip(cp,sps):q.copy_(p);q.grad=raw_grads[n]*row['critic_independent_clip_scale']
            jopt.step();sopt.step()
            jd=[q.detach()-prev[n] for (n,p),q in zip(named,jps) if n.startswith('critic.')]
            sd=[q.detach()-prev[n] for (n,p),q in zip(cp,sps)]
            row['shadow_joint_critic_step_norm']=norm(jd);row['shadow_separate_clip_step_norm']=norm(sd)
            row['shadow_delta_relative_difference']=norm([x-y for x,y in zip(jd,sd)])/(norm(jd)+1e-30)
            row['shadow_actor_step_norm']=norm([q.detach()-prev[n] for (n,p),q in zip(named,jps) if not n.startswith('critic.')])
            # Do not mutate actor weights, including in a cloned policy.
            actor_grads=[]
            for n,p in named:
                if not n.startswith('critic.'):actor_grads.append((p,p.grad));p.grad=None
            super().step(closure)
            for p,g in actor_grads:p.grad=g
            assert state_hash(m)==actor_hash
            actual=[p.detach()-prev[n] for n,p in cp]
            row['critic_parameter_update_norm']=norm(actual)
            row['critic_relative_update_norm']=norm(actual)/norm([prev[n] for n,p in cp])
            row['actual_vs_joint_shadow_max_abs']=max(float((d-j).abs().max()) for d,j in zip(actual,jd))
            row['encoder_update_norm']=norm([p.detach()-prev[n] for n,p in cp if n.startswith('critic.encoder.')])
            row['head_update_norm']=norm([p.detach()-prev[n] for n,p in cp if n.startswith('critic.head.')])
            row['actor_actual_update_norm']=norm([p.detach()-prev[n] for n,p in named if not n.startswith('critic.')])
            dump(out/'minibatches.json',rows)

    tr.compute_gae=gae;torch.nn.utils.clip_grad_norm_=clip;torch.optim.Adam=AuditedAdam
    config=tr.ResourcePPOConfig(total_steps=1,seed=a.seed,episode_complete_rollout=True)
    m,updates=tr.train_resource_marl(config,out/'native',Path(a.checkpoint))
    torch.nn.utils.clip_grad_norm_=original_clip;torch.optim.Adam=original_adam;tr.compute_gae=original_gae
    assert state_hash(m)==actor_hash and buffer['done'][-1]==1
    n=len(buffer['reward']);batches=math.ceil(n/config.minibatch_size);coverage_rows=[]
    for epoch in range(config.update_epochs):
        allids=[i for r in rows[epoch*batches:(epoch+1)*batches] for i in r['ids']];counts=np.bincount(allids,minlength=n)
        assert np.all(counts==1);coverage_rows.append(dict(epoch=epoch+1,unique_transitions=len(set(allids)),total=len(allids),min_count=int(counts.min()),max_count=int(counts.max())))
    np.savez_compressed(out/'rollout.npz',**buffer)
    cuts=[]
    for start in range(0,n,512):
        sl=slice(start,min(start+512,n));adv,ret=original_gae(buffer['reward'][sl],buffer['value'][sl],buffer['next_value'][sl],buffer['done'][sl],1.,1.)
        diff=ret-buffer['target'][sl];cuts.append(dict(start=start,count=len(ret),terminal=bool(buffer['done'][sl][-1]),target_bias_vs_full_mc=float(diff.mean()),target_rmse_vs_full_mc=float(np.sqrt(np.mean(diff**2)))))
    dump(out/'summary.json',dict(**provenance(a.checkpoint,a.seed,n,len(rows),'canonical sampling, fixed actor during actual trainer update; no policy update'),
        actor_hash_before=actor_hash,actor_hash_after=state_hash(m),actor_parameter_updates=0,ppo_updates=len(updates),episodes=1,
        scenario_seeds=[int(r['seed']) for r in __import__('csv').DictReader((out/'native/resource_marl_train_episodes.csv').open())],
        coverage=coverage_rows,cutoff_target_probe=cuts,value_target=metrics(buffer['value'],buffer['target']),
        after_value_fit=metrics(predict(m.critic,torch.tensor(buffer['obs'])),buffer['target'])))
    print(json.dumps(dict(seed=a.seed,decisions=n,critic_optimizer_steps=len(rows),actor_updates=0)),flush=True)

if __name__=='__main__':main()
