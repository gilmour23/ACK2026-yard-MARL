"""Build auditable aggregate evidence and learning-curve figure, no training."""
from critic_diag_common import *
import argparse,csv,platform

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--old',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);old=Path(a.old)
    baseline=json.loads((out/'dataset/baseline.json').read_text());dataset=json.loads((out/'dataset/dataset_manifest.json').read_text())
    fits={};budgets={};ppo={};ppo_rows=[]
    for init in ['warm','fresh']:
        for seed in [21,22,23]:
            label=f'{init}{seed}';p=out/'fits'/label;d=json.loads((p/'summary.json').read_text());assert d['epochs']==20 and d['optimizer_updates']==1960
            assert d['actor_hash_before']==d['actor_hash_after']==dataset['actor_hash_before']
            steps=list(csv.DictReader((p/'optimizer_steps.csv').open()));assert len(steps)==1960
            d['gradient_statistics']={k:{'mean':float(np.mean([float(r[k]) for r in steps])),'min':float(min(float(r[k]) for r in steps)),'max':float(max(float(r[k]) for r in steps))} for k in ['grad_before','grad_after','clip_scale','clipped','parameter_update_norm','value_loss','encoder_grad_before','head_grad_before']}
            d['selected_step_statistics']=[r for r in steps if int(r['step']) in [1,10,20,40,60,98,1960]]
            fits[label]=d
    for seed in [21,22,23]:
        budgets[str(seed)]=json.loads((out/f'budget{seed}.json').read_text());assert budgets[str(seed)]['epoch1_reproduced']
        p=out/'ppo_trace'/f'seed{seed}';d=json.loads((p/'summary.json').read_text());rs=json.loads((p/'minibatches.json').read_text());assert len(rs)==10
        assert all(r['actor_actual_update_norm']==0 and r['actual_vs_joint_shadow_max_abs']==0 for r in rs)
        assert d['actor_hash_before']==d['actor_hash_after']==dataset['actor_hash_before']
        d['minibatch_statistics']={k:{'mean':float(np.mean([r[k] for r in rs])),'min':float(min(r[k] for r in rs)),'max':float(max(r[k] for r in rs))} for k in rs[0] if k not in ['ids']}
        ppo[str(seed)]=d;ppo_rows.extend(rs)
    labels=['canonical','A21','A22','A23','B21','B22','B23'];pilot={}
    for label in labels:
        v=json.loads((old/'validation'/label/'summary.json').read_text());e=json.loads((old/'validation'/label/'episodes.json').read_text());assert len(e)==30
        pilot[label]={'validation':v,'episode_count':len(e),'validation_scenarios':list(range(601,611)),'evaluation_mode':'stochastic','policy_seeds':[r['policy_seed'] for r in e]}
        if label!='canonical':pilot[label]['training']=json.loads((old/'runs'/label/'train_summary.json').read_text())
    comparisons=[{'model':'A canonical online critic',**baseline['online_validation']}]
    for label,d in fits.items():comparisons.append({'model':f'B {label}',**d['final']['validation']})
    comparisons.append({'model':'C time OLS',**baseline['regression_validation']})
    fixed=np.load(out/'dataset/fixed_policy_dataset.npz');assert len(fixed['obs'])==35370 and fixed['terminal'].sum()==30
    assert np.array_equal(fixed['obs'],np.load(old/'validation/canonical/critic_dataset.npz')['obs'])
    held=fixed['scenario']>=608;bins=[]
    pred={label:np.load(out/'fits'/label/'predictions.npz')['prediction'] for label in fits}
    for lo in range(0,541,60):
        take=held&(fixed['simulation_time']>=lo)&(fixed['simulation_time']<lo+60)
        if take.any():bins.append(dict(minute_start=lo,n=int(take.sum()),target_mean=float(fixed['mc'][take].mean()),online_mean=float(fixed['current_critic_prediction'][take].mean()),warm_fit_means={label:float(v[take].mean()) for label,v in pred.items() if label.startswith('warm')}))
    scenario_metrics=[]
    for seed in [608,609,610]:
        take=fixed['scenario']==seed
        scenario_metrics.append(dict(scenario=seed,online=metrics(fixed['current_critic_prediction'][take],fixed['mc'][take]),warm={label:metrics(v[take],fixed['mc'][take]) for label,v in pred.items() if label.startswith('warm')}))
    summary=dict(baseline=baseline,dataset=dataset,fits=fits,budgets=budgets,ppo=ppo,pilot=pilot,comparison=comparisons,time_calibration=bins,scenario_metrics=scenario_metrics,
        totals=dict(preserved_training_decisions=12870,preserved_evaluation_episodes=210,preserved_evaluation_transitions=248551,
            metadata_replay_decisions=35370,ppo_trace_decisions=sum(d['actual_environment_decisions'] for d in ppo.values()),
            offline_fit_optimizer_steps=sum(d['optimizer_updates'] for d in fits.values()),budget_replay_optimizer_steps=sum(d['optimizer_updates'] for d in budgets.values()),
            ppo_trace_critic_optimizer_steps=sum(d['optimizer_updates'] for d in ppo.values()),actor_parameter_updates=0),
        runtime=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),torch_threads=torch.get_num_threads()))
    dump(out/'analysis.json',summary)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(15,4.2));colors=['#2367a0','#d46b35','#28816b']
    for seed,color in zip([21,22,23],colors):
        r=budgets[str(seed)]['curve'];ax[0].plot([x['step'] for x in r],[x['validation']['ev'] for x in r],'-o',color=color,label=f'Seed {seed}')
        c=json.loads((out/'fits'/f'warm{seed}'/'learning_curve.json').read_text());ax[1].plot([x['optimizer_step'] for x in c],[x['validation']['ev'] for x in c],color=color)
        ax[2].plot([x['optimizer_step'] for x in c],[x['validation']['rmse'] for x in c],color=color)
    ax[0].axhline(.2,color='gray',ls=':',lw=1);ax[0].axhline(.5,color='gray',ls='--',lw=1);ax[0].legend(frameon=False)
    for a,title in zip(ax,['Early optimizer budget (same warm fit)','Held-out EV, warm-start critic','Held-out RMSE, warm-start critic']):a.set_title(title);a.set_xlabel('Adam steps');a.grid(alpha=.2)
    ax[0].set_ylabel('Explained variance');ax[1].set_ylabel('Explained variance');ax[2].set_ylabel('Return RMSE')
    fig.suptitle('Fixed canonical actor; exact MC targets; scenarios 601–607 train / 608–610 validation',fontsize=11)
    fig.tight_layout();fig.savefig(out/'ACK2026_Critic_Learning_Curves_20260922.png',dpi=180);plt.close(fig)
    print(json.dumps(summary['totals']),flush=True)

if __name__=='__main__':main()
