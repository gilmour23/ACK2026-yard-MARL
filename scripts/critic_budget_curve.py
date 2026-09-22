"""Additional measurement: match early optimizer budgets on fixed data.

No hyperparameter change. Reproduce the first epoch of warm-start fits, expose
held-out value metrics at steps10/20/40/60/98 omitted by epoch-only metrics.
"""
from critic_diag_common import *
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--old',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',required=True);ap.add_argument('--fit',required=True)
    a=ap.parse_args();torch.manual_seed(a.seed);d,eps,seeds,train=load_data(a.old);m=model_from_checkpoint(a.checkpoint);h=state_hash(m)
    x=torch.tensor(d['obs']);y=torch.tensor(d['mc'],dtype=torch.float32);opt=torch.optim.Adam(m.critic.parameters(),lr=3e-4);order=np.random.default_rng(a.seed).permutation(np.flatnonzero(train));rows=[]
    for step,start in enumerate(range(0,len(order),256),1):
        ix=order[start:start+256];loss=.25*(m.critic(x[ix])-y[ix]).square().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.critic.parameters(),.5);opt.step()
        if step in [10,20,40,60,98]:
            p=predict(m.critic,x[~train]);r=dict(step=step,validation=metrics(p,d['mc'][~train]));rows.append(r);print(json.dumps(r),flush=True)
    expected=json.loads((Path(a.fit)/'learning_curve.json').read_text())[1]['validation']
    assert abs(rows[-1]['validation']['ev']-expected['ev'])<1e-7
    assert state_hash(m)==h
    dump(a.out,dict(**provenance(a.checkpoint,a.seed,0,step,'fixed canonical stochastic trajectory value inference'),
        reason='matched early optimization exposure measurement; same data/order/loss/lr as primary warm fit',curve=rows,epoch1_reproduced=True,actor_updates=0))

if __name__=='__main__':main()
