"""Frozen-actor, same-architecture exact-MC fit; bounded offline experiment."""
from critic_diag_common import *
import argparse, csv, time, copy

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--old',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--initialization',choices=['warm','fresh'],required=True);ap.add_argument('--epochs',type=int,default=20)
    a=ap.parse_args();assert 1<=a.epochs<=20
    torch.manual_seed(a.seed);d,eps,seeds,train=load_data(a.old);x=torch.tensor(d['obs']);y=torch.tensor(d['mc'],dtype=torch.float32)
    m=model_from_checkpoint(a.checkpoint);actor_hash=state_hash(m)
    for p in m.storage_actor.parameters():p.requires_grad_(False)
    for p in m.yc_actor.parameters():p.requires_grad_(False)
    if a.initialization=='fresh':
        # Reset only critic initialization, not actor or architecture.
        torch.manual_seed(a.seed)
        from v4_networks import PermutationInvariantCritic
        m.critic=PermutationInvariantCritic(8,4,25,8,4,128)
    c=m.critic;params=list(c.parameters());opt=torch.optim.Adam(params,lr=3e-4)
    rng=np.random.default_rng(a.seed);ids=np.flatnonzero(train);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    curves=[];step=0;start=time.monotonic()
    def assess(epoch):
        p=predict(c,x);r=dict(epoch=epoch,optimizer_step=step,train=metrics(p[train],d['mc'][train]),validation=metrics(p[~train],d['mc'][~train]))
        with torch.no_grad():
            z,_,_=c.encoder(x[::40]);r['fusion_fraction_abs_gt_099']=float((z.abs()>.99).float().mean())
        curves.append(r);dump(out/'learning_curve.json',curves)
        print(json.dumps(dict(initialization=a.initialization,seed=a.seed,epoch=epoch,step=step,train_ev=r['train']['ev'],validation_ev=r['validation']['ev'],validation_rmse=r['validation']['rmse'],seconds=time.monotonic()-start)),flush=True)
    assess(0)
    with (out/'optimizer_steps.csv').open('w',newline='') as f:
        writer=None
        for epoch in range(1,a.epochs+1):
            order=rng.permutation(ids);coverage=np.zeros(len(x),int)
            for start_idx in range(0,len(order),256):
                ix=order[start_idx:start_idx+256];coverage[ix]+=1;p=c(x[ix]);loss=.25*(p-y[ix]).square().mean()
                opt.zero_grad();loss.backward();before=norm(params,True)
                encoder_before=norm(list(c.encoder.parameters()),True);head_before=norm(list(c.head.parameters()),True)
                torch.nn.utils.clip_grad_norm_(params,.5);after=norm(params,True)
                prev=[v.detach().clone() for v in params];opt.step();step+=1
                delta=norm([v.detach()-old for v,old in zip(params,prev)])
                r=dict(step=step,epoch=epoch,samples=len(ix),value_loss=float(loss.detach()),
                    target_mean=float(y[ix].mean()),target_std=float(y[ix].std(unbiased=False)),prediction_mean=float(p.detach().mean()),prediction_std=float(p.detach().std(unbiased=False)),
                    grad_before=before,grad_after=after,clip_scale=after/before if before else 1.,clipped=int(before>.5),
                    encoder_grad_before=encoder_before,head_grad_before=head_before,parameter_update_norm=delta)
                if writer is None:writer=csv.DictWriter(f,fieldnames=r);writer.writeheader()
                writer.writerow(r)
            assert np.all(coverage[train]==1) and np.all(coverage[~train]==0)
            f.flush();assess(epoch)
    assert state_hash(m)==actor_hash
    p=predict(c,x);np.savez_compressed(out/'predictions.npz',prediction=p,scenario=seeds)
    torch.save({'critic_state_dict':c.state_dict(),'optimizer_state_dict':opt.state_dict(),'initialization':a.initialization,'seed':a.seed,'epochs':a.epochs},out/'critic_only.pt')
    dump(out/'summary.json',dict(**provenance(a.checkpoint,a.seed,0,step,'fixed canonical stochastic trajectories; deterministic value inference'),
        dataset_transitions=len(x),train_transitions=int(train.sum()),validation_transitions=int((~train).sum()),source_dataset_sha256=sha(Path(a.old)/'validation/canonical/critic_dataset.npz'),
        actor_hash_before=actor_hash,actor_hash_after=state_hash(m),initialization=a.initialization,epochs=a.epochs,lr=3e-4,loss='0.25*MSE',max_grad_norm=.5,final=curves[-1],elapsed_seconds=time.monotonic()-start))

if __name__=='__main__':main()
