"""Read-only canonical dependencies and shared fixed-policy diagnostics."""
import sys, json, hashlib, subprocess
from pathlib import Path
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from train_yc_marl import ResourceCooperativeModel, ResourcePPOConfig
from yc_marl_env import ResourceMARLYardEnv

torch.set_num_threads(1)
BASE='f6e40f53e2ac7f34bea0dcd4c8761204cc3b965f'
CHECKPOINT_SHA='2bbd3a2e795a215d3fcbd654d58fad154d4298b55bb50ce92f54ad50b82ddb59'

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def model_from_checkpoint(path):
    assert sha(path)==CHECKPOINT_SHA
    c=torch.load(path,map_location='cpu',weights_only=True)
    m=ResourceCooperativeModel(c['global_obs_dim'],c['storage_obs_dim'],c['yc_obs_dim'],hidden=128)
    m.load_state_dict(c['state_dict'],strict=True);m.eval()
    return m

def metrics(pred,target):
    p=np.asarray(pred,dtype=np.float64);y=np.asarray(target,dtype=np.float64);e=p-y
    return dict(n=len(y),ev=float(1-np.var(e)/np.var(y)),rmse=float(np.sqrt(np.mean(e*e))),
        r2=float(1-np.mean(e*e)/np.var(y)),bias=float(np.mean(e)),
        prediction_mean=float(p.mean()),prediction_std=float(p.std()),prediction_variance=float(p.var()),
        target_mean=float(y.mean()),target_std=float(y.std()),target_variance=float(y.var()),
        prediction_min=float(p.min()),prediction_max=float(p.max()))

def predict(critic,x):
    with torch.no_grad():return torch.cat([critic(v) for v in x.split(512)]).numpy()

def norm(params,grad=False):
    vals=[(p.grad if grad else p).detach().double().square().sum() for p in params if not grad or p.grad is not None]
    return float(torch.stack(vals).sum().sqrt()) if vals else 0.0

def state_hash(model,prefixes=('storage_actor.','yc_actor.')):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        if k.startswith(prefixes):h.update(k.encode());h.update(v.numpy().tobytes())
    return h.hexdigest()

def load_data(old):
    root=Path(old)/'validation/canonical';d=np.load(root/'critic_dataset.npz')
    eps=json.loads((root/'episodes.json').read_text());seeds=np.array([eps[int(e)]['seed'] for e in d['episode_ids']])
    assert len(eps)==30 and set(seeds)==set(range(601,611))
    return d,eps,seeds,seeds<=607

def provenance(checkpoint,seed,decisions,updates,mode):
    return dict(base_commit=BASE,instrumentation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        checkpoint_sha256=sha(checkpoint),training_seed=seed,actual_environment_decisions=decisions,
        optimizer_updates=updates,evaluation_mode=mode,train_scenarios=list(range(601,608)),validation_scenarios=list(range(608,611)))
