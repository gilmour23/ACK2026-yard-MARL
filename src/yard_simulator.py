from __future__ import annotations

"""Core V4 yard data structures and transparent information-aware heuristics."""

from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple, Iterable
import heapq, random


@dataclass
class Container:
    cid:str
    yard_arrival:float
    truck_actual:float
    eta:float
    prev_eta:float
    cohort:str='initial'
    retrievable_this_episode:bool=False
    block:Optional[int]=None
    stack:Optional[int]=None
    stored:bool=False
    storage_complete:Optional[float]=None
    truck_arrived:bool=False
    retrieval_requested:bool=False
    retrieval_complete:Optional[float]=None
    premarshal_pending:bool=False
    eta_delta:float=0.0
    last_eta_update:Optional[float]=None


@dataclass
class Task:
    kind:str
    container_id:str
    duration:float=2.0
    target_block:Optional[int]=None
    target_stack:Optional[int]=None
    created_at:float=0.0


@dataclass(order=True)
class Event:
    time:float
    priority:int
    seq:int
    kind:str=field(compare=False)
    payload:dict=field(compare=False,default_factory=dict)


class Yard:
    def __init__(self,n_blocks:int,stacks_per_block:int,max_tier:int,relocation_buffer_slots_per_block:int=0):
        self.n_blocks=n_blocks;self.stacks_per_block=stacks_per_block;self.max_tier=max_tier;self.relocation_buffer_slots_per_block=relocation_buffer_slots_per_block
        self.stacks:List[List[List[str]]]=[[[] for _ in range(stacks_per_block)] for _ in range(n_blocks)]
        self.reserved=[[0 for _ in range(stacks_per_block)] for _ in range(n_blocks)]
    def feasible_storage_slots(self)->List[Tuple[int,int]]:
        out=[];bc=self.stacks_per_block*self.max_tier
        for b in range(self.n_blocks):
            used=sum(len(st) for st in self.stacks[b]);reserved=sum(self.reserved[b])
            if used+reserved>=bc-self.relocation_buffer_slots_per_block:continue
            for s in range(self.stacks_per_block):
                if len(self.stacks[b][s])+self.reserved[b][s]<self.max_tier:out.append((b,s))
        return out
    def feasible_relocation_stacks(self,block:int,source_stack:int)->List[int]:
        return [s for s in range(self.stacks_per_block) if s!=source_stack and len(self.stacks[block][s])+self.reserved[block][s]<self.max_tier]
    def push(self,block:int,stack:int,cid:str)->None:
        if len(self.stacks[block][stack])>=self.max_tier:raise RuntimeError('Stack full')
        self.stacks[block][stack].append(cid)
    def reserve(self,block:int,stack:int)->None:
        if len(self.stacks[block][stack])+self.reserved[block][stack]>=self.max_tier:raise RuntimeError('Cannot reserve full stack')
        self.reserved[block][stack]+=1
    def release_reservation(self,block:int,stack:int)->None:
        if self.reserved[block][stack]<=0:raise RuntimeError('No reservation to release')
        self.reserved[block][stack]-=1
    def pop_top(self,block:int,stack:int)->str:
        if not self.stacks[block][stack]:raise RuntimeError('Stack empty')
        return self.stacks[block][stack].pop()
    def top(self,block:int,stack:int)->Optional[str]:return self.stacks[block][stack][-1] if self.stacks[block][stack] else None
    def blockers_above(self,cid:str,containers:Dict[str,Container])->int:
        c=containers[cid]
        if not c.stored or c.block is None or c.stack is None:return 0
        st=self.stacks[c.block][c.stack];idx=st.index(cid);return len(st)-idx-1
    def occupancy(self)->float:
        used=sum(len(st) for block in self.stacks for st in block);return used/(self.n_blocks*self.stacks_per_block*self.max_tier)


class YardCrane:
    def __init__(self,block:int):
        self.block=block;self.queue:Deque[Task]=deque();self.busy=False;self.current:Optional[Task]=None;self.busy_time=0.0;self.completed_moves=0;self.max_queue_len=0


class InformationAwareHeuristic:
    """Transparent baseline used for storage placement and reactive destinations."""
    def __init__(self,queue_penalty:float=0.08,announced_inbound_queue_scale:float=0.5):
        self.queue_penalty=queue_penalty;self.announced_inbound_queue_scale=announced_inbound_queue_scale
    def choose_storage(self,sim:'YardSimulator',cid:str,candidates:Iterable[Tuple[int,int]])->Tuple[int,int]:
        c=sim.containers[cid];scored=[]
        window=30.0
        recent=sum(1 for row in sim.event_log if row.get('event_type')=='CONTAINER_ARRIVAL' and sim.now-window<=float(row.get('timestamp',-1e9))<=sim.now)
        rate=float(getattr(sim,'arrival_rate_per_hour',0.0) or 0.0);expected=max(rate*window/60.0,1.0);pending=len(getattr(sim,'pending_storage_queue',()))
        inbound_pressure=min(recent/expected+pending/10.0,3.0)
        for b,s in candidates:
            stack=sim.yard.stacks[b][s];inversion=sum(1 for lower_id in stack if sim.containers[lower_id].eta<c.eta);height=len(stack)/sim.yard.max_tier;queue=len(sim.ycs[b].queue)+(1 if sim.ycs[b].busy else 0)
            qp=self.queue_penalty*(1+self.announced_inbound_queue_scale*inbound_pressure);scored.append((inversion+0.35*height+qp*queue,b,s))
        if not scored:raise RuntimeError('No feasible storage location')
        scored.sort();return scored[0][1],scored[0][2]
    def choose_relocation_destination(self,sim:'YardSimulator',moving_cid:str,block:int,source_stack:int)->int:
        c=sim.containers[moving_cid];cand=sim.yard.feasible_relocation_stacks(block,source_stack)
        if not cand:raise RuntimeError('No feasible relocation destination within block')
        scored=[]
        for s in cand:
            stack=sim.yard.stacks[block][s];inv=sum(1 for lower_id in stack if sim.containers[lower_id].eta<c.eta);height=len(stack)/sim.yard.max_tier;scored.append((inv+0.35*height,s))
        scored.sort();return scored[0][1]


class YardSimulator:
    def __init__(self,n_blocks:int=4,stacks_per_block:int=25,max_tier:int=4,move_time:float=2.0,seed:int=0,policy:Optional[InformationAwareHeuristic]=None,relocation_buffer_slots_per_block:int=0):
        self.rng=random.Random(seed);self.seed=seed;self.yard=Yard(n_blocks,stacks_per_block,max_tier,relocation_buffer_slots_per_block);self.ycs=[YardCrane(b) for b in range(n_blocks)];self.move_time=move_time;self.policy=policy or InformationAwareHeuristic();self.now=0.0;self._seq=0;self.events:List[Event]=[];self.containers:Dict[str,Container]={};self.event_log=[];self.rehandling_moves=0;self.proactive_moves=0;self.storage_moves=0;self.retrieval_moves=0
    def schedule(self,time:float,kind:str,**payload)->None:
        self._seq+=1;priority={'YC_TASK_COMPLETE':0,'TRUCK_ARRIVAL':1,'CONTAINER_ARRIVAL':1,'TRUCK_ETA_UPDATE':2,'STORAGE_RETRY':3,'OPERATING_WINDOW_END':4}.get(kind,3);heapq.heappush(self.events,Event(float(time),priority,self._seq,kind,payload))
    def log(self,event_type:str,cid:Optional[str]=None,**extra)->None:self.event_log.append({'timestamp':self.now,'event_type':event_type,'container_id':cid,**extra})
    def add_initial_container(self,cid:str,block:int,stack:int,truck_actual:float,initial_eta:float)->None:
        c=Container(cid=cid,yard_arrival=0.0,truck_actual=truck_actual,eta=initial_eta,prev_eta=initial_eta,cohort='initial',block=block,stack=stack,stored=True,storage_complete=0.0);self.containers[cid]=c;self.yard.push(block,stack,cid)
