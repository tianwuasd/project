"""Post-decision temporal-difference learning with matched sign-constrained ablation."""
import copy
import random
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from environment import ReservationEnv


class ValueNetwork(nn.Module):
    def __init__(self,capacity_dim,monotone=True):
        super().__init__();self.monotone=monotone
        self.context=nn.Sequential(nn.Linear(4,32),nn.Tanh(),nn.Linear(32,capacity_dim))
        nn.init.zeros_(self.context[-1].weight)
        nn.init.constant_(self.context[-1].bias,-3. if monotone else float(F.softplus(torch.tensor(-3.)) ))
        self.intercept=nn.Sequential(nn.Linear(4,16),nn.Tanh(),nn.Linear(16,1))

    def forward(self,x):
        weights=self.context(x[:,:4])
        if self.monotone: weights=F.softplus(weights)
        # Concave increasing transform: capacity increases cannot reduce value
        # when every contextual weight is nonnegative. Temporal context fixed.
        capacity=1-torch.exp(-x[:,4:])
        return (weights*capacity).sum(dim=1)+self.intercept(x[:,:4]).squeeze(-1)


def choose(network,candidates,scale):
    with torch.no_grad(): values=network(torch.from_numpy(np.array([c['features'] for c in candidates]))).numpy()
    scores=np.array([c['reward']/scale for c in candidates])+values
    return int(np.argmax(scores))


def evaluate(network,episodes,config):
    rewards=[]
    for orders in episodes.values():
        env=ReservationEnv(orders,config)
        while not env.done:
            cs=env.candidates();env.step(cs[choose(network,cs,config.reward_scale)]['action'])
        assert env.audit()['feasible'];rewards.append(env.reward)
    return float(np.mean(rewards))


def train(episodes,validation,config,seed,monotone=True,epochs=400):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
    net=ValueNetwork(3*config.slots,monotone);target=copy.deepcopy(net)
    optimizer=torch.optim.Adam(net.parameters(),lr=0.001)
    replay=deque(maxlen=50000);days=list(episodes);updates=0;steps=0;log=[]
    best=copy.deepcopy(net.state_dict());best_score=-float('inf')
    for epoch in range(epochs):
        env=ReservationEnv(episodes[random.choice(days)],config)
        while not env.done:
            cs=env.candidates();epsilon=max(.05,.7*(1-epoch/(.8*epochs)))
            index=random.randrange(len(cs)) if random.random()<epsilon else choose(net,cs,config.reward_scale)
            previous=cs[index]['features'];env.step(cs[index]['action']);steps+=1
            # z_t is after action t: its target starts at the NEXT reward.
            if env.done:
                replay.append((previous,np.zeros((1,len(previous)),dtype=np.float32),np.zeros(1,dtype=np.float32),True))
            else:
                nxt=env.candidates()
                replay.append((previous,np.array([c['features'] for c in nxt]),np.array([c['reward']/config.reward_scale for c in nxt],dtype=np.float32),False))
            if len(replay)>=64 and steps%4==0:
                batch=random.sample(list(replay),64)
                x=torch.from_numpy(np.array([b[0] for b in batch]))
                lengths=[len(b[1]) for b in batch]
                xn=torch.from_numpy(np.concatenate([b[1] for b in batch]))
                rewards=torch.from_numpy(np.concatenate([b[2] for b in batch]))
                with torch.no_grad():
                    # Double estimator uses online selection and target evaluation.
                    online=net(xn)+rewards;fixed=target(xn)+rewards;ys=[];start=0
                    for b,length in zip(batch,lengths):
                        k=int(torch.argmax(online[start:start+length]))
                        ys.append(torch.tensor(0.) if b[3] else fixed[start+k]);start+=length
                    y=torch.stack(ys)
                loss=F.smooth_l1_loss(net(x),y)
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step();updates+=1
                if updates%100==0: target.load_state_dict(net.state_dict())
        if (epoch+1)%50==0 or epoch==epochs-1:
            score=evaluate(net,validation,config);log.append({'episode':epoch+1,'validation_reward':score,'steps':steps,'updates':updates})
            if score>best_score: best_score=score;best=copy.deepcopy(net.state_dict())
    net.load_state_dict(best)
    return net,{'seed':seed,'monotone':monotone,'training_episodes':epochs,'steps':steps,'updates':updates,'best_validation_reward':best_score,'history':log}
