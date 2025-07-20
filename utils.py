import torch
import random
from collections import deque
import torch.nn.functional as F


class ReplayBuffer(): # for iid - indep and ident distri

    # store : (state, action, reward, next_state, done)
    # mini batching - sample = random.sample(buffer, batch_size)
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_1, a_1, r_1, s_prime_1, done_flag_1 = [], [], [], [], []  

        for transition in mini_batch:
            s, a, r, s_prime, done_flag = transition
            s_1.append(s)
            a_1.append([a])
            r_1.append([r])
            s_prime_1.append(s_prime)
            done_flag_1.append([done_flag])

        return torch.tensor(s_1, dtype=torch.float32), torch.tensor(a_1, dtype=torch.long), \
               torch.tensor(r_1, dtype=torch.float32), torch.tensor(s_prime_1, dtype=torch.float32), \
               torch.tensor(done_flag_1, dtype=torch.float32)
    
    def size(self):
        return len(self.buffer)
    

def train(q_net, 
          q_target,
          memory,
          optimizer,
          batch_size,
          gamma,
          device):
    
    for _ in range(10):
        s, a , r, s_prime, done_flag = memory.sample(batch_size) # monte carlo batching

        # Move tensors to device
        s = s.to(device)
        a = a.to(device)
        r = r.to(device)
        s_prime = s_prime.to(device)
        done_flag = done_flag.to(device)

        q_out= q_net(s) # q values

        q_a = q_out.gather(1,a) # update rule
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_flag
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        
        #  clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
        
        optimizer.step()







