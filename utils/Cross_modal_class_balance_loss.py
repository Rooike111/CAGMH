import torch
import torch.optim as optim
from scipy.linalg import hadamard
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class Cross_modal_class_balance_loss(torch.nn.Module):
    def __init__(self, args, bit, gamma=2., alpha=0.25):
        super(Cross_modal_class_balance_loss, self).__init__()
        self.hash_targets = self.get_hash_targets(args.numclass, bit).to(args.device)
        self.args=args
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(args.device)
        self.balance_weight = torch.tensor(args.balance_weight).float().to(args.device)
        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def reszet_parameters(self):
        # Initial weight
        self.weight.data.fill_(0.5)
        #self.weight.data.fill_(1)

    def hloss(self, outputs, targets):

        if torch.min(targets) >= 0:
            targets = 2 * targets - 1  

        # 计算Hinge Loss
        hinge_loss = 1 - outputs * targets
        hinge_loss[hinge_loss < 0] = 0  
        return hinge_loss.mean()

    def forward(self, u, y, args):
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.hloss(u,hash_center)
        Q_loss = (u.abs() - 1).pow(3).mean()
        y = y.float().to(self.args.device)

        balance_loss = self.balance_weight * ((y.mean(dim=0) - 0.5).abs().mean())

        return torch.sigmoid(self.weight)*(center_loss  + args.lambda1 * Q_loss +balance_loss)

    def label2center(self, y):
        #将标签转成float 放在cpu上
        y = y.float().to(self.args.device)
        center_sum = y @ self.hash_targets
        random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
        center_sum[center_sum == 0] = random_center[center_sum == 0]
        hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()
        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for _ in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets