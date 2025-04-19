import torch
import numpy as np
from typing import Union
import torch.nn as nn
from torch.nn import functional as F
from utils.get_args import threshold
from sklearn.metrics.pairwise import euclidean_distances
from utils.get_args import get_args


class multimodal_proxy_loss(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.args = get_args()
        #随机数种子选择0
        torch.manual_seed(self.args.hypseed)
        # Initialization
        self.proxies = torch.nn.Parameter(torch.randn(self.args.numclass, self.args.output_dim).to(self.args.device))
        nn.init.kaiming_normal_(self.proxies, mode = 'fan_out')

    def forward(self, x=None, y=None, label=None):
        """
        parem: x  图像哈希 [128, 16]
        parem: y, 文本哈希 [128, 16]
        label [128, 24]
        """
        # 初始化
        P_one_hot = label
        
        """
        计算了 x 和 self.proxies 中每个归一化向量之间的余弦相似度，并将结果存储在 cos 中。这通常用于计算查询（x）与一组原型（self.proxies）之间的相似度，例如在最近邻搜索、分类或聚类等任务中。
        """
        cos = F.normalize(x, p = 2, dim = 1).mm(F.normalize(self.proxies, p = 2, dim = 1).T)
        # 正负相似度
        pos = 1 - cos
        neg = F.relu(cos - threshold)
        #计算余弦相似度:
        cos_t = F.normalize(y, p = 2, dim = 1).mm(F.normalize(self.proxies, p = 2, dim = 1).T)
        # 计算正向
        # 由于余弦相似度的值范围在 [-1, 1] 之间，1 - cos_t 的结果将是一个在 [0, 2] 范围内的值，其中 0 表示完全不相似，2 表示完全相似。
        pos_t = 1 - cos_t
        #负项 neg_t 是通过应用 ReLU 函数（F.relu）到 cos_t - threshold 的结果来计算的。这里，threshold 是一个阈值，
        # 用于确定何时应该考虑一个样本和一个代理之间的相似度为负。只有当 cos_t 小于 threshold 时，neg_t 的值才会是非零的（具体为 cos_t - threshold），
        # 否则为 0。这通常用于鼓励模型学习不同类别之间的更大间隔。
        neg_t = F.relu(cos_t - threshold)

        #统计每个样本有多少个正标签
        P_num = len(P_one_hot.nonzero())
        #统计这代表了每个样本有多少个负标签（即多少个位置上的值是0）。
        N_num = len((P_one_hot == 0).nonzero())

        """
        这段代码的目的是计算多标签分类任务中的正项和负项，这些项将用于计算最终的损失函数。正项鼓励模型将相同类别的样本表示得更接近，而负项则鼓励模型将不同类别的样本表示得更远。
        """
        pos_term = torch.where(P_one_hot  ==  1, pos.to(torch.float32), torch.zeros_like(cos).to(torch.float32)).sum() / P_num
        neg_term = torch.where(P_one_hot  ==  0, neg.to(torch.float32), torch.zeros_like(cos).to(torch.float32)).sum() / N_num

        pos_term_t = torch.where(P_one_hot  ==  1, pos_t.to(torch.float32), torch.zeros_like(cos_t).to(torch.float32)).sum() / P_num
        neg_term_t = torch.where(P_one_hot  ==  0, neg_t.to(torch.float32), torch.zeros_like(cos_t).to(torch.float32)).sum() / N_num
        
        #判断其是否启用了正则化，若未启用则直接设置正则化为0
        if self.args.alpha > 0:

            """
            这里筛选出了那些具有多于一个标签的样本。label.sum(dim=1) > 1 会返回一个布尔向量，
            其中每个元素表示对应样本是否有多于一个的标签。然后，根据这个布尔向量筛选出对应的标签、特征 x 和目标 t。
            """
            index = label.sum(dim = 1) > 1
            label_ = label[index].float()

            x_ = x[index]
            t_ = y[index]
            # 计算了筛选后的标签之间的余弦相似度矩阵 cos_sim
            cos_sim = label_.mm(label_.T)
            """
            如果 self.args.alpha 不大于 0，则正则化项被设置为 0；否则，它们是基于样本之间的相似度计算得到的。这些正则化项将用于最终的损失计算，以鼓励模型学习到更好的表示。
            """
            if len((cos_sim == 0).nonzero()) == 0:
                #如果 cos_sim 中没有值为 0 的元素（即所有样本对至少共享一个标签），则不计算正则化项，直接设置为 0。
                reg_term = 0
                reg_term_t = 0
                reg_term_xt = 0
            else:
                x_sim = F.normalize(x_, p = 2, dim = 1).mm(F.normalize(x_, p = 2, dim = 1).T)
                t_sim = F.normalize(t_, p = 2, dim = 1).mm(F.normalize(t_, p = 2, dim = 1).T)
                xt_sim = F.normalize(x_, p = 2, dim = 1).mm(F.normalize(t_, p = 2, dim = 1).T)

                neg = self.args.alpha * F.relu(x_sim - threshold)
                neg_t = self.args.alpha * F.relu(t_sim - threshold)
                neg_xt = self.args.alpha * F.relu(xt_sim - threshold)

                reg_term = torch.where(cos_sim == 0, neg, torch.zeros_like(x_sim)).sum() / len((cos_sim == 0).nonzero())
                reg_term_t = torch.where(cos_sim == 0, neg_t, torch.zeros_like(t_sim)).sum() / len((cos_sim == 0).nonzero())
                reg_term_xt = torch.where(cos_sim == 0, neg_xt, torch.zeros_like(xt_sim)).sum() / len((cos_sim == 0).nonzero())
        else:
            reg_term = 0
            reg_term_t = 0
            reg_term_xt = 0
        return pos_term + neg_term + pos_term_t + neg_term_t + reg_term + reg_term_t + reg_term_xt





def compute_metrics(x):
    # 取复值的原因在于cosine的值越大说明越相似，但是需要取的是前N个值，所以取符号变为增函数s
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics


def calc_neighbor(a: torch.Tensor, b: torch.Tensor):
    # print(a.dtype, b.dtype)
    return (a.matmul(b.transpose(0, 1)) > 0).float()


def euclidean_similarity(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        similarity = torch.cdist(a, b, p=2.0)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        similarity = euclidean_distances(a, b)
    else:
        raise ValueError("input value must in [torch.Tensor, numpy.ndarray], but it is %s, %s"%(type(a), type(b)))
    return similarity


def euclidean_dist_matrix(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """
    calculate euclidean distance as inner product
    :param tensor1: a tensor with shape (a, c)
    :param tensor2: a tensor with shape (b, c)
    :return: the euclidean distance matrix which each point is the distance between a row in tensor1 and a row in tensor2.
    """
    dim1 = tensor1.shape[0]
    dim2 = tensor2.shape[0]
    multi = torch.matmul(tensor1, tensor2.t())
    a2 = torch.sum(torch.pow(tensor1, 2), dim=1, keepdim=True).expand(dim1, dim2)
    b2 = torch.sum(torch.pow(tensor2, 2), dim=1, keepdim=True).t().expand(dim1, dim2)
    dist = torch.sqrt(a2 + b2 - 2 * multi)
    return dist


def cosine_similarity(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        a = a / a.norm(dim=-1, keepdim=True) if len(torch.where(a != 0)[0]) > 0 else a
        b = b / b.norm(dim=-1, keepdim=True) if len(torch.where(b != 0)[0]) > 0 else b
        return torch.matmul(a, b.t())
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        a = a / np.linalg.norm(a, axis=-1, keepdims=True) if len(np.where(a != 0)[0]) > 0 else a
        b = b / np.linalg.norm(b, axis=-1, keepdims=True) if len(np.where(b != 0)[0]) > 0 else b
        return np.matmul(a, b.T)
    else:
        raise ValueError("input value must in [torch.Tensor, numpy.ndarray], but it is %s, %s"%(type(a), type(b)))

def calc_map_k(qB, rB, query_L, retrieval_L, k=None, rank=0):
    num_query = query_L.shape[0]
    qB = torch.sign(qB)
    rB = torch.sign(rB)
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)      # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calcHammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.to(rank)
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map


def calcHammingDist(B1, B2):

    if len(B1.shape) < 2:
        B1.view(1, -1)
    if len(B2.shape) < 2:
        B2.view(1, -1)
    q = B2.shape[1]
    if isinstance(B1, torch.Tensor):
        distH = 0.5 * (q - torch.matmul(B1, B2.t()))
    elif isinstance(B1, np.ndarray):
        distH = 0.5 * (q - np.matmul(B1, B2.transpose()))
    else:
        raise ValueError("B1, B2 must in [torch.Tensor, np.ndarray]")
    return distH
