import torch
import numpy as np



def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def CCA_loss(H1, H2, outdim_size, use_all_singular_values=False):
    r1 = 1e-3
    r2 = 1e-3
    eps = 1e-9

    H1, H2 = H1.t(), H2.t()

    o1 = o2 = H1.size(0)

    m = H1.size(1)

    H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
    H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

    SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
    SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar, H1bar.t()) + r1 * torch.eye(o1, device=H1.device)
    SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.t()) + r2 * torch.eye(o2, device=H1.device)

    # Calculating the root inverse of covariance matrices by using eigen decomposition
    [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
    [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)

    # Added to increase stability
    posInd1 = torch.gt(D1, eps).nonzero(as_tuple=False)[:, 0]
    D1 = D1[posInd1]
    V1 = V1[:, posInd1]
    posInd2 = torch.gt(D2, eps).nonzero(as_tuple=False)[:, 0]
    D2 = D2[posInd2]
    V2 = V2[:, posInd2]

    SigmaHat11RootInv = torch.matmul(torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
    SigmaHat22RootInv = torch.matmul(torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

    Tval = torch.matmul(torch.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

    if use_all_singular_values:
        # all singular values are used to calculate the correlation
        tmp = torch.matmul(Tval.t(), Tval)
        corr = torch.trace(torch.sqrt(tmp))
        # assert torch.isnan(corr).item() == 0
    else:
        # just the top outdim_size singular values are used
        trace_TT = torch.matmul(Tval.t(), Tval)
        trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(H1.device)) # regularization for more stability
        U, V = torch.symeig(trace_TT, eigenvectors=True)
        U = torch.where(U>eps, U, (torch.ones(U.shape)*eps).to(H1.device))
        U = U.topk(outdim_size)[0]
        corr = torch.sum(torch.sqrt(U))

    return - corr


class HardNegtive_loss(torch.nn.Module):

    def __init__(self, tau_plus=0.1, beta=1.0, temperature=0.5, alpha=256, estimator='hard'):
        super(HardNegtive_loss, self).__init__()
        self.tau_plus = tau_plus
        self.beta = beta
        self.temperature = temperature
        self.estimator = estimator
        self.alpha = alpha

    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def forward(self, out_1, out_2):
        # normalization
        batch_size, c = out_1.shape
        #out_1 = F.normalize(out_1, dim=1)
        #out_2 = F.normalize(out_2, dim=1)
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        # old_neg = neg.clone()
        mask = self.get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        # negative samples similarity scoring
        if self.estimator == 'hard':
            N = batch_size * 2 - 2
            imp = (self.beta * neg.log()).exp()
            reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
            Ng = (-self.tau_plus * N * pos + reweight_neg) / (1 - self.tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.temperature))
        elif self.estimator == 'easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng))).mean()
        # eqco
        #print(batch_size, Ng.shape)
        #loss = (- torch.log(pos / (pos + self.alpha / Ng.shape[0] * Ng))).mean()

        return loss