
import torch
from torch.nn import functional as F

def main():
    # 40 batches, 10 wisdoms
    S_wisdom_lit = torch.ones(size=(40, 10))
    S_wisdom_fig = torch.ones(size=(40, 10))
    loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(torch.log_softmax(S_wisdom_lit, dim=1),
                                                                      torch.log_softmax(S_wisdom_fig, dim=1))
    print(loss)

    # 40 batches, 10 wisdoms
    S_wisdom_lit = torch.ones(size=(40, 10))
    S_wisdom_lit[0][0] = 300
    S_wisdom_fig = torch.ones(size=(40, 10))
    S_wisdom_fig[0][0] = 200
    loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(torch.log_softmax(S_wisdom_lit, dim=1),
                                                                      torch.log_softmax(S_wisdom_fig, dim=1))
    print(loss)
    S_wisdom_lit = torch.ones(size=(40, 10))
    S_wisdom_lit[0][0] = 300
    S_wisdom_fig = torch.ones(size=(40, 10))
    S_wisdom_fig[0][0] = 200
    loss = F.mse_loss(S_wisdom_fig, S_wisdom_lit)
    print(loss)

if __name__ == '__main__':
    main()
