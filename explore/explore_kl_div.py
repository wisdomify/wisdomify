
import torch


def main():
    # 40 batches, 10 wisdoms
    S_wisdom_lit = torch.randn(size=(40, 10))
    S_wisdom_fig = torch.randn(size=(40, 10))
    loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(torch.log_softmax(S_wisdom_lit, dim=1),
                                                                      torch.log_softmax(S_wisdom_fig, dim=1))
    print(loss)

if __name__ == '__main__':
    main()
