"""
cross_entropy - 로짓을 입력으로 받는가? 아니면 확률을 입력으로 받는가?
"""

import torch
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss


def main():
    # --- predictions and labels --- #
    inf = 99999999  # if you set this too large, NaN will pop up
    logits = torch.FloatTensor([[-inf, inf, -inf],
                                [-inf, -inf, inf]])
    # we set the logits like above so that +inf gets a near-1 probability
    print(F.softmax(logits, dim=1))
    classes = torch.Tensor([1, 2]).long()  # target should be typed as long
    print("input shape:", logits.shape)
    print("output shape:", classes.shape)
    # --- functional: cross entropy with built-in log softmax --- #
    # pytorch - https://stackoverflow.com/a/49399421
    # 파이토치는 cross_entropy 안에 softmax가 빌트인.
    loss = F.cross_entropy(logits, classes)
    print(loss)  # should be zero

    # --- oop: cross entropy with built-in log softmax --- #
    criterion = CrossEntropyLoss()
    loss = criterion(logits, classes)
    print(loss)  # should be zero

    # --- What's actually going on above is --- #
    log_softmax = F.log_softmax(logits, dim=1)
    loss = F.nll_loss(log_softmax, classes)
    print(loss)  # should also be zero


if __name__ == '__main__':
    main()
