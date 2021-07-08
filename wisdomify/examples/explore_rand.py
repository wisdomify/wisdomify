import torch


def main():
    # A batch of size 10 by 5.
    preds = torch.randn(10, 5).softmax(dim=-1)
    print(preds)

    # 10 labels, the values of which are capped at 5.
    target = torch.randint(high=5, size=(10,))
    print(target)


if __name__ == '__main__':
    main()
