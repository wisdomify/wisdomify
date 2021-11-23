import torch


def main():
    W = 10
    K = 11
    H = 768
    wisdom2subwords_embeddings = torch.ones(size=(W, K, H))
    print(wisdom2subwords_embeddings.shape)
    lstm = torch.nn.LSTM(input_size=H, hidden_size=H // 2, batch_first=True,
                         num_layers=1, bidirectional=True)
    hidden_state, _ = lstm(wisdom2subwords_embeddings)
    print(hidden_state.shape)
    last_hidden = hidden_state[:, -1]  # (W, K, H) -> (W, H)
    print(last_hidden.shape)


if __name__ == '__main__':
    main()
