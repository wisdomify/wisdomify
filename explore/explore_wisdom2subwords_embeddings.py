from transformers import BertForMaskedLM, BertTokenizer

from wisdomify.connectors import connect_to_wandb
from wisdomify.flows import WisdomsFlow
from wisdomify.loaders import load_device
from wisdomify.tensors import Wisdom2SubwordsBuilder


def main():
    with connect_to_wandb(job_type="explore", config={}) as run:
        flow = WisdomsFlow(run, ver="a")("d", {})
    wisdoms = [row[0] for row in flow.raw_table.data]
    device = load_device(gpu=False)
    bert_mlm = BertForMaskedLM.from_pretrained("beomi/kcbert-base")
    tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
    wisdom2subwords = Wisdom2SubwordsBuilder(tokenizer, 11, device)(wisdoms)
    print(bert_mlm.bert.embeddings.word_embeddings)  # (30000, 768)
    print(bert_mlm.bert.embeddings.word_embeddings(wisdom2subwords))  # (10, K, 768)
    print(bert_mlm.bert.embeddings.word_embeddings(wisdom2subwords).shape)  # (10, K, 768)
    """
    torch.Size([10, 11, 768])
    """



if __name__ == '__main__':
    main()