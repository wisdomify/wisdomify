from typing import List
from transformers import AutoModelForMaskedLM, AutoConfig, AutoTokenizer
from loaders import load_device, load_config
from models import RDAlpha
from tensors import Wisdom2SubwordsBuilder


def get_wisdoms_a() -> List[str]:
    return [
        "가는 날이 장날",
        "갈수록 태산",
        "꿩 대신 닭",
        "등잔 밑이 어둡다",
        "소문난 잔치에 먹을 것 없다",
        "핑계 없는 무덤 없다",
        "고래 싸움에 새우 등 터진다",
        "서당개 삼 년이면 풍월을 읊는다",
        "원숭이도 나무에서 떨어진다",
        "산 넘어 산"
    ]


def main():
    device = load_device(use_gpu=False)
    config = load_config()['rd_alpha']['a']
    bert = config['bert']
    wisdoms = get_wisdoms_a()
    k = config['k']
    lr = config['lr']
    bert_mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(bert))
    tokenizer = AutoTokenizer.from_pretrained(bert)
    wisdom2subwords = Wisdom2SubwordsBuilder(tokenizer, k, device)(wisdoms)
    rd = RDAlpha(k, lr, bert_mlm, wisdom2subwords, device)
    print(rd.state_dict())  # key-value pairs of the weights.
    # checkpoints aren't like that, it is a huge blob.


if __name__ == '__main__':
    main()
