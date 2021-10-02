from typing import List, Tuple
from wisdomify.builders import BuilderBeta
from wisdomify.loaders import load_conf
from transformers import AutoTokenizer

# 예시 문장.
WISDOM2SENT: List[Tuple[str, str]] = [
    ('소문난 잔치에 먹을 것 없다',
     '2005년 출범 후 6차례 열린 챔피언결정전에서 삼성화재가 4회 현대캐피탈이 2회 우승컵을 가져갔다 그러나 [WISDOM]고 최근에는 삼성화재의 독주였다 2008∼2009시즌 중반부터 올 3라운드까지 정규시즌 13경기에서 12승 1패를 거뒀다'),
    ("꿩 대신 닭", "[WISDOM]이라는데 코카콜라가 없으니 펩시 마신다.")
]
VER: str = "4"


def main():
    conf = load_conf()
    vers = conf['versions']
    selected_ver = vers[VER]
    bert_model: str = selected_ver['bert_model']
    k: int = selected_ver['k']
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    X = BuilderBeta.build_X(wisdom2sent=WISDOM2SENT, tokenizer=tokenizer, k=k)
    print(X)


if __name__ == '__main__':
    main()

