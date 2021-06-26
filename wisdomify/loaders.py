from typing import List, Tuple
import csv
import json
from wisdomify.paths import WISDOM2DEF_TSV, CONF_JSON


def load_wisdom2def() -> List[Tuple[str, str]]:
    """
    가는 날이 장날 -> 어떤 일을 하려고 하는데 뜻하지 않은 일을 공교롭게 당함.
    :return:
    """
    with open(WISDOM2DEF_TSV, 'r') as fh:
        tsv_reader = csv.reader(fh, delimiter="\t")
        next(tsv_reader)
        return [
            (row[0], def_)  # wisdom, def pair
            for row in tsv_reader
            for def_ in row[1:]
        ]


def load_wisdom2eg() -> List[Tuple[str, str]]:
    """
    This is to be implemented later.
    e.g.
    가는 날이 장날 -> 한 달 전부터 기대하던 소풍날 아침에 굵은 빗방울이 떨어지기 시작했다.
    """
    raise NotImplementedError


def load_conf() -> dict:
    with open(CONF_JSON, 'r') as fh:
        return json.loads(fh.read())


