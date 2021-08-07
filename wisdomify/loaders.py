from typing import List, Tuple
import csv
import json
from wisdomify.paths import WISDOMDATA_VER_0_DIR, CONF_JSON
from dataclasses import dataclass
from os import path


class WisdomDataLoader:
    """
    def, eg. -> 고민...
    파라미터로 받아서... 따로 로드.
    이건, def, eg을 모두 한번에 사용할 것이냐, 아니냐에 달려있음.
    둘을 분리해서 훈련에 사용해야 한다? -> 지금은 분리해서.
    사실 분리해서 training 한다고 해도...
    이런 가정: 데이터 수집을 할때, 1. 가는날이 장날이다를 설명해주세요 (wisdom2def). 2. 가느날의 장날이다의 예시를 들어주세요 (wisdom2eg).
    확실히 다른 데이터니까, 분리해서 관리를 하는 것이 좋을 것.
    나중에.. 예시만을 보여주고 싶을때도 있을 것.
    둘의 버전은 통합관리.
    둘을 합쳐도 상관이 없다? ->
    """

    def __init__(self, ver: str, wisdom2def=None, wisdom2eg=None):
        self.ver: str = ver
        # this must come as a pair
        self.wisdom2def: List[Tuple[str, str]] = wisdom2def
        self.wisdom2eg: List[Tuple[str, str]] = wisdom2eg

    def __call__(self) -> 'WisdomDataLoader':
        if self.ver == "0":
            version_dir = WISDOMDATA_VER_0_DIR
        elif self.ver == "1":
            version_dir = WISDOMDATA_VER_0_DIR
        else:
            raise NotImplementedError
        wisdom2def_tsv_path = path.join(version_dir, "wisdom2def.tsv")
        wisdom2eg_tsv_path = path.join(version_dir, "wisdom2eg.tsv")
        wisdom2def = self.load_wisdom2sent(wisdom2def_tsv_path)
        wisdom2eg = self.load_wisdom2sent(wisdom2eg_tsv_path)
        data = WisdomDataLoader(self.ver, wisdom2def, wisdom2eg)
        return data

    @staticmethod
    def load_wisdom2sent(tsv_path: str) -> List[Tuple[str, str]]:
        """
        wisdom2def인 경우: 가는 날이 장날 -> 어떤 일을 하려고 하는데 뜻하지 않은 일을 공교롭게 당함.
        wisdom2eg인 경우: 가는 날이 장날 -> 한 달 전부터 기대하던 소풍날 아침에 굵은 빗방울이 떨어지기 시작했다.
        """
        with open(tsv_path, 'r') as fh:
            tsv_reader = csv.reader(fh, delimiter="\t")
            next(tsv_reader)
            return [
                (row[0], row[1])  # wisdom, sent pair
                for row in tsv_reader
            ]


def load_conf() -> dict:
    with open(CONF_JSON, 'r') as fh:
        return json.loads(fh.read())
