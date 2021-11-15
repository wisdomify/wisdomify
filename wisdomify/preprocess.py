import json
import requests
import re

import pandas as pd

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from soynlp.normalizer import emoticon_normalize, only_text
from typing import Tuple


def check_grammar(text: str) -> str:
    base_url = 'https://m.search.naver.com/p/csearch/ocontent/spellchecker.nhn'
    payload = {
        '_callback': 'window.__jindo2_callback._spellingCheck_0',
        'q': text
    }
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
        'referer': 'https://search.naver.com/',
    }
    r = requests.get(
        base_url,
        params=payload,
        headers=headers
    )
    data = json.loads(r.text[42:-2])

    corrected = data['message']['result']['notag_html']

    return corrected


def augment(df: pd.DataFrame) -> pd.DataFrame:
    # TODO implement augmentation.
    return df


def parse(df: pd.DataFrame) -> pd.DataFrame:
    """
    parse <em> ...</em>  to [WISDOM].
    :param df: raw_df includes 'wisdom' and 'eg' field
    :return: 'eg' field parsed df
    """

    df['eg'] = df['eg'].apply(
        # return list of example only on 'eg' column (proverb is converted to [WISDOM])
        lambda r: list(map(
            # while iterating list of 'hits'
            # convert <em> ... lalib ... </em> to [WISDOM]
            lambda hit: re.sub(r"<em>.*</em>", "[WISDOM]", hit['highlight']['sents'][0]),
            # loading json to dict -> taking dict['hits']['hits']
            json.loads(r)['hits']['hits']
        ))
    )

    # 'eg' column contains list object
    # -> converted to single value with multiple columns
    df = df.explode('eg')

    return df


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. normalise the emoticons.
    2. normalise the spacings.
    3. normalise grammatical errors.
    :param df:
    :return:
    """

    # =============== normalise emoticons and shorts =============== #
    df['eg'] = df['eg'].apply(lambda r: re.sub('\.*!+', '!', r))  # (....)! match
    df['eg'] = df['eg'].apply(lambda r: re.sub('\.*\?+', '?', r))  # (....)? match
    df['eg'] = df['eg'].apply(lambda r: re.sub('\.+', '.', r))  # (....). match
    df['eg'] = df['eg'].apply(lambda r: re.sub(',+', ',', r))  # (,,,,), match
    # ㄱ-ㅎ이 따로 쓰일 경우를 대비해 ㄱ-ㅎ을 매칭시키게했었으나
    # ㅋ가 3번 사용한경우는 emoticon_normalise 에 걸리지 않아 자음 단독으로 쓰인 경우도 제거
    df['eg'] = df['eg'].apply(lambda r: re.sub('[^A-Za-z0-9가-힣\s\[\].,!?\"\']', '', r))

    df['eg'] = df['eg'].apply(lambda r: emoticon_normalize(only_text(r), num_repeats=1))

    # ===================== normalise spacing ===================== #
    df['eg'] = df['eg'].apply(lambda r: re.sub('\s+', ' ', r))  # multiple spacing match

    # ==================== grammar error check +=================== #

    df['eg'] = df['eg'].apply(lambda r: check_grammar(r))  # grammar check

    return df


def upsample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    counts = df.groupby(by='wisdom').count().sort_values(by='eg', ascending=False)['eg']
    major_count = counts.values[0]
    major_wisdom = counts.index[0]

    # Upsample minority class
    total_df = df.loc[df['wisdom'] == major_wisdom]
    for wis, ct in counts[1:].items():
        df_minority_upsampled = resample(df[df['wisdom'] == wis],
                                         replace=True,  # sample with replacement
                                         n_samples=major_count,  # to match majority class
                                         random_state=seed)  # reproducible results

        total_df = total_df.append(df_minority_upsampled)

    return total_df


def cleanse(df: pd.DataFrame) -> pd.DataFrame:
    """
    사용자가 입력하지 않을만한 것들 - 모델에게 혼란을 줄 수 있는 부분은 다 전처리를 진행하기.
    e.g. 올해 인수합병(M & A) 시장.. -> 올해 인수합병 시장
    :param df:
    :return:
    """
    # TODO: implement cleansing
    return df


def stratified_split(df: pd.DataFrame, ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    stratified-split the given df into two df's.
    :param df:
    :param ratio:
    :param seed:
    :return:
    """
    total = len(df)
    ratio_size = int(total * ratio)
    other_size = total - ratio_size
    ratio_df, other_df = train_test_split(df, train_size=ratio_size,
                                          stratify=df['wisdom'],
                                          test_size=other_size, random_state=seed,
                                          shuffle=True)
    return ratio_df, other_df
