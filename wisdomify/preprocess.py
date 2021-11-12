from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def augment(df: pd.DataFrame) -> pd.DataFrame:
    # TODO implement augmentation.
    return df


def parse(df: pd.DataFrame) -> pd.DataFrame:
    """
    parse <em> ...</em>  to [WISDOM].
    :param df:
    :return:
    """
    # TODO: implement parsing
    return df


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. normalise the emoticons.
    2. normalise the spacings.
    3. normalise grammatical errors.
    :param df:
    :return:
    """
    # TODO: implement normalisation
    return df


def upsample(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: implement upsampling
    return df


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
