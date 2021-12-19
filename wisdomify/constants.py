"""
paths to data, paths to saved models should be saved here
"""
from pathlib import Path
from os import path
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# --- The directories --- #
ROOT_DIR = Path(__file__).resolve().parent.parent.__str__()
WANDB_DIR = path.join(ROOT_DIR, "wandb")  # for saving wandb logs
# --- for configuring experiments --- #
CONFIG_YAML = path.join(ROOT_DIR, "config.yaml")
CORPORA_DIR = path.join(ROOT_DIR, "corpora")
ARTIFACTS_DIR = path.join(ROOT_DIR, "artifacts")
# corpora
SC_DIR = path.join(CORPORA_DIR, "sc")  # 감성대화
MR_DIR = path.join(CORPORA_DIR, "mr")  # 기계독해
DS_DIR = path.join(CORPORA_DIR, "ds")   # 문서요약 텍스트
SFC_DIR = path.join(CORPORA_DIR, "sfc")     # 전문분야 말뭉치
KESS_DIR = path.join(CORPORA_DIR, "kess")     # 한국어-영어 번역 말뭉치 (사회과학)
KJ_DIR = path.join(CORPORA_DIR, "kj")     # 한국어-일본어 번역 말뭉치
KCSS_DIR = path.join(CORPORA_DIR, "kcss")     # 한국어-중국어 번역 말뭉치 사회과학
BS_DIR = path.join(CORPORA_DIR, "bs")  # 도서자료 요약
GK_DIR = path.join(CORPORA_DIR, "gk")  # 일반상식
SFKE_DIR = path.join(CORPORA_DIR, "sfke")  # 전문분야 한영 말뭉치
KSNS_DIR = path.join(CORPORA_DIR, "ksns")  # 한국어 SNS
KC_DIR = path.join(CORPORA_DIR, "kc")  # 한국어 대화
KETS_DIR = path.join(CORPORA_DIR, "kets")  # 한국어-영어 번역 말뭉치 (기술과학)
KEPT_DIR = path.join(CORPORA_DIR, "kept")  # 한국어-영어 번역(병렬) 말뭉치
NEWS_DIR = path.join(CORPORA_DIR, "news")  # 뉴스데이터
KOREA_UNIV_DIR = path.join(CORPORA_DIR, "korea_univ")  # 고려대 코퍼스
WOONGJIN_DIR = path.join(CORPORA_DIR, "woongjin")  # 웅진 말뭉치 데이터셋   
# --- url's --- #
WISDOM2DEF_RAW_A = "https://docs.google.com/spreadsheets/d/1n550JrAYnyy2j1CQAeXPjeuw0zD5RYNbpR4wKFTq8DI/export?format=tsv&gid=0"  # noqa
WISDOM2DEF_RAW_B = "https://docs.google.com/spreadsheets/d/1n550JrAYnyy2j1CQAeXPjeuw0zD5RYNbpR4wKFTq8DI/export?format=tsv&gid=1300142415"  # noqa
WISDOM2QUERY_RAW_A = "https://docs.google.com/spreadsheets/d/17t-WFD9e8a9VUu2nda56I_akeyYJ9RYilUCIVTra7Es/export?format=tsv&gid=1307694002"  # noqa
WISDOMS_A = "https://docs.google.com/spreadsheets/d/1--hzu43sd8nk8-R_Qf2jTZ0iuGQCIgRo3bwQ63SYVbo/export?format=tsv&gid=0"  # noqa
WISDOMS_B = "https://docs.google.com/spreadsheets/d/1--hzu43sd8nk8-R_Qf2jTZ0iuGQCIgRo3bwQ63SYVbo/export?format=tsv&gid=822745026"  # noqa
# --- wandb --- #
WANDB_ENTITY = "wisdomify"
WANDB_PROJECT = "wisdomify"

# --- es-related --- #
ES_CLOUD_ID = os.getenv("ES_CLOUD_ID")
ES_USERNAME = os.getenv("ES_USERNAME")
ES_PASSWORD = os.getenv("ES_PASSWORD")


# --- news_api --- #
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
