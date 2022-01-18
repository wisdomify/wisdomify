"""
paths to data/ paths to saved models should be saved here
"""
from pathlib import Path

# --- dirs --- #
ROOT_DIR = Path(__file__).resolve().parent.parent
WANDB_DIR = ROOT_DIR / "wandb"  # for saving wandb logs
CONFIG_YAML = ROOT_DIR / "config.yaml"
CORPORA_DIR = ROOT_DIR / "corpora"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
# --- corpora dirs --- #
SC_DIR = CORPORA_DIR / "sc"  # 감성대화
MR_DIR = CORPORA_DIR / "mr"  # 기계독해
DS_DIR = CORPORA_DIR / "ds"  # 문서요약 텍스트
SFC_DIR = CORPORA_DIR / "sfc"     # 전문분야 말뭉치
KESS_DIR = CORPORA_DIR / "kess"     # 한국어-영어 번역 말뭉치 (사회과학)
KJ_DIR = CORPORA_DIR / "kj"    # 한국어-일본어 번역 말뭉치
KCSS_DIR = CORPORA_DIR / "kcss"     # 한국어-중국어 번역 말뭉치 사회과학
BS_DIR = CORPORA_DIR / "bs"  # 도서자료 요약
GK_DIR = CORPORA_DIR / "gk"  # 일반상식
SFKE_DIR = CORPORA_DIR / "sfke"  # 전문분야 한영 말뭉치
KSNS_DIR = CORPORA_DIR / "ksns"  # 한국어 SNS
KC_DIR = CORPORA_DIR / "kc"  # 한국어 대화
KETS_DIR = CORPORA_DIR / "kets"  # 한국어-영어 번역 말뭉치 (기술과학)
KEPT_DIR = CORPORA_DIR / "kept"  # 한국어-영어 번역(병렬) 말뭉치
NEWS_DIR = CORPORA_DIR / "news"  # 뉴스데이터
KOREA_UNIV_DIR = CORPORA_DIR / "korea_univ"  # 고려대 코퍼스
