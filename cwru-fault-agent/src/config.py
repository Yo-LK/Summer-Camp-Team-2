"""
src/config.py
프로젝트 공통 경로 설정. 이 파일 위치(src/config.py) 기준으로 PROJECT_ROOT를 자동 계산하므로,
프로젝트를 어느 경로에 클론/이동해도 그대로 동작한다.
"""
from pathlib import Path

# src/config.py -> parent(src) -> parent(project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "runs"
REPORT_DIR = PROJECT_ROOT / "report"

# 실행 시 필요한 디렉터리 자동 생성
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)