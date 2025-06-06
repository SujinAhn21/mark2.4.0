# run_all.py (최종 안정화 버전 - soft label 제거 및 .pt 생성 생략)

import os
import subprocess
import logging
from datetime import datetime
import argparse

# ===== 파라미터 설정 =====
parser = argparse.ArgumentParser()
parser.add_argument("--mark_version", type=str, default="mark2.4.0", help="실행할 모델 버전")
args = parser.parse_args()
mark_version = args.mark_version

# ===== 경로 설정 =====
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "logFiles")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"run_pipeline_{mark_version}.log")

# ===== 로그 설정 =====
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ===== 데코레이터 =====
def timed_step(func):
    def wrapper(*args, **kwargs):
        step_name = func.__name__.replace("run_", "Step : ")
        logging.info(f"\n[실행 시작] --> {step_name}")
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        duration = (end - start).total_seconds()
        logging.info(f"[완료] --> {step_name} (소요시간: {duration:.2f}초)")
        return result
    return wrapper

# ===== 서브프로세스 실행 함수 =====
def run_subprocess(command_list):
    try:
        logging.info(f"[CMD] {' '.join(command_list)}")
        result = subprocess.run(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors='replace'
        )

        if result.stdout.strip():
            logging.info("[STDOUT]\n" + result.stdout.strip())
        if result.stderr.strip():
            logging.info("[STDERR]\n" + result.stderr.strip())

        return result.returncode

    except Exception as e:
        logging.error(f"[ERROR] Subprocess 실행 중 예외 발생: {e}")
        return 1

# ===== 단계별 실행 함수 정의 =====
@timed_step
def run_step0():
    return run_subprocess(["python", "fix_audio_length_to_240000.py", "--mark_version", mark_version])

@timed_step
def run_step1():
    return run_subprocess(["python", "generate_dataset_index.py", "--mark_version", mark_version])

@timed_step
def run_step2():
    return run_subprocess(["python", "teacher_train.py", "--mark_version", mark_version])

@timed_step
def run_step3():
    return run_subprocess(["python", "extract_hard_labels.py", "--mark_version", mark_version])

@timed_step
def run_step4():
    return run_subprocess(["python", "student_train.py", "--mark_version", mark_version])

@timed_step
def run_step5():
    return run_subprocess(["python", "eval.py", "--mark_version", mark_version])

@timed_step
def run_step6():
    return run_subprocess(["python", "plot_audio.py", "--mark_version", mark_version])

# ===== 메인 실행 =====
if __name__ == "__main__":
    logging.info("=== 소음 분류 전체 학습 파이프라인 시작 ===")
    logging.info(f"모델 버전: {mark_version}")
    logging.info("현재 모델은 soft label -> hard label 전환 후 CrossEntropyLoss 기반으로 학습됩니다.")

    steps = [run_step0, run_step1, run_step2, run_step3, run_step4, run_step5, run_step6]
    for step in steps:
        code = step()
        if code != 0:
            logging.error(f"[ERROR] 실패: {step.__name__} (코드 {code}) -> 이후 단계 생략됨.")
            break

    logging.info(f"\n[종료] 전체 파이프라인 완료. 로그 파일: {log_file_path}")
