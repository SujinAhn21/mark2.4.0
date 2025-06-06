# train.py
import argparse
import os
import sys

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(BASE_DIR, "utils")
sys.path.append(UTILS_DIR)

# 유틸 및 학습 모듈 import
from seed_utils import set_seed
from teacher_train import train_teacher
from student_train import train_student

def main():
    parser = argparse.ArgumentParser(description="Train Teacher or Student model.")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['teacher', 'student'],
        required=True,
        help="학습 모드 선택 (teacher 또는 student)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="전역 랜덤 시드 값"
    )
    parser.add_argument(
        '--mark_version',
        type=str,
        default="mark2.4.0",
        help="모델 및 데이터셋 버전 (예: mark2.4.0)"
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if args.mode == "teacher":
        print(f"[INFO] ViLD-text Teacher 모델 ({args.mark_version}) 학습을 시작합니다. (Seed: {args.seed})")
        train_teacher(seed_value=args.seed, mark_version=args.mark_version)

    elif args.mode == "student":
        print(f"[INFO] ViLD-image Student 모델 ({args.mark_version}) 학습을 시작합니다. (Seed: {args.seed})")
        train_student(seed_value=args.seed, mark_version=args.mark_version)

if __name__ == "__main__":
    main()
