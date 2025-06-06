# extract_hard_labels.py

import os
import sys
import pickle
import csv
from tqdm import tqdm

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

# 설정 및 유틸 가져오기
from vild_config import AudioViLDConfig
from autoNor_utils import normalize_label
from vild_parser_student import AudioParser  # 학생 파서에서 세그먼트 수 가져옴

def extract_hard_labels(mark_version="mark2.4.0"):
    config = AudioViLDConfig(mark_version=mark_version)
    csv_path = os.path.join(BASE_DIR, f"dataset_index_{mark_version}.csv")

    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV 파일 없음: {csv_path}")
        return 1

    parser = AudioParser(config)
    hard_label_entries = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Generating hard labels"):
            path = row['path']
            label = normalize_label(row['label'])

            try:
                label_idx = config.get_class_index(label)
            except ValueError:
                print(f"[Warning] 알 수 없는 라벨: {label} ({path})")
                continue

            segment_list = parser.load_and_segment(path)

            if segment_list is None or len(segment_list) == 0:
                print(f"[Warning] 세그먼트 없음: {path}")
                continue

            num_segments = len(segment_list)
            hard_labels = [label_idx] * num_segments

            hard_label_entries.append({
                "path": path,
                "hard_labels": hard_labels
            })

    out_path = os.path.join(BASE_DIR, f"hard_labels_{mark_version}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(hard_label_entries, f)

    print(f"[완료] hard_labels 저장됨: {out_path}")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mark_version', type=str, default="mark2.4.0")
    args = parser.parse_args()
    exit(extract_hard_labels(mark_version=args.mark_version))
