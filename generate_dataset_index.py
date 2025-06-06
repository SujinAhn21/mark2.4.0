# generate_dataset_index.py

import os
import sys
import glob
import csv
import pickle
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

# === 경로 설정 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(BASE_DIR, "utils")
sys.path.append(UTILS_DIR)

from seed_utils import set_seed
from autoNor_utils import normalize_label
from vild_config import AudioViLDConfig


def get_filename_keyword_map_for_version(config: AudioViLDConfig):
    mark_version = config.mark_version
    labeled = config.labeled_classes
    unlabeled = config.unlabeled_class_identifier

    keyword_map = {
        "mark2.1.3": {
            "thumping": labeled[0],
            "others": labeled[1],
            "unlabeled": unlabeled
        },
        "mark2.2.0": {
            "water": labeled[0],
            "others": labeled[1],
            "unlabeled": unlabeled
        },
        "mark2.3.0": {
            "construction": labeled[0],
            "others": labeled[1],
            "unlabeled": unlabeled
        },
        "mark2.4.0": {
            "daily_human": labeled[0],
            "others": labeled[1],
            "unlabeled": unlabeled
        }
    }

    if mark_version not in keyword_map:
        raise ValueError(f"No keyword map defined for mark_version '{mark_version}'")

    return keyword_map[mark_version]


def save_csv(entries, output_path):
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(entries)


def save_pkl(entries, output_path):
    data = [{"path": path, "label": label} for path, label in entries]
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


def plot_label_distribution(label_count: dict, mark_version: str, save_dir: str = "plots"):
    os.makedirs(save_dir, exist_ok=True)
    labels = list(label_count.keys())
    counts = [label_count[label] for label in labels]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, counts, color='skyblue')
    plt.title(f"Label Distribution - {mark_version}")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"label_dist_{mark_version}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[완료] 라벨 분포 시각화 저장됨: {save_path}")


def generate_index(mark_version: str, seed_value: int = 42):
    set_seed(seed_value)
    config = AudioViLDConfig(mark_version=mark_version)

    current_data_dir = os.path.join(BASE_DIR, "data_wav")
    if not os.path.isdir(current_data_dir):
        raise FileNotFoundError(f"[ERROR] Data directory not found: {current_data_dir}")

    output_csv_path = os.path.join(BASE_DIR, f"dataset_index_{mark_version}.csv")
    output_pkl_path = os.path.join(BASE_DIR, f"dataset_index_{mark_version}.pkl")
    filename_keyword_map = get_filename_keyword_map_for_version(config)

    print(f"\n[INFO] Checking data directory: {current_data_dir}")
    print(f"--- Generating dataset index for: {mark_version} ---")
    print(f"  Data source directory: {current_data_dir}")
    print(f"  Output CSV file: {os.path.basename(output_csv_path)}")
    print(f"  Output PKL file: {os.path.basename(output_pkl_path)}")

    audio_paths = sorted(glob.glob(os.path.join(current_data_dir, "*.wav")))
    entries = []
    found_labels_in_data = set()
    label_count = defaultdict(int)

    for path in audio_paths:
        basename = os.path.basename(path).lower()
        matched_label = None

        for keyword in sorted(filename_keyword_map, key=len, reverse=True):
            if keyword in basename:
                label = filename_keyword_map[keyword]
                label = normalize_label(label)
                try:
                    _ = config.get_class_index(label)
                    matched_label = label
                    break
                except ValueError:
                    print(f"[Warning] Invalid label '{label}' in config.")

        if matched_label in config.labeled_classes:
            entries.append((path.replace("\\", "/"), matched_label))
            found_labels_in_data.add(matched_label)
            label_count[matched_label] += 1
        elif matched_label == config.unlabeled_class_identifier:
            continue
        else:
            print(f"[Notice] Skipping unrecognized file: {basename}")

    entries.sort(key=lambda x: x[0])

    # 저장
    save_csv(entries, output_csv_path)
    save_pkl(entries, output_pkl_path)

    # 요약 출력
    print(f"\nSaved {len(entries)} entries to:")
    print(f"  - CSV: {output_csv_path}")
    print(f"  - PKL: {output_pkl_path}")

    print("\n--- Summary ---")
    expected_labels = set(config.labeled_classes)
    missing_labels = expected_labels - found_labels_in_data
    if missing_labels:
        print("[Warning] Missing expected labels:")
        for label in sorted(missing_labels):
            print(f"  - {label}")

    unexpected_labels = found_labels_in_data - expected_labels
    if unexpected_labels:
        print("[Warning] Unexpected labels found in data:")
        for label in sorted(unexpected_labels):
            print(f"  - {label}")

    print(f"\nTotal .wav files scanned: {len(audio_paths)}")
    print(f"Total labeled entries written: {len(entries)}")
    print("\n[Label Distribution]")
    for label in sorted(label_count.keys()):
        print(f"  {label}: {label_count[label]} files")

    # 시각화 추가
    plot_label_distribution(label_count, mark_version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mark_version', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    generate_index(
        mark_version=args.mark_version,
        seed_value=args.seed
    )
  