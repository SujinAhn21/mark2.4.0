# eval.py

import os
import sys
import csv
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from vild_config import AudioViLDConfig
from vild_model import SimpleAudioEncoder
from vild_head import ViLDHead
from vild_parser_student import AudioParser
from seed_utils import set_seed


def evaluate(audio_label_list, seed_value=42, mark_version="mark2.4.0"):
    set_seed(seed_value)
    config = AudioViLDConfig(mark_version=mark_version)
    parser = AudioParser(config)
    device = config.device

    class_names = config.classes
    idx_to_label = {i: label for i, label in enumerate(class_names)}
    label_to_idx = {label: i for i, label in enumerate(class_names)}

    encoder = SimpleAudioEncoder(config).to(device)
    head = ViLDHead(config.embedding_dim, len(class_names)).to(device)

    encoder_path = f"best_student_encoder_{mark_version}.pth"
    head_path = f"best_student_head_{mark_version}.pth"

    if not os.path.exists(encoder_path) or not os.path.exists(head_path):
        print(f"[ERROR] 모델 파일 없음: {encoder_path} / {head_path}")
        return

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    head.load_state_dict(torch.load(head_path, map_location=device))
    encoder.eval()
    head.eval()

    y_true, y_pred, paths = [], [], []

    for path, true_label in audio_label_list:
        if true_label not in label_to_idx:
            continue
        true_idx = label_to_idx[true_label]
        segments = parser.load_and_segment(path)
        if not segments:
            continue

        total_score = torch.zeros(len(class_names), device=device)
        valid_segments = 0

        with torch.no_grad():
            for seg in segments:
                if seg is None or seg.ndim not in (3, 4):
                    print(f"[Skip] 잘못된 segment: {path}")
                    continue
                if seg.ndim == 3:
                    seg = seg.unsqueeze(0)  # [1, 1, 64, 101]

                seg = seg.to(device)
                feat = encoder(seg)
                logits = head(feat)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                total_score += probs
                valid_segments += 1

        if valid_segments == 0 or total_score.sum() == 0:
            print(f"[WARN] 무효 segment/score: {path}")
            continue

        avg_score = total_score / valid_segments
        pred_idx = torch.argmax(avg_score).item()

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        paths.append(path)

        print(f"[Eval] {os.path.basename(path)} | True: {true_label} | Pred: {idx_to_label[pred_idx]}")

    if not y_true:
        print("[WARN] 평가 가능한 예측 없음.")
        return

    # Plot 저장 경로
    plot_dir = os.path.join(BASE_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f"Confusion Matrix (Student - {mark_version})")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix_student_{mark_version}.png"))
    plt.close()
    print("[INFO] Confusion matrix 저장 완료.")

    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(class_names)))
    )

    # 경고 출력
    for i, (p, r) in enumerate(zip(precision, recall)):
        if p == 0.0 and r == 0.0:
            print(f"[주의] 클래스 '{class_names[i]}'에 대해 예측이 없습니다.")

    # Bar Plot
    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1, width, label='F1-Score')
    ax.set_ylabel('Score')
    ax.set_title('Evaluation Metrics by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'metrics_bar_plot_{mark_version}.png'))
    plt.close()
    print("[INFO] Metrics plot 저장 완료.")

    # CSV 저장
    csv_save_path = os.path.join(plot_dir, f'pred_results_{mark_version}.csv')
    with open(csv_save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'True Label', 'Predicted Label'])
        for path, t, p in zip(paths, y_true, y_pred):
            writer.writerow([os.path.basename(path), idx_to_label[t], idx_to_label[p]])
    print(f"[INFO] 예측 결과 CSV 저장 완료: {csv_save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mark_version', type=str, default="mark2.4.0")
    args = parser.parse_args()

    config = AudioViLDConfig(mark_version=args.mark_version)
    csv_path = os.path.join(BASE_DIR, f"dataset_index_{args.mark_version}.csv")

    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = [(row['path'], row['label']) for row in reader if row['label'] in config.classes]

        samples = []
        per_class_max = 30
        class_counter = defaultdict(int)
        for path, label in data:
            if class_counter[label] < per_class_max:
                samples.append((path, label))
                class_counter[label] += 1

        if not samples:
            print("[ERROR] 평가 샘플 없음.")
        else:
            evaluate(samples, seed_value=42, mark_version=args.mark_version)

    except Exception as e:
        print(f"[ERROR] 평가 중 예외 발생: {e}")
