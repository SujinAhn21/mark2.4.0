# teacher_train.py

import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import argparse

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(BASE_DIR, "utils")
sys.path.append(UTILS_DIR)

# 내부 모듈 import
from vild_utils import normalize_mel_shape
from vild_config import AudioViLDConfig
from vild_model import SimpleAudioEncoder, ViLDTextHead
from vild_parser_teacher import AudioParser
from vild_losses import ViLDLosses
from seed_utils import set_seed

class LabeledAudioDataset(Dataset):
    def __init__(self, file_label_list, parser, config):
        self.samples = []
        self.parser = parser
        self.config = config
        valid_labels = set(config.get_classes_for_text_prompts())

        for path, label in file_label_list:
            if label in valid_labels:
                try:
                    segments, _ = parser.parse_sample(path, label)
                    for seg in segments:
                        seg = normalize_mel_shape(seg)
                        if seg is not None:
                            self.samples.append((seg, label))
                except Exception as e:
                    print(f"[ERROR] Failed to parse {path}: {e}")

        if not self.samples:
            print("[Warning] No valid audio segments found in dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def custom_collate(batch):
    mels, labels = zip(*batch)
    mels = torch.stack(mels, dim=0)
    return mels, labels

def train_teacher(seed_value=42, mark_version="mark2.4.0"):
    set_seed(seed_value)
    config = AudioViLDConfig(mark_version=mark_version)
    parser = AudioParser(config, segment_mode=True)  # 선택적 세그먼트 모드 적용
    device = config.device

    csv_path = os.path.join(BASE_DIR, f"dataset_index_{mark_version}.csv")
    file_label_list = []

    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_label_list.append((row["path"], row["label"]))
    except FileNotFoundError:
        print(f"[ERROR] {csv_path} not found.")
        return

    full_dataset = LabeledAudioDataset(file_label_list, parser, config)
    if len(full_dataset) == 0:
        print("[ERROR] Dataset is empty. Check input CSV or parser.")
        return

    val_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed_value))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate)

    text_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    classes = config.get_classes_for_text_prompts()
    prompt_texts = [f"A sound of {cls.replace('_', ' ')} in the room" for cls in classes]
    text_emb = torch.tensor(text_model.encode(prompt_texts), dtype=torch.float).to(device)
    label_map = config.get_target_label_map()

    teacher_encoder = SimpleAudioEncoder(config).to(device)
    teacher_classifier = ViLDTextHead(config).to(device)
    optimizer = optim.Adam(list(teacher_encoder.parameters()) + list(teacher_classifier.parameters()), lr=config.learning_rate)
    loss_fn = ViLDLosses(config)

    best_val_loss = float('inf')
    patience = 2
    trigger_times = 0
    train_loss_history, val_loss_history = [], []

    print(f"[INFO] Teacher training started for {mark_version} on {device}")

    for epoch in range(config.num_epochs):
        teacher_encoder.train()
        teacher_classifier.train()
        total_train_loss = 0.0

        for mel_batch, label_batch in train_loader:
            mel = mel_batch.to(device)
            if mel.dim() == 5:
                mel = mel.squeeze(1)
            elif mel.dim() == 4 and mel.shape[1] != 1:
                mel = mel.unsqueeze(1)

            targets = torch.tensor([label_map[label] for label in label_batch], dtype=torch.long).to(device)

            region_emb = teacher_encoder(mel)
            logits = teacher_classifier(region_emb, text_emb)
            loss = loss_fn.compute_text_loss(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        teacher_encoder.eval()
        teacher_classifier.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for mel_batch, label_batch in val_loader:
                mel = mel_batch.to(device)
                if mel.dim() == 5:
                    mel = mel.squeeze(1)
                elif mel.dim() == 4 and mel.shape[1] != 1:
                    mel = mel.unsqueeze(1)

                targets = torch.tensor([label_map[label] for label in label_batch], dtype=torch.long).to(device)
                region_emb = teacher_encoder(mel)
                logits = teacher_classifier(region_emb, text_emb)
                loss = loss_fn.compute_text_loss(logits, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}", flush = True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(teacher_encoder.state_dict(), f"best_teacher_encoder_{mark_version}.pth")
            torch.save(teacher_classifier.state_dict(), f"best_teacher_classifier_{mark_version}.pth")
            print("[INFO] Model improved. Saved.")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"[INFO] Early stopping counter: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("[INFO] Early stopping triggered.")
                break

    torch.save(teacher_encoder.state_dict(), f"teacher_encoder_{mark_version}.pth")
    torch.save(teacher_classifier.state_dict(), f"teacher_classifier_{mark_version}.pth")
    torch.save({
        "model": teacher_encoder.state_dict(),
        "head": teacher_classifier.state_dict()
    }, f"teacher_checkpoint_{mark_version}.pt")
    print(f"[INFO] Saved teacher_checkpoint_{mark_version}.pt for soft label extraction.")

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Teacher Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/loss_curve_teacher_train_val_{mark_version}.png")
    print("[INFO] Loss curve saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mark_version", type=str, default="mark2.4.0")
    args = parser.parse_args()
    train_teacher(seed_value=42, mark_version=args.mark_version)
