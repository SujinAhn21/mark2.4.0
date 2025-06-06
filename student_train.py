# student_train.py (CrossEntropyLoss 기반)
'''세그먼트 수 불일치 해결을 위한 collect_fn 함수 수정'''

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import functools # 답답해서 로그 강제 출력..
print = functools.partial(print, flush = True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(BASE_DIR, "utils")
sys.path.append(UTILS_DIR)

from vild_config import AudioViLDConfig
from vild_model import SimpleAudioEncoder
from vild_head import ViLDHead
from vild_parser_student import AudioParser
from seed_utils import set_seed
from vild_utils import normalize_mel_shape


def load_hard_labels(config, mark_version):
    path = os.path.join(BASE_DIR, f"hard_labels_{mark_version}.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)

    samples = []
    for entry in data:
        file_path = entry["path"]
        labels = torch.tensor(entry["hard_labels"], dtype=torch.long)  # [K]
        samples.append((file_path, labels))
    return samples

'''
<<세그먼트 수 불일치 해결>>
- 최대 K 값 계산
- 세그먼트 수 불일치 해결: 부족시 zero padding.
=> 실험 일관성 유지(제발 이게 마지막이었으면..)

'''

def collate_fn(batch):
    mel_list, label_list = [], []

    for path, label_tensor in batch:
        segments = parser.load_and_segment(path)
        if not segments:
            continue

        normed_segments = []
        for seg in segments: 
            normed = normalize_mel_shape(seg)
            if normed is not None:
                normed_segments.append(normed)
        
        if len(normed_segments) == 0:
            continue
        
        mel_tensor = torch.stack(normed_segments)  # [K, 1, 64, 101]
        mel_list.append(mel_tensor)
        label_list.append(label_tensor)
        
    if not mel_list:
        return torch.empty(0), torch.empty(0)
        
    # (1) 모든 샘플 중 최대 K를 계산산
    max_k = max(mel.shape[0] for mel in mel_list)
        
    # (2) K 맞추기: 부족하면 zero-padding 추가
    for i in range(len(mel_list)):
        mel = mel_list[i]
        label = label_list[i]
            
        cur_k = mel.shape[0]
        if cur_k < max_k:
            pad = torch.zeros((max_k - cur_k, 1, 64, 101))
            mel_list[i] = torch.cat([mel, pad], dim=0)
                
            label_pad = torch.full((max_k - cur_k,), -1, dtype=torch.long) # -1은 ignore_index로 추론 제외 가능
            label_list[i] = torch.cat([label, label_pad], dim=0)
        
        elif cur_k > max_k:
            mel_list[i] = mel[:max_k]
            label_list[i] = label[:max_k]
           
    return torch.stack(mel_list), torch.stack(label_list)  # [B, K, 1, 64, 101], [B, K]


def train_student(seed_value=42, mark_version="mark2.4.0"):  
    set_seed(seed_value)
    config = AudioViLDConfig(mark_version=mark_version)
    global parser
    parser = AudioParser(config)

    device = torch.device(config.device)
    samples = load_hard_labels(config, config.mark_version)

    X, y = zip(*samples)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    train_data = list(zip(X_train, y_train))
    val_data = list(zip(X_val, y_val))

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    encoder = SimpleAudioEncoder(config).to(device)
    head = ViLDHead(config.embedding_dim, len(config.classes)).to(device)
    model = nn.Sequential(encoder, nn.Flatten(start_dim=1), head).to(device)

    # criterion = nn.CrossEntropyLoss() 
    criterion = nn.CrossEntropyLoss(ignore_index=-1) # 수정. 원래 기본적으로 ignore_index가 -100이지만, 직관성을 위해.
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")
    train_loss_history = []
    val_loss_history = []

    print(f"[INFO] Student training started for {mark_version} on {device}")

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

        for mel_batch, label_batch in tqdm(train_loader, desc=f"[Train Epoch {epoch+1}]"):
            if mel_batch.numel() == 0:
                continue

            B, K, C, H, W = mel_batch.shape
            mel_batch = mel_batch.view(B * K, C, H, W).to(device)
            label_batch = label_batch.view(B * K).to(device)

            output = model(mel_batch)
            loss = criterion(output, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mel_batch, label_batch in val_loader:
                if mel_batch.numel() == 0:
                    continue

                B, K, C, H, W = mel_batch.shape
                mel_batch = mel_batch.view(B * K, C, H, W).to(device)
                label_batch = label_batch.view(B * K).to(device)

                output = model(mel_batch)
                loss = criterion(output, label_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(encoder.state_dict(), f"best_student_encoder_{config.mark_version}.pth")
            torch.save(head.state_dict(), f"best_student_head_{config.mark_version}.pth")
            print(f"[INFO] Best model saved (Epoch {epoch+1})")

    # Loss 그래프 저장
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Student Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/loss_curve_student_train_val_{mark_version}.png")
    print("[INFO] Student loss curve saved.")


if __name__ == "__main__":
    train_student()  
