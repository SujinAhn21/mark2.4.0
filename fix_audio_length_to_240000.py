# fix_audip_length_to_240000.py
# 여러가지 사정에 의해 데이터를 일정하게 유지하는 편이 세그먼트 난리 안날 것 같아 만들었다.
import os
import sys
import torch
import torchaudio
import argparse

# ==== 인자 parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--mark_version", type=str, default="", 
                    help="모델 버전(예: mark2.4.0)")
args = parser.parse_args()

# === 기본 경로 설정 ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_INPUT_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "data_wav")

    
# === 보정 파라미터 === """생각보다 중요한 부분분"""    
TARGET_SAMPLE_RATE = 16000 # sample rate 도 제한을 해서 주파수 해상도를 일정하게 맞추기
TARGET_NUM_SAMPLES = TARGET_SAMPLE_RATE * 15  # 240,000 samples (딱 15초기준)

def fix_wav_length(wav_path, save_path):
    waveform, sr = torchaudio.load(wav_path)

    # 샘플레이트 맞추기(보정하기)
    if sr != TARGET_SAMPLE_RATE:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
        waveform = resample(waveform)

    # 자르기 or 패딩
    # 길면 자르고 짧으면 제로패딩으로 정적으로라도 채우기
    num_samples = waveform.shape[1]
    if num_samples > TARGET_NUM_SAMPLES:
        fixed_waveform = waveform[:, :TARGET_NUM_SAMPLES]
    elif num_samples < TARGET_NUM_SAMPLES:
        pad_len = TARGET_NUM_SAMPLES - num_samples
        pad_tensor = torch.zeros((waveform.shape[0], pad_len))
        fixed_waveform = torch.cat([waveform, pad_tensor], dim=1)
    else:
        fixed_waveform = waveform

    torchaudio.save(save_path, fixed_waveform, TARGET_SAMPLE_RATE)
    print(f"저장됨: {save_path} (길이: {fixed_waveform.shape[1]} samples)")

def process_all(input_dir=None, output_dir=None):
    if input_dir is None:
        input_dir = DEFAULT_INPUT_DIR
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    
    print(f"[INFO] 오디오 길이 보정 시작: {input_dir} -> {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    for fname in os.listdir(input_dir):
        if fname.endswith(".wav"):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            fix_wav_length(in_path, out_path)
    
    print("[DONE] 모든 오디오 보정 완료.")

if __name__ == "__main__":
    process_all()