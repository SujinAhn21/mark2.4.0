# vild_parser_teacher.py

import torch
import torchaudio
import torchaudio.transforms as T
import os
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from parser_utils import load_audio_file
from vild_utils import normalize_mel_shape


class AudioParser:
    def __init__(self, config, segment_mode=False):
        self.config = config
        self.segment_mode = segment_mode
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.fft_size,
            hop_length=config.hop_length,
            n_mels=config.n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        self.resampler_cache = {}

        try:
            torchaudio.set_audio_backend("soundfile")
        except RuntimeError:
            pass

    def get_all_audio_files(self):
        audio_dir = self.config.audio_dir
        return sorted(glob.glob(os.path.join(audio_dir, "*.wav")))

    def load_and_segment(self, file_path):
        waveform = load_audio_file(file_path, self.config.sample_rate, self.resampler_cache)
        if waveform is None or waveform.numel() == 0:
            print(f"[Parser] Skipped unreadable file: {file_path}")
            return []

        if not self.segment_mode:
            try:
                mel = self.mel_transform(waveform)
                mel_db = self.amplitude_to_db(mel)
                mel_tensor = normalize_mel_shape(mel_db)
                return [mel_tensor] if mel_tensor is not None else []
            except Exception as e:
                print(f"[Parser] Mel 변환 오류 (non-segment mode): {e}")
                return []

        # Segment mode
        total_samples = waveform.size(1)
        segment_samples = self.config.segment_samples
        min_len = int(self.config.segment_duration * 0.1 * self.config.sample_rate)
        segments = []

        for start in range(0, total_samples, segment_samples):
            end = start + segment_samples
            chunk = waveform[:, start:end]

            if chunk.size(1) < min_len:
                continue
            if chunk.size(1) < segment_samples:
                pad = torch.zeros(1, segment_samples - chunk.size(1), device=waveform.device)
                chunk = torch.cat([chunk, pad], dim=1)

            try:
                mel = self.mel_transform(chunk)
                mel_db = self.amplitude_to_db(mel)
                mel_tensor = normalize_mel_shape(mel_db)
                if mel_tensor is not None:
                    segments.append(mel_tensor)
            except Exception as e:
                print(f"[Parser] Segment 오류 ({file_path}): {e}")
                continue

        if len(segments) == 0:
            print(f"[Parser] No valid segments from: {file_path}")
            return []

        max_segments = getattr(self.config, "max_segments", 5)
        if len(segments) > max_segments:
            segments = segments[:max_segments]
        elif len(segments) < max_segments:
            last_valid = segments[-1]
            segments += [last_valid.clone()] * (max_segments - len(segments))

        return segments

    def parse_sample(self, file_path, label_text):
        segments = self.load_and_segment(file_path)
        segments = [seg for seg in segments if isinstance(seg, torch.Tensor)]

        if not segments:
            raise ValueError(f"[Parser] No mel segments from {file_path}")

        try:
            mel_tensor = torch.cat(segments, dim=0)
        except Exception as e:
            print(f"[Parser] concat error ({file_path}): {e}")
            for i, t in enumerate(segments):
                print(f"  Segment {i} -> shape: {getattr(t, 'shape', None)}")
            raise

        label_idx = self.config.get_class_index(label_text)
        return mel_tensor, label_idx
