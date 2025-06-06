# vild_parser_student.py

import torch
import torchaudio
import torchaudio.transforms as T
import os
import sys

# utils 경로 추가
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from parser_utils import load_audio_file
from vild_utils import normalize_mel_shape


class AudioParser:
    """
    Student 모델 학습용 오디오 파서

    주요 기능:
    - 오디오 로드 및 리샘플링
    - Mel-spectrogram 변환
    - 일정 길이의 segment로 분할 후 normalize
    - 각 segment는 [1, 64, 101] Tensor로 반환됨
    """

    def __init__(self, config):
        self.config = config
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

    def load_and_segment(self, file_path, max_segment=None):
        """
        오디오 파일을 mel-spectrogram으로 변환하고 segment 단위로 나눔

        Args:
            file_path (str): 오디오 파일 경로
            max_segment (int, optional): 최대 segment 수 (K개만 추출)

        Returns:
            List[Tensor]: [1, 64, 101] 크기의 mel segment 텐서 리스트
        """
        waveform = load_audio_file(file_path, self.config.sample_rate, self.resampler_cache)
        if waveform is None or waveform.numel() == 0:
            print(f"[Warning] Invalid waveform from: {file_path}")
            return []

        try:
            mel = self.mel_transform(waveform)        # [1, n_mels, time]
            mel_db = self.amplitude_to_db(mel)        # [1, 64, time]

            if mel_db.ndim != 3 or mel_db.shape[1] != 64:
                print(f"[Warning] Unexpected mel shape: {mel_db.shape} from {file_path}")
                return []

            _, _, total_time = mel_db.shape
            stride = self.config.segment_hop
            window = self.config.segment_length

            if total_time < window:
                print(f"[Warning] Mel too short for segmentation: {total_time} < {window} in {file_path}")
                return []

            segment_list = []
            for i, start in enumerate(range(0, total_time - window + 1, stride)):
                segment = mel_db[:, :, start:start + window]  # [1, 64, 101]
                normed = normalize_mel_shape(segment)
                if normed is not None:
                    segment_list.append(normed)
                else:
                    print(f"[Skip] Segment normalize 실패: index {i} in {file_path}")

                if max_segment is not None and len(segment_list) >= max_segment:
                    break

            if len(segment_list) == 0:
                print(f"[Warning] No valid segment for: {file_path}")

            return segment_list

        except Exception as e:
            print(f"[ERROR] Exception while parsing {file_path}: {e}")
            return []
