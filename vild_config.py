# vild_config.py

import torch
from sentence_transformers import SentenceTransformer
import os

class AudioViLDConfig:
    def __init__(self, mark_version="mark2.1.2"):
        self.mark_version = mark_version

        # === 클래스 설정 ===
        if self.mark_version == "mark2.1.3":
            self.classes = ["thumping", "others"]
        elif self.mark_version == "mark2.2.0":
            self.classes = ["water", "others"]
        elif self.mark_version == "mark2.3.0":
            self.classes = ["construction", "others"]
        elif self.mark_version == "mark2.4.0":
            self.classes = ["daily_human", "others"]
        else:
            raise ValueError(
                f"[Error] Unknown or unsupported mark_version: '{self.mark_version}'.\n"
                f"지원되는 값: ['mark2.1.3', 'mark2.2.0', 'mark2.3.0', 'mark2.4.0']"
            )

        self.labeled_classes = self.classes
        self.unlabeled_class_identifier = "unlabeled"
        self.num_distinct_labeled_classes = len(self.labeled_classes)

        # === 오디오 파라미터 ===
        self.sample_rate = 16000
        self.segment_duration = 1.0
        self.segment_samples = int(self.sample_rate * self.segment_duration)

        self.fft_size = 1024
        self.hop_length = 160
        self.n_mels = 64

        # === segment 단위 처리 ===
        self.segment_length = 101   # Mel spectrogram time frame 수
        self.segment_hop = 50       # Segment 간 stride
        self.max_segments = 5       # Teacher가 사용할 최대 segment 수

        # === 모델 파라미터 ===
        self.embedding_dim = 384
        self.use_background_embedding = True

        # === 학습 파라미터 ===
        self.batch_size = 16
        self.num_epochs = 80 # 100에서 80으로 조정
        self.learning_rate = 1e-4

        self.text_loss_weight = 1.0
        self.image_loss_weight = 1.0

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # === 데이터 경로 ===
        self.audio_dir = os.path.join("data_wav")  # mark_version 별 하위 폴더화 가능

        # === 내부 캐시 ===
        self._text_emb = None

    def get_class_index(self, class_name: str) -> int:
        if class_name in self.labeled_classes:
            return self.labeled_classes.index(class_name)
        elif class_name == self.unlabeled_class_identifier:
            return -1
        else:
            raise ValueError(
                f"[Config Error] '{class_name}'는 mark_version '{self.mark_version}'에 등록되지 않은 클래스입니다.\n"
                f"-> 현재 사용 가능한 클래스: {self.labeled_classes}"
            )

    def get_classes_for_text_prompts(self) -> list:
        return self.labeled_classes

    def get_target_label_map(self) -> dict:
        return {class_name: i for i, class_name in enumerate(self.get_classes_for_text_prompts())}

    def get_class_text_embeddings(self) -> torch.Tensor:
        if self._text_emb is None:
            # 프롬프트 문구는 zero-shot 성능 개선 시 수정 가능
            prompts = [f"a sound of {c.replace('_', ' ')} in the room" for c in self.get_classes_for_text_prompts()]
            model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            self._text_emb = model.encode(prompts, convert_to_tensor=True).to(self.device)
        return self._text_emb
