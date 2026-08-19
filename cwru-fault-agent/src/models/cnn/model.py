"""
src/models/cnn/model.py
1D-CNN 베어링 고장진단 모델.

feature_extractor와 classifier를 분리해서 정의한 이유:
DANN 구현 시 동일한 feature_extractor 뒤에 (1) 기존 classifier와
(2) 도메인 판별기(domain discriminator)를 각각 붙이는 구조가 되기 때문.
지금(베이스라인) 단계에서는 classifier만 사용한다.
"""
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """원신호(1, 8192) -> 128차원 임베딩 벡터."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (batch, 128, 1) - 입력 길이에 무관하게 고정 출력
        )

    def forward(self, x):
        # x: (batch, 8192) -> (batch, 1, 8192)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = self.net(x)          # (batch, 128, 1)
        return out.squeeze(-1)     # (batch, 128)


class Classifier(nn.Module):
    def __init__(self, in_dim: int = 128, n_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNNClassifier(nn.Module):
    """베이스라인용: feature_extractor + classifier 결합."""

    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier(in_dim=128, n_classes=n_classes)

    def forward(self, x):
        feat = self.feature_extractor(x)
        return self.classifier(feat)


if __name__ == "__main__":
    # 스모크 테스트: 임의 입력으로 shape 확인
    model = CNNClassifier(n_classes=4)
    dummy = torch.randn(8, 8192)  # batch=8
    out = model(dummy)
    print("입력 shape:", dummy.shape)
    print("출력 shape:", out.shape, "(batch, n_classes)여야 함")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 수: {n_params:,}")