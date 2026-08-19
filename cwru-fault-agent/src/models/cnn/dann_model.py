"""
src/models/cnn/dann_model.py
DANN (Domain-Adversarial Neural Network, Ganin et al. 2016).

구조: 기존 CNNClassifier의 feature_extractor(model.py)를 그대로 재사용하고,
그 뒤에 두 개의 head를 붙인다.
  1) label_classifier   : 고장유형 분류 (source 라벨로만 학습)
  2) domain_discriminator: source/target 중 어디서 왔는지 판별 (양쪽 다 라벨 없이,
                            "도메인" 자체를 라벨로 사용 - 이건 항상 알 수 있음)

핵심 트릭: Gradient Reversal Layer(GRL). 순전파 시엔 항등함수처럼 그대로 통과시키지만,
역전파 시엔 그래디언트에 -lambda를 곱해서 반전시킨다.
따라서 domain_discriminator는 "도메인을 더 잘 맞추는" 방향으로 학습되는데,
그 그래디언트가 feature_extractor에 반전되어 전달되므로 feature_extractor는 반대로
"도메인을 구분 못 하게 만드는" 방향(도메인-불변 표현)으로 학습된다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import torch
import torch.nn as nn
from torch.autograd import Function

from src.models.cnn.model import FeatureExtractor, Classifier


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainDiscriminator(nn.Module):
    """source(0) / target(1) 이진 판별기."""

    def __init__(self, in_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class DANN(nn.Module):
    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.label_classifier = Classifier(in_dim=128, n_classes=n_classes)
        self.grl = GradientReversalLayer(lambda_=0.0)  # 학습 중 스케줄링으로 값 조정
        self.domain_discriminator = DomainDiscriminator(in_dim=128)

    def forward(self, x, return_domain: bool = True):
        feat = self.feature_extractor(x)
        class_logits = self.label_classifier(feat)
        if not return_domain:
            return class_logits
        domain_logits = self.domain_discriminator(self.grl(feat))
        return class_logits, domain_logits

    def set_grl_lambda(self, lambda_: float):
        self.grl.set_lambda(lambda_)

    def load_pretrained_backbone(self, cnn_baseline_state_dict: dict):
        """train_baseline.py로 학습한 CNNClassifier의 state_dict에서
        feature_extractor 가중치만 가져와 초기화 (전이학습의 출발점으로 활용)."""
        fe_state = {k.replace("feature_extractor.", ""): v
                    for k, v in cnn_baseline_state_dict.items()
                    if k.startswith("feature_extractor.")}
        missing, unexpected = self.feature_extractor.load_state_dict(fe_state, strict=True)
        return missing, unexpected


if __name__ == "__main__":
    model = DANN(n_classes=4)
    dummy = torch.randn(8, 8192)
    class_logits, domain_logits = model(dummy)
    print("class_logits:", class_logits.shape, "(batch, n_classes)여야 함")
    print("domain_logits:", domain_logits.shape, "(batch, 2)여야 함")

    # GRL 동작 확인: lambda=1일 때 backward 그래디언트 부호가 반전되는지
    model.set_grl_lambda(1.0)
    x = torch.randn(4, 8192, requires_grad=False)
    feat = model.feature_extractor(x)
    feat.retain_grad()
    out = model.domain_discriminator(model.grl(feat))
    loss = out.sum()
    loss.backward()
    print("\nGRL 그래디언트 반전 테스트: feat.grad 부호가 반전되어 있어야 함 (lambda=1)")
    print("feat.grad 일부:", feat.grad.flatten()[:5])