"""
src/models/transfer/domain_adapt.py
비지도 도메인 적응(unsupervised domain adaptation) 특징 정렬 방법 2종.

두 방법 모두 target 도메인의 "라벨 없는" 데이터(target_adaptation)만 사용한다.
target_test는 최종 평가에만 쓰고 적응 과정에는 절대 사용하지 않는다 (데이터 누수 방지).

1) CORAL (Sun & Saenko, 2016 "Correlation Alignment for Unsupervised Domain Adaptation")
   - source 특징의 2차 통계량(공분산)을 target 공분산에 맞춰 재정렬(whitening + recoloring).
   - 분류기는 정렬된 source로 학습, target은 원본 그대로 평가.

2) TCA (Pan et al., 2011 "Domain Adaptation via Transfer Component Analysis")
   - source/target을 함께 투영해 두 도메인 간 MMD(Maximum Mean Discrepancy)를
     최소화하는 저차원 공유 표현(latent space)을 학습.
   - 여기서는 선형 커널 기반의 간소화 구현을 사용 (계산량 절약을 위해 source는
     클래스 균형 서브샘플링 후 커널 행렬 구성, 이후 out-of-sample 투영으로 전체 데이터에 적용).
"""
import numpy as np
from scipy.linalg import fractional_matrix_power, eigh


# =================================================================
# CORAL
# =================================================================
def coral_transform(X_source: np.ndarray, X_target_unlabeled: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    X_source: (ns, d) source 도메인 특징 (라벨 있음, 학습에 사용될 데이터)
    X_target_unlabeled: (nt, d) target 도메인 특징 (라벨 없음, 통계량 추정에만 사용 = target_adaptation)

    반환: X_source와 동일한 shape. source 데이터를 target의 공분산 구조에 맞게 재정렬한 결과.
    이후 이 정렬된 데이터로 분류기를 학습하고, target_test는 변환 없이(raw) 그대로 평가한다.
    """
    Xs = X_source.astype(np.float64)
    Xt = X_target_unlabeled.astype(np.float64)

    mu_s, mu_t = Xs.mean(axis=0), Xt.mean(axis=0)
    Xs_c = Xs - mu_s
    Xt_c = Xt - mu_t

    d = Xs.shape[1]
    Cs = np.cov(Xs_c, rowvar=False) + eps * np.eye(d)
    Ct = np.cov(Xt_c, rowvar=False) + eps * np.eye(d)

    Cs_inv_sqrt = fractional_matrix_power(Cs, -0.5).real
    Ct_sqrt = fractional_matrix_power(Ct, 0.5).real

    # whitening(source 공분산 제거) -> recoloring(target 공분산 부여) -> target 평균으로 이동
    Xs_aligned = Xs_c @ Cs_inv_sqrt @ Ct_sqrt + mu_t
    return Xs_aligned


# =================================================================
# TCA (선형 커널, 원본 특징공간에서 직접 계산하는 안정화된 구현)
# =================================================================
class LinearTCA:
    """
    선형 커널 TCA. 특징 차원 d가 표본 수보다 훨씬 작을 때
    (d x d) 일반화 고유값 문제로 등가 변환하여 계산한다.

    수학적 근거: 선형 커널에서 MMD 계수행렬 L에 대해
        X^T L X = (mu_source - mu_target)(mu_source - mu_target)^T   (rank-1, d x d)
        X^T H X = 전체 데이터의 산포행렬(scatter matrix), H는 중심화 행렬  (d x d)
    이 되므로, (n x n) 커널 행렬을 만들 필요 없이 바로 d x d 문제로 풀 수 있다.
    표본 수가 많아도 특징 차원(d=15)이 작아 매우 빠르고 안정적이다.
    """

    def __init__(self, n_components: int = 8, mu: float = 1.0):
        self.n_components = n_components
        self.mu = mu
        self._W = None          # 투영행렬 (d, n_components)
        self._overall_mean = None

    def fit(self, X_source: np.ndarray, X_target_unlabeled: np.ndarray, y_source: np.ndarray = None,
            random_state: int = 42):
        Xs = X_source.astype(np.float64)
        Xt = X_target_unlabeled.astype(np.float64)
        d = Xs.shape[1]

        mu_s, mu_t = Xs.mean(axis=0), Xt.mean(axis=0)
        diff = (mu_s - mu_t).reshape(-1, 1)
        M = diff @ diff.T  # X^T L X, rank-1 (d x d)

        X_all = np.vstack([Xs, Xt])
        overall_mean = X_all.mean(axis=0)
        Xc = X_all - overall_mean
        St = Xc.T @ Xc  # X^T H X (총 산포행렬, d x d)

        A = M + self.mu * np.eye(d)
        B = St + 1e-3 * np.trace(St) / d * np.eye(d)  # 산포행렬 크기에 비례한 정칙화

        eigvals, eigvecs = eigh(A, B)
        order = np.argsort(eigvals)  # TCA는 min-trace 문제 -> 가장 작은 고유값 선택
        n_comp = min(self.n_components, d)
        W = eigvecs[:, order[:n_comp]]

        self._W = W.real
        self._overall_mean = overall_mean
        return self

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        X_new = X_new.astype(np.float64)
        return (X_new - self._overall_mean) @ self._W