# -*- coding: utf-8 -*-
"""
출력층과 손실 함수.

[학생 구현 대상]
- SoftmaxWithLoss.forward
- SoftmaxWithLoss.backward

Softmax와 Cross Entropy를 하나의 층으로 묶었습니다.
따로 미분하면 Softmax의 Jacobian이 복잡해지지만, 묶으면 gradient가 (확률 - 정답)으로 단순해집니다.
"""

import numpy as np


class SoftmaxWithLoss:
    """
    Softmax + Cross Entropy 출력층.

    forward는 모델이 낸 logits와 정답 레이블을 함께 받아 손실 하나(스칼라)를 반환합니다.
    backward는 인자를 받지 않습니다. forward에서 저장해 둔 확률과 정답만으로 gradient를
    만들 수 있기 때문입니다.
    """

    def __init__(self):
        self.y_prob = None
        self.y_true = None

    def forward(self, logits, y_true):
        """
        Args:
            logits: (batch_size, num_classes) 모델의 마지막 Affine 출력. 확률이 아닙니다.
            y_true: (batch_size,) 정답 레이블. 0~9 정수입니다.

        Returns:
            batch 평균 cross entropy 손실 (스칼라)

        구현 순서:
            1) 수치 안정성을 위해 각 행에서 그 행의 최댓값을 뺍니다.
               (그래도 softmax 결과는 수학적으로 동일합니다. 왜 그런지 생각해 보세요.)
            2) exp를 취하고 행 방향 합으로 나눠 확률을 만듭니다. 각 행의 합은 1이 됩니다.
            3) 확률을 self.y_prob에, 정답을 self.y_true에 저장합니다. backward에서 씁니다.
            4) 정답 클래스 확률의 log를 취해 부호를 바꾸고 batch 평균을 냅니다.
               log(0)을 피하려면 np.clip(확률, 1e-7, 1.0)처럼 아주 작은 하한을 둡니다.
               주의: log(확률 + 1e-7)처럼 더하는 방식은 backward 식과 미세하게 달라져
               gradient check(허용 상대 오차 1e-6)가 1e-5 수준의 오차로 실패할 수 있습니다.

        힌트: 정답 클래스 확률만 골라내려면 y_prob[np.arange(batch_size), y_true]를 씁니다.

        검증: 모든 logits가 0이면 확률이 모두 1/10이므로 손실은 ln(10) = 2.3026 이 됩니다.
        """
        # TODO: 위 1~4 순서대로 구현하세요.
        # overflow 방지를 위해 각 행의 최댓값을 빼줌 (확률 결과는 동일)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.y_prob = probs  # backward에서 (확률 - 정답) 계산용 저장
        self.y_true = y_true
        batch_size = logits.shape[0]
        # 각 샘플의 정답 클래스 확률만 추출 후 log(0) 방지를 위해 clip
        correct_probs = self.y_prob[np.arange(batch_size), y_true]
        loss = -np.mean(np.log(np.clip(correct_probs, 1e-7, 1.0)))
        return loss
        
    def backward(self):
        """
        Returns:
            dx: (batch_size, num_classes) logits에 대한 gradient

        Softmax와 Cross Entropy를 묶으면 gradient는 다음과 같이 단순해집니다.

            dx = (예측 확률 - 정답 one-hot) / batch_size

        구현 순서:
            1) self.y_prob를 복사합니다. (원본을 수정하면 안 됩니다)
            2) 각 샘플의 정답 클래스 위치에서 1을 뺍니다.
            3) batch_size로 나눕니다. forward에서 batch 평균을 냈으므로 여기서도 나눠야 합니다.

        검증: dx의 전체 합은 0에 가까워야 합니다. 왜 그런지 설명할 수 있어야 합니다.
        """
        # TODO: 위 1~3 순서대로 구현하세요.
        raise NotImplementedError("SoftmaxWithLoss.backward를 구현하세요.")
