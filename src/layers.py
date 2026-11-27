# -*- coding: utf-8 -*-
"""
신경망 layer 모음.

[학생 구현 대상]
- Affine.forward
- Affine.backward

[제공 코드 - 구현하지 않습니다]
- BatchNorm, Dropout
  핵심 과제를 마친 뒤 구조를 더 살펴볼 때만 사용합니다. 기본 설정에서는 꺼져 있습니다.
"""

import numpy as np


class Affine:
    """
    완전연결층(Fully Connected Layer).

    수식은 y = x @ W + b 입니다.
    MNIST에서는 784개 픽셀을 은닉층 차원으로, 은닉층을 10개 클래스로 선형 변환합니다.

    중요: W와 b는 TwoLayerNet의 params dict와 **같은 배열 객체**를 공유합니다.
    optimizer가 params를 제자리(in-place)에서 갱신하면 이 레이어의 W도 함께 바뀝니다.
    """

    def __init__(self, W, b):
        """
        Args:
            W: (input_dim, output_dim) 가중치. params dict와 공유되는 배열입니다.
            b: (output_dim,) 편향. params dict와 공유되는 배열입니다.
        """
        self.W = W
        self.b = b
        self.x = None
        self.dW = np.zeros_like(W)
        self.db = np.zeros_like(b)

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)

        Returns:
            (batch_size, output_dim)

        예시 shape:
            x (4, 784) @ W (784, 128) -> (4, 128), 여기에 b (128,)가 각 행에 더해집니다.
        """
        # TODO 1: backward에서 다시 써야 하므로 입력 x를 self.x에 저장하세요.
        self.x = x
        # TODO 2: x @ W + b 를 계산해 반환하세요. b는 broadcasting으로 모든 행에 더해집니다.
        return x @ self.W + self.b

    def backward(self, dout):
        """
        Args:
            dout: (batch_size, output_dim) 다음 층에서 넘어온 gradient

        Returns:
            dx: (batch_size, input_dim) 이 층의 입력에 대한 gradient

        Side effects:
            self.dW: W와 같은 shape
            self.db: b와 같은 shape

        확인용 규칙:
            dW.shape == W.shape, db.shape == b.shape, dx.shape == x.shape
        """
        # TODO 1: self.dW를 계산하세요. forward에서 저장한 x와 dout을 사용합니다.
        # TODO 2: self.db를 계산하세요. bias는 batch의 모든 샘플에 더해졌으므로
        #         gradient는 batch 방향으로 합쳐야 합니다.
        # TODO 3: dx를 계산해 반환하세요.
        raise NotImplementedError("Affine.backward를 구현하세요.")


class BatchNorm:
    """
    Batch Normalization. (제공 코드 - 구현하지 않습니다)

    미니배치마다 각 feature의 평균과 분산을 맞춰 학습을 안정화합니다.
    학습 시에는 현재 배치 통계를, 추론 시에는 누적된 running 통계를 사용합니다.

    필수 구현과 검증을 마친 뒤 TwoLayerNet(use_batchnorm=True)로 동작을 관찰할 수 있습니다.
    mini-GPT에서는 이 BatchNorm 대신 LayerNorm을 사용합니다.
    """

    train_mode_layer = True

    def __init__(self, gamma, beta, momentum=0.9, eps=1e-7):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.eps = eps
        self.running_mean = np.zeros_like(beta)
        self.running_var = np.ones_like(beta)
        self.dgamma = np.zeros_like(gamma)
        self.dbeta = np.zeros_like(beta)

    def forward(self, x, train=True):
        if train:
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            self.batch_size = x.shape[0]
            self.xc = x - mu
            self.std = np.sqrt(var + self.eps)
            self.xn = self.xc / self.std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            self.xn = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * self.xn + self.beta

    def backward(self, dout):
        self.dbeta = dout.sum(axis=0)
        self.dgamma = (self.xn * dout).sum(axis=0)

        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = dxc.sum(axis=0)
        return dxc - dmu / self.batch_size


class Dropout:
    """
    Dropout. (제공 코드 - 구현하지 않습니다)

    학습 중 일부 뉴런 출력을 무작위로 0으로 만들어 과적합을 줄입니다.
    학습 때 살아남은 값을 keep 확률로 나누는 inverted dropout 방식이므로
    추론 때는 입력을 그대로 통과시킵니다.

    필수 구현과 검증을 마친 뒤 TwoLayerNet(use_dropout=True)로 동작을 관찰할 수 있습니다.
    """

    train_mode_layer = True

    def __init__(self, drop_ratio=0.2, rng=None):
        self.drop_ratio = drop_ratio
        self.mask = None
        self.rng = np.random.default_rng() if rng is None else rng

    def forward(self, x, train=True):
        if not train:
            return x

        keep_ratio = 1.0 - self.drop_ratio
        self.mask = (self.rng.random(x.shape) < keep_ratio) / keep_ratio
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask
