# -*- coding: utf-8 -*-
"""
활성화 함수 모음.

[학생 구현 대상]
- ReLU.forward
- ReLU.backward
"""

import numpy as np


class ReLU:
    """
    ReLU(Rectified Linear Unit) 활성화 함수.

    양수는 그대로 통과시키고 음수와 0은 0으로 막습니다.
    이 비선형성이 없으면 Affine을 여러 층 쌓아도 결국 하나의 선형 변환과 같아집니다.

    forward에서 만든 mask는 backward에서 "gradient를 어느 위치로 흘릴지" 결정합니다.
    """

    def __init__(self):
        self.mask = None

    def forward(self, x):
        """
        Args:
            x: 임의 shape의 입력 배열

        Returns:
            x와 같은 shape. x > 0인 위치만 원래 값을 유지하고 나머지는 0입니다.

        주의: 입력 배열 x를 직접 수정하지 마세요. 복사한 배열을 수정해 반환합니다.
        """
        self.mask = (x<=0) # np.array() 변수에 x<=0 으로 할 시에 비교 연산을 시행하여 위치마다 T/F가 담긴 Shape 배열이 나옴
        out = x.copy() # 복사
        out[self.mask] = 0 # 불리언 배열 인덱스 사용 시 True인 위치만 골라서 값을 넣음
        # TODO 1: x <= 0 인 위치를 self.mask에 저장하세요. (backward에서 사용합니다)
        # TODO 2: x를 복사한 뒤 mask 위치를 0으로 만들어 반환하세요.
        return out
        raise NotImplementedError("ReLU.forward를 구현하세요.")

    def backward(self, dout):
        """
        Args:
            dout: 다음 층에서 넘어온 gradient. forward 출력과 같은 shape입니다.

        Returns:
            ReLU 입력에 대한 gradient. forward 때 x <= 0이었던 위치는 0입니다.

        생각해 볼 점: 왜 forward의 mask를 그대로 재사용할 수 있을까요?
        """

        # TODO: forward에서 저장한 self.mask를 이용해 gradient가 흐를 위치만 남기세요.
        copy_dout = dout.copy()
        copy_dout[self.mask] = 0
        return copy_dout
        raise NotImplementedError("ReLU.backward를 구현하세요.")
