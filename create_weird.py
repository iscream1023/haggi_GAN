#확률 분포 만드는 함수
import numpy as np
def get_custom_samples(target_pdf, x_min, x_max, y_max, n_samples):
    """
    target_pdf: 원하는 확률밀도함수 (함수 객체)
    x_min, x_max: x축 범위
    y_max: pdf의 최대 높이 (넉넉하게 잡기)
    """
    samples = []
    while len(samples) < n_samples:
        # 1. 박스 안에서 랜덤 좌표 (x, y) 찍기
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(0, y_max)

        # 2. 그래프 아래쪽이면 합격(Accept), 아니면 기각(Reject)
        if y < target_pdf(x):
            samples.append(x)

    return np.array(samples)

def weird_pdf(x):
    # 사용할 분포 정의
    return np.sin(x) ** 2

my_samples = get_custom_samples(weird_pdf, 0, np.pi, 1.0, 1000)