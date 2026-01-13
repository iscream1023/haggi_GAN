# 🧪 Generative AI & Math Study Log

PyTorch를 활용한 GAN 기초 실습과 이를 위한 확률/통계 베이스를 다진 학습 기록입니다.
생성 모델의 발전 과정을 이해하기 위해 학습했으며, GAN의 구조적 한계를 확인하고 학습을 마무리했습니다.

## 📚 Study Contents

### 1. GAN (Generative Adversarial Networks)
PyTorch 튜토리얼을 기반으로 생성자와 판별자의 경쟁 구조(Minimax Game)를 실습했습니다.
- **Vanilla GAN**: MNIST 데이터셋 생성 실습
- **Implementation**: `Discriminator`와 `Generator`의 손실 함수(Loss) 설계 및 역전파 구현

### 2. Probability & Statistics (Base)
GAN 모델을 더 깊게 이해하고자 
- **몬테카를로 키뮬레이션으로 배우는 확률통계 with python, 장철원**: 해당 도서를 통해 기초적인 확률 개념과 몬테카를로 시뮬레이션을 통한 데이터 예측 실습을 수행했습니다.
- **머신 러닝 마스터 클래스, 민재식**: 머신러닝에 대한 기본기를 바로잡았습니다.

## 🛑 Why I Moved On (Retrospective)

위 책을 공부하면서 GAN 모델에 대한 다음과 같은 한계점을 공부했습니다.

> **Reason for Conclusion**
> 1. **GAN의 한계 (Diversity Issue)**
>    GAN 모델에서 판별자는 생성자가 만든 데이터 진위 여부만을 판단합니다.
>    * 개 이미지를 생성하는 경우, 판별자는 골든 리트리버 10장 + 말티즈 10장의 이미지와 골든 리트리버 20장의 이미지의 차이를 구별하지 못합니다.
>    * 이는 생성자가 높은 점수를 받는 특정 데이터 출력에 매몰되게 만들 수 있으며, 생성 데이터의 다양성을 떨어지게 만듭니다.
>
> 2. **Diffusion의 가능성 (Structure Learning)**
>    이에 반해 Diffusion 모델은 데이터에 대해 가우시안 노이즈를 쌓아 데이터와 그 주변을 구조화합니다. 다시 말해 단순 점으로 표현되던 개별 데이터에 가우시안 노이즈를 더하여 그 주변 분포를 생성할 수 있도록 합니다.
>    * 그리고 이 가우시안 노이즈들을 학습함으로써 비록 오랜 시간이 걸릴지언정 입력 데이터와 그 주변 분포를 학습하기 때문에 GAN에 비해 다양한 데이터를 생성할 수 있습니다.


이러한 결론을 바탕으로 GAN에 대한 학습은 현재 수준에서 중단하기로 결심했습니다.

## 🛠 Tech Stack
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white"/> <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>


