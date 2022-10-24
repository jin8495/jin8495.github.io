---
title: "ResNet - Deep residual learning for image recognition 리뷰"
tags:
  - 딥러닝
  - CNN
  - CNN 모델
---

이번에 CNN 기반 backbone network에 대해서 공부할 기회가 생겼다.

이번 글은 CNN 기반 backbone network 중 가장 유명한 ResNet에 대한 리뷰이다.

{% assign img_dir = "/assets/images/dnn/models/2022-10-23-resnet/" %}

---

# Introduction

네트워크의 깊이는 성능에 매우 중요한 요소이다.
ImageNet 데이터셋을 학습시킨 많은 모델들이 “very deep (약 16~30 레이어)” 모델을 이용했다.

하지만 깊은 모델을 만들면 다양한 문제가 발생한다.
문제는 크게 2가지로 나뉜다.

하나는 vanishing/exploding gradient이다.
이는 학습시 기울기가 점점 사라지거나, 폭발적으로 커지는 문제인데,
이를 해결하기 위해,
  xavier's initialization이나 he's initialization과 같은 normalized initialization,
  또는 batch normalization과 같은 normalization layer과 같은 방법들이 제시되었다.

다른 하나는 accuracy degradation이다.
이는 모델이 깊어질수록 정확도가 수렴하다가, 일정 깊이부터 정확도가 감소하는 문제를 말한다.
ResNet 논문에서 주목하는 문제가, accuracy degradation이다.
그럼 이에 대해 좀 더 자세히 살펴보자.

|<a name="Figure 1">![alt accuracy degradation experiments]({{ img_dir }}1-acc_degradation.png)</a>|
|:---:|
|Figure 1. CIFAR 10을 학습한 plain network의 training error(좌)와 test error(우)|

논문에서는 accuracy degradation이 존재한다는 것을 보이기 위해 한 가지 실험을 진행했다.
Plain network[^1]라는 모델을 만들어, CIFAR 10 데이터셋을 학습시키는 것이었다.
이때, plain network의 깊이를 바꿔가며 모델을 학습시켰다.

[Figure 1.](#Figure 1)과 같이, 모델의 깊이는 깊어졌음에도 불구하고 error는 증가한다는 사실을 발견할 수 있었다.
즉, accuracy degradation이 실제로 존재한다는 것을 보인 것이다.

논문에서는 accuracy degradation 문제를 해결하기 위해 한 가지 방법을 제안했다.<br>
**Shortcut connection**을 활용한 **deep residual learning**이다.

이를 자세히 알아보기 전에, ResNet에 근간이 되는 2가지를 잠깐 살펴보고 넘어가자.

먼저, **residual representation**이다.
Residual representation은 어떤 값과 다른 값의 차이를 이용한 표현법이다.
이 표현법은 주로 데이터 압축과 최적화 문제 해결에 활용한다.

TODO:

  - **Residual representation**: 어떤 값과 다른 값의 차이를 이용하는 표현법
    - 데이터 압축에 활용
      - Original 값보다 residual representation의 절대값이 작은 경우, 압축에 유리하게 인코딩 가능
    - 최적화 문제 해결에 활용
      - 최적화 문제를 residual representation으로 변형하면, 최적화 목표 지점에 훨씬 빠르게 도달 (preconditioning 이라고도 부름)
        최적화 목표 지점에 가깝게 시작 지점을 초기화해주기 때문

  - **Shortcut connections**: 한 지점과 다른 지점을 잇는 지름길
    - 신경망 학습에서, vanishing/exploding gradient 문제를 해결
      - Gradient가 중간 레이어들을 거치지 않고 shortcut connection을 통해 온전하게 역전파될 수 있음

---
# Deep Residual Learning
## Residual Learning




---
[^1]: Plain Net은 VGG를 기반으로 만든 모델이다. 모델의 자세한 구조는 [Network Architectures](#Network Architecture)를 참고.
 


---
# 참고 자료
- He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
