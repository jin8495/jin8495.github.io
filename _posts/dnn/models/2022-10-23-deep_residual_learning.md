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

- 데이터 압축에 활용하는 경우
  - Original 값보다 residual representation의 절대값이 작은 경우, 압축에 유리하게 인코딩 가능하다.
- 최적화 문제 해결에 활용하는 경우
  - 최적화 문제를 residual representation으로 변형하면, 최적화 목표 지점에 훨씬 빠르게 도달할 수 있다.
    그 이유는 최적화 목표 지점에 가깝게 시작 지점을 초기화해주기 때문이다.<br>
    종종 preconditioning이라고도 부른다.

그 다음은 **shortcut connection**이다.
Shortcut connection은 말 그대로, 한 지점과 다른 지점을 잇는 지름길이다.
이는 신경망 학습에서 vanishing/exploding gradient 문제를 해결하기 위해 활용된다.

- Vanishing/exploding gradient 문제 해결에 활용하는 경우
  - Gradient가 중간 레이어들을 거치지 않고 shortcut connection을 통해 전파된다.
    덕분에, 온전한 gradient가 레이어들 사이를 이동한다.


---
# Deep Residual Learning

이 섹션에서는 ResNet 논문에서 제안한 deep residual learning에 대해서 설명한다.

What? Why? How? 순서로 설명을 할 것이다.

## What?

### Residual Learning

**Residual learning은 복잡한 함수를 residual representation을 이용해 근사하는 방법이다.**

Residual learning을 바로 설명하기 전에, 원래 함수를 어떻게 residual representation으로 표현하는지 먼저 알아보자.

- $H(x)$: 원래 함수이다. 여러 개의 non-linear layer를 쌓아 올려 만든 함수이고 복잡한 함수를 근사할 수 있다.

- $F(x):=H(x)-x$: 원래 함수에서 입력 값을 뺀 residual representation이다. (이때, 입력 $x$와 출력 $H(x)$의 차원은 동일하다 가정한다.)

위 두 함수를 가지고, 아래와 같이 원래 함수 $H(x)$를 $F(x)$에 대한 식으로 치환할 수 있다.

- $H(x)=F(x)+x$

$H(x)$를 곧장 이용하지 않고, $F(x)+x$로 치환하면 신경망 모델을 조금 더 쉽게 학습할 수 있다.
왜 그런지는 아래에서 조금 더 살펴보겠다.

## Why?

### Thought Experiment

**왜 residual representation을 이용하는가?**

논문에서는 사고 실험을 통해, 이유를 정당화한다.
단순히 정당화에서 그치는 건 아니고,
  [Analysis of Layer Responses](#Analysis of Layer Responses)에서 실제 실험을 통해 증명도 한다.

그럼 다시 사고 실험 이야기로 돌아와서 어떤 이유에서 residual representation을 이용하는지 알아보겠다.

- **Shallow model A**가 있다고 가정해보자. 모델 A를 바탕으로 3가지 모델을 만들어보면서 이유를 설명하겠다.

  1. **Deep model A'** : A'은 A의 뒷 단에 identity layer를 쌓아 만든 모델이다.
    - A와 A'의 출력 값은 동일할 것이다. 즉, training error가 동일할 것이다.
    - A'이 최적의 모델 (optimal model)이라고 가정해보자. A'보다 더 나은 모델은 세상에 없다.

  2. **Deep model B** : B는 A의 뒷 단에 non-linear layer들을 쌓아 만든 모델이다.
    - A의 뒤에 일반적인 신경망 연산들을 쌓아 올릴 모델이란 의미이다.
    - 만약 B의 학습이 잘 진행되고 있다면, B는 최적의 모델인 A'과 점점 동일해질 것이다.
      즉, non-linear layer들은 identity layer들을 모방하게 될 것이다.
    - 하지만 accuracy degradation 문제에서 알 수 있듯, identity layer에 가까워지지 않는다.

  3. **Deep model B'** : B'은 A의 뒷 단에 residual representation을 포함해 non-linear layer들을 쌓아 만든 모델이다.
    - B'의 non-linear layer들은 학습을 할 때, 모든 파라미터를 0으로만 만들어도 최적의 모델 A'을 얻을 수 있다.

  - 위 사고 실험은 모델 A'이 최적의 모델이라는 가정 하에 성립한다. 하지만 실제로는 A'이 최적의 모델이 아닐 것이다.
    그렇다 하더라도 B'과 같이 모델을 만들게 되면, 조금 더 학습이 잘 될 것으로 기대할 수 있다.<br>
    그리고 실제로 이를 증명하기 위해, 논문에서는 한 가지 실험을 추가했다. 이는 아래에서 자세히 설명하도록 하겠다.

## How?

### Identity Mapping as a Shortcut

ResNet은 **building block**이란 단위로 residual learning을 적용한다.

여기서 building block은 shortcut connection을 통한 residual learning이 적용된 블럭이다.
ResNet은 building block이 기본 구성 단위이기 때문에, building block을 쌓아 올려 신경망 모델을 구성한다.

Building block은 아래 수식 조건을 만족한다면, 어떻게 구성하더라도 상관없다.

$$ y=F(x, W_i) + x$$

- $x$, $y$는 각각 입력과 출력 벡터.
- $F(x, W_i)$는 학습시킬 residual mapping.
- $F+x$는 element-wise addition으로, shortcut connection을 의미.
- Shorcut connection 이후 non-linearity를 적용할 것.

아래의  [Figure 2](#Figure 2)는 실제로 ResNet-18, 34에 적용한 building block이다.

|<a name="Figure 2">![alt Building block]({{ img_dir }}2-building_block.png)</a>|
|:---:|
|Figure 2. Residual learning이 적용된 building block|

[Figure 2](#Figure 2)의 그림을 위에서 서술한 수식에 빗대어 표현하면,

$$F=W_2 \sigma(W_1 x),\ \ y=\sigma(W_2\sigma (W_1 x)+x)$$

- $\sigma$는 ReLU 함수.
- Bias는 편의상 생략.

로 표현이 가능하다.

Shortcut connection으로 linear projection $W_s$를 적용할 수도 있지만, ResNet에서는 단순한 element-wise addtion을 적용했다.
후술할 실험 결과에서 확인할 수 있는데, linear projection는 연산 오버헤드에 비하면 신경망 모델의 성능 향상이 그리 크지않다.
따라서 ResNet에서는 element-wise addtion을 shortcut connection으로 사용한다.

만약, linear projection을 shortcut connection으로 사용한다면, 출력인 $F(x)$와 입력인 $x$의 차원에 따라 다른 방식을 적용해야 한다.
둘의 차원이 다르다면, 차원을 동일하게 맞춰줄 수 있도록 $W_s$의 형상을 결정해줘야한다.
하지만, 둘의 차원이 같다면, $W_s$의 형상은 정사각행렬로 결정하면 된다.

### Network Architectures

TODO:

### Implementation


---





---
[^1]: Plain Net은 VGG를 기반으로 만든 모델이다. 모델의 자세한 구조는 [Network Architectures](#Network Architectures)를 참고.
 


---
# 참고 자료
- He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
