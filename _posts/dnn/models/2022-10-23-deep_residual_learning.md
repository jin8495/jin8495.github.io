---
title: "ResNet - Deep residual learning for image recognition 리뷰"
tags:
  - 딥러닝
  - CNN
  - CNN 모델
---

이번에 CNN 기반 backbone network에 대해서 공부할 기회가 생겼다.

이번 글은 CNN 기반 backbone network 중 가장 유명한 ResNet에 대한 리뷰이다.


---

# Introduction

네트워크의 깊이는 성능에 매우 중요한 요소이다.
- ImageNet 데이터셋을 학습시킨 많은 모델들이 “very deep” 모델을 이용 (약 16~30 레이어)

하지만 깊은 모델을 만들면 다양한 문제가 발생한다.
- Vanishing/exploding gradient
  - 학습시 기울기가 점점 사라지거나, 폭발적으로 커지는 문제
  - 이를 해결하기 위해 normalized initialization, normalization layer과 같은 방법들이 제시됨

- Accuracy degradation
  - 모델이 깊어질수록 정확도가 수렴하다가, 일정 깊이부터 정확도가 감소하는 문제

---
# 참고 자료
- He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
