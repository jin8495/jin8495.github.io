---
title: "GP-GPU 구조 시리즈: 챕터 1 - Introduction"
tags:
  - 컴퓨터구조
  - GPGPU
  - 시리즈
---

초창기 GPU는 비디오 게임의 실시간 렌더링을 담당하기 위해 개발되었다.
하지만 근 몇 년 사이에 GPU는 General-Purpose의 기능이 강해지면서, 다양한 연산들을 가속하기 위해 사용되었다.
그래서 GPU는 GP-GPU (General-Purpose Graphics Processing Unit)이란 이름이 붙게 되었다.

GP-GPU를 이용하면, 엄청난 양의 데이터를 병렬로 처리가 가능하기 때문에 머신러닝 연산에 사용이 됐고, 덕분에 엄청난 관심을 끌게 되었다.
GP-GPU는 머신러닝 어플리케이션 외에도 데이터 간의 의존성이 없고, 데이터 플로우 역시 단순하다면 엄청난 병렬 처리 성능을 활용해 압도적인 속도로 연산이 가능하다.

다른 이야기이긴 하지만, CPU와 GPU와 같은 General-Purpose processor의 한계(Energy, Frequency, Heat 등)가 뚜렷해지고 있기 때문에
  NPU, TPU와 같은 특정 어플리케이션만을 타겟으로 하는 Accelerator들의 연구도 활발히 이루어지고 있다.

# GP-GPU Hardware Basics

## GP-GPU와 CPU의 관계

기본적으로 GPU는 스스로 동작하는 하드웨어가 아니다.
가속기의 개념으로 사용이 되는 하드웨어이기 때문에 CPU의 제어가 필요하다.
CPU와 GPU의 역할이 나뉘어진 이유 중 하나로, 연산 가속의 처음과 끝에는 I/O 디바이스의 접근이 필요하기 때문이다.
이외에도 효과적인 GPU 연산 가속을 위해서 다양한 작업들이 필요하다.
그렇기 때문에 GPU 제조회사들은 CPU가 GPU를 효과적으로 제어할 수 있도록 여러 API를 제공한다. 

## GPU 컴퓨팅 시스템

아래 그림은 CPU와 GPU가 포함된 일반적인 컴퓨터 시스템의 Abstract diagram이다.

**TODO: Figure 1.1.과 같은 그림 첨부하기. (a)와 (b) 위치 바꿔서 새로 그리기.**

좌측(a)과 같은 구조는 휴대폰과 같은 모바일 시스템에서 찾아볼 수 있고,
  우측(b)과 같은 구조는 PC나 서버와 같은 시스템에서 찾아볼 수가 있고,
  
### Integrated GPU

좌측(a)의 구조를 Integrated GPU라 부른다. 여기서는 CPU와 GPU가 하나의 DRAM 메모리 공간을 공유한다.
그런 이유로, 둘은 동일한 메모리 인터페이스를 사용할 수 밖에 없다.
Integrated GPU는 위에서 언급한 바와 같이 모바일 시스템에서 주로 찾아볼 수 있기 때문에 저전력이 중요하다.
그래서 대부분의 Integrated GPU는 전력 소모가 적은 LPDDR 메모리를 사용한다.

### Discrete GPU

우측(b)의 구조를 Discrete GPU라 부르는데, 여기서는 CPU의 메모리와 GPU의 메모리가 분리되어 있다.
일반적으로 CPU의 메모리는 Host memory 또는 System memory라 부르고, GPU의 메모리는 Device memory라 부른다.

리눅스 기반 시스템에서 NVIDIA GPU를 사용할 때, `$ nvidia-smi` 커맨드를 입력하면 아래와 같은 출력을 얻을 수가 있다.
여기서 나타나는 메모리 용량이 각 GPU의 Device memory 용량을 의미한다.

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.86       Driver Version: 470.86       CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:01:00.0 Off |                  N/A |
| 30%   26C    P8    22W / 350W |      1MiB / 24268MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|              ...              |         ...          |          ...         |
+-------------------------------+----------------------+----------------------+
|   5  NVIDIA GeForce ...  On   | 00000000:E1:00.0 Off |                  N/A |
| 30%   24C    P8    10W / 350W |      1MiB / 24268MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

Discrete GPU 구조에서는 CPU와 GPU가 서로 다른 DRAM technology를 사용하기 때문에, 각 프로세서에 좀 더 특화된 메모리 인터페이스를 사용한다.
CPU의 경우 Bandwidth보다는 Latency가 더 중요하기 때문에 DDR을 사용하며, GPU의 경우 Latency보다 Bandwidth가 더 중요하기 때문에 GDDR을 사용한다.
GPU가 Bandwidth를 중요시하는 이유는 조금 뒤에 자세히 설명하도록 하겠다.




