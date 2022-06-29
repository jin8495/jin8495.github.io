---
title: "GP-GPU 구조 시리즈: 챕터 4 - Memory System"
tags:
  - 컴퓨터구조
  - GPGPU
  - 시리즈
---

GPU는 많은 양의 데이터를 한 번에 처리한다.
코어의 높은 throughput에 버금가는 데이터를 전달해야할 의무가 있기 때문에,
  큰 용량을 가지기보다는 넓은 bandwidth를 제공하는 방식으로 발전해왔다.
이번 챕터는 메모리 시스템에 관한 내용이다.


---

# First-Level Memory Structures

GPU는 first-level 메모리 계층으로 다양한 하드웨어를 갖고 있다.

하나는 "shared memory"를 위한 scratch pad와 data cache가 하나의 메모리를 함께 사용하는
  unified L1 data cache이다.
다른 하나는 그래픽 데이터만을 따로 저장하는 L1 texture cache이다.

이 섹션에서는 둘의 구조가 어떠한지, core pipeline과 어떤 식으로 상호작용하는지 알아볼 예정이다.

## Scratch Pad and L1 Data Cache

먼저 scratch pad에 대해 간단히 설명하겠다.

CUDA 프로그래밍 모델에서 shared memory는 같은 block 내의 모든 thread가 접근 가능한
  상대적으로 작고 빠른 메모리 영역이다.
Shared memory 영역의 데이터는 scratch pad라고 하는 메모리에 저장된다.
프로그래머 입장에서, 이 shared memory를 사용할 때 용량 뿐만 아니라 추가적으로 고려해야 할 게 있는데,
  바로 bank conflict이다.

NVIDIA 특허 자료에 따르면, scratch pad로 SRAM이 사용된다.
이때 SRAM은 lane마다 bank가 하나씩 있으며, 각 bank는 하나의 read port와 write port가 달려있다.
덕분에 각 thread는 모든 bank에 접근이 가능하다.
하지만 여러 thread가 같은 bank의 다른 위치를 접근하게 되면 bank conflict가 발생한다.


