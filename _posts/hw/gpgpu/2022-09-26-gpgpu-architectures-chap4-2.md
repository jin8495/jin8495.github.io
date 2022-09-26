---
title: "GP-GPU 구조 시리즈: 챕터 4-2 - Memory System"
tags:
  - 컴퓨터구조
  - GPGPU
  - 시리즈
---

이번 포스트에서는 GPU의 L1 cache외의 메모리 계층인 L1 texture cache, L2 cache, memroy partition에 대해 다뤄볼 예정이다.

{%assign img_path = "/assets/images/hw/gpgpu/2022-09-26-gpgpu-architectures-chap4" %}

---

# First-Level Memory Structures

이전 포스트에서 GPU의 first-level 메모리 계층에는 
  unified L1 data cache와 L1 texture cache가 있다고 소개했다.

Unified L1 data cache는 이전 포스트에서 다뤘으니, L1 texture cache에 대해 이야기해볼까 한다.

## L1 Texture Cache

최근 NVIDIA의 GPU는 L1 data cache와 texture cache를 통합해 면적을 아끼는 방향으로 설계를 한다.
통합된 cache가 어떻게 동작하는지 알기 위해서, 이 챕터에서는 둘을 따로 떼어내 설명을 하고 있다.
이를 염두해두고 L1 texture cache에 대해 알아보도록 하자.

이 책에서는 GPU의 "general-purpose" 기능에 초점을 맞추고 있기 때문에, 그래픽만을 담당하는 가볍게 훑고 넘어갈 예정이다.

그 전에 3D graphics에 대해 알아볼 필요가 있다.
3D graphics을 컴퓨터로 구현하려면, 3D model을 만든 뒤 그 위에 texture를 입혀야 한다.
이 과정을 texture mapping이라 부르는데, texture mapping을 구현하기 위해서는
  texture 이미지 중 일부 샘플의 좌표를 먼저 알아야한다.
이 샘플을 texel이라 부른다.

Texel을 메모리에 저장할 때, texel의 좌표를 메모리 주소로 활용한다.
왜냐하면 인접 texel들은 memory access 시에 함께 활용될 가능성이 높기 때문이다.
이러한 locality를 잘 활용한다면 cache의 효율을 높일 수 있다.

|<a name="Figure 1">![alt L1 texture cache 구조]({{ img_path }}-fig1.jpg)</a>|
|:-------|
|Figure 1. L1 texture cache 구조|

[Figure 1.](#Figure 1)은 L1 texture cache의 구조이다.
L1 data cache와 달리, tag array와 data array는 FIFO buffer ③에 의해 분리되어있다.
이는 DRAM에서 서비스 되는 miss reqeust의 latency를 숨기기 위함이다.

L1 texture cache는 FIFO로 인해 data array를 tag array보다 한참 뒤에 접근하게 된다.
이 시간은 대략 DRAM의 miss latency와 동일하다.
FIFO를 이용해 miss latency와 hit latency를 비슷하게 맞췄기 때문에, throughput은 유지를 할 수 있게 됐다.

## Unified Texture and Data Cache

이해를 돕기 위해, 여태까지 L1 data cache와 L1 texture cache를 분리해 설명했다.
하지만 실제로는 이 둘이 통합되어 있다.

그렇다 하더라도, 구조적 틀은 크게 변하지 않는다.
Texture 값들의 경우 read-only 속성을 갖고 있기 때문에, 이를 통해 texture인지 아닌지 구분이 가능하다.

만약 read-only라면 texture cache의 FIFO buffer를 통해 data array를 접근하게 하고, 
  그렇지 않다면 address crossbar를 통해 data array를 접근하게 한다면,
  여태 설명한 두 구조를 모두 간직하면서 통합을 시킬 수 있다.

# On-Chip Interconnection Network




---

# 정리


---

# 참고 자료

- T. M. Aamodt, W. W. L. Fung, and T. G. Rogers, General-purpose graphics processor architectures. San Rafael, California: Morgan & Claypool Publishers, 2018. doi: 10.2200/S00848ED1V01Y201804CAC044.
