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

{%assign img_path = "/assets/images/hw/gpgpu/2022-06-29-gpgpu-architectures-chap4" %}

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

NVIDIA 특허 자료에 따르면, scratch pad로 SRAM이 사용된다.
이때 SRAM은 lane마다 bank가 하나씩 있으며, 각 bank는 하나의 read port와 write port가 달려있다.
덕분에 각 thread는 모든 bank에 접근이 가능하다.
하지만 여러 thread가 같은 bank의 다른 위치를 접근하게 되면 bank conflict가 발생한다.


이제 L1 data cache를 알아보자.

L1 data cache는 global memory 주소 공간의 데이터를 저장하는 용도로 사용된다.
여기서 특정 warp 내의 모든 thread가 요구하는 데이터가 하나의 L1 data cache block에 모두 존재한다면
  단일 request 만으로 모든 데이터를 접근할 수 있다.
이러한 접근을 coalesced access라 한다.
반대로 특정 warp 내의 모든 thread가 요구하는 데이터가 여러 L1 data cache block에 흩뿌려져 존재한다면
  여러 request를 보내야 모든 데이터를 접근할 수 있다.
이러한 접근은 uncoalesced access라 한다.



프로그래머는 scratch pad와 L1 data cache를 모두 효율적으로 사용하기 위해
  bank conflict와 uncoalesced access를 모두 고려하며 프로그래밍해야한다.


|<a name="Figure 1">![alt Unified L1 data cache와 scratch pad 메모리]({{ img_path }}-fig1.jpg)</a>|
|:-------|
|Figure 1. Unified L1 data cache와 scratch pad 메모리|

[Figure 1.](#Figure 1)은 NVIDIA의 Fermi와 Kepler 마이크로 아키텍쳐에서 보여준 unified L1 cache의 구조이다.
Unified L1 cache는 L1 data cache와 scratch pad가 동일한 SRAM data array ⑤를 사용하는 구조로, 
  일부는 scratch pad 메모리를 위해 direct mapped cache로 설정되어 있고 다른 일부는 set associative cache로 설정되어 있다.
해당 디자인은 instruction pipeline의 stall을 막기 위해 bank conflict와 L1 data cache miss 발생 시 replay mechanism을 활용한다.
Replay mechanism은 뒤에서 자세히 설명하도록 하겠다.

먼저 자세한 동작을 설명하기 전에 모든 memory access request는 instruction pipeline의 Load/Store Unit ①으로 보내진다.
Memory access request는 여러 개의 memory address의 집합으로 이뤄져 있는데, 각 address는 warp 내의 각각의 thread가 필요로 하는 데이터의 주소이다.

이제 순서대로, scratch pad access의 동작에 대해 먼저 설명하고,
  L1 data cache의 coalesced cache hit, cache miss, uncoalesced cache hit의 동작을 설명하겠다.
위 내용을 머릿 속에 잘 염두해둔 채 읽으면 동작을 이해하는데 도움이 될 것이다.

### Scratch Pad Memory Access Operation


