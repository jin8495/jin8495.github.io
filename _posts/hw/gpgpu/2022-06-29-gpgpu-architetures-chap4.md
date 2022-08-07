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

Shared memory 접근을 하게 되면, 가장 먼저 Arbiter ②가 해당 접근의 메모리 주소가 bank conflict를 일으키는지,
  일으키지 않는지를 먼저 판단하게 된다.
만약 bank conflict를 일으킬 것 같다면, Arbiter ②는 request를 두 부분으로 나눈다.
첫 번째 부분은 bank conflict를 일으키지 않는 thread들이고, 나머지는 bank conflict를 일으키는 thread들이다.
Bank conflict를 일으키는 thread가 요청한 request는 다시 instruction pipeline으로 돌아가 다시 실행된다.
이러한 execution 방식을 replay라 부른다.
Replay는 area와 energy efficiency 사이의 trade-off가 존재한다.
Replay를 하게 되면, memory access instruction이 instruction buffer로 다시 돌려보내지기 때문에 
  area를 아낄 수 있지만, 큰 buffer를 다시 접근하기 때문에 energy를 소모하게 된다.
Instruction buffer의 buffering을 제한하는 것으로, 이를 해결할 수 있다.
Buffer의 크기를 제한한 뒤, buffer의 남은 공간이 부족하면 memory access operation을 스케쥴링하지 않는다.
그러면 request가 나가지 않기 때문에, 버퍼를 사용하지 않는다.
덕분에 버퍼를 접근하지 않아 energy를 아낄 수 있고 buffer의 용량을 줄여 area도 아낄 수 있다.
되돌려 보내진 reqeust들이 replay 되는 방식을 자세히 알아보기 전에,
  bank conflict를 일으키지 않는 request가 어떻게 처리되는지 먼저 알아보자.

Shared memory는 direct-mapped cache이기 때문에 request는 Tag Unit ③을 look up 하지 않는다.
Arbiter ②가 request를 받아들임과 동시에 register file에 writeback을 미리 스케쥴한다.
왜냐하면 bank conflict가 발생하지 않는 상황에서 direct-mapped cache의 access latency는 일정하기 때문이다.
Tag Unit ③을 look up 하지는 않지만, 각 thread의 주소가 어느 bank에 맵핑되어 있는지 알기 때문에 
Address Crossbar ④를 제어해 reqeust가 Data Array ⑤를 올바르게 접근하도록 한다.
Data Array ⑤는 32-bit 크기이며, 각 bank의 개별 row를 개별적으로 접근하더라도 처리할 수 있도록 decoder를 갖고 있다.
이렇게 반환된 데이터는 Data Crossbar ⑥를 거쳐 올바른 thread lane의 register file로 쓰여지게 된다.
이때 active thread가 아니라면, register file에 쓰여지지 않고 무시된다.

만약 shared memory 접근에 1 cycle이 소모된다면, 1 cycle 이후에 replay가 진행된다.
Replay 된 request들은 다시 L1 cache의 Arbiter ②를 접근하며, bank conflict가 또 일어난다면, request는 다시 나뉘게 된다.

### Cache Read Operations

L1 cache의 Data Array ⑤는 많은 수의 bank로 이루어져 있기 때문에 개별 warp는 shared memory를 각각 접근할 수 있다.
반면, global memory 영역에 속하는 L1 memory는 1 cycle에 하나의 cache-line만 접근 가능하도록 제약이 걸려있다.
이 제약 덕분에 tag overhead를 줄일 수 있다.

Fermi와 Kepler micro-architecture에서는 128B cache-line 크기를 갖지만, Maxwell부터는 이를 4개로 나눈 32B cache-line 크기를 갖는다.
각각의 32B를 sector라 부르는데, 덕분에 낭비되는 GP-GPU의 대역폭을 아낄 수 있게 됐다.
Sector의 크기를 32B로 둔 이유는, GP-GPU에서 사용하는 DRAM의 접근 크기 때문이다.
GDDR5나 HBM2와 같은 GP-GPU 용 메모리 인터페이스 표준은 최소 접근 단위가 32B이다.
Sector의 크기가 32B보다 더 작아도 DRAM은 무조건 32B로만 접근이 가능하기 때문에, 32B의 sector를 갖게 됐다.

이제 cache read 과정을 살펴보자.
Load/Store Unit ①은 가장 먼저 request의 메모리 주소를 계산한 다음 coalescing 여부를 판단한다.
그 다음 coalesced request는 Arbiter ②로 전달된다.
이때 Arbiter ②는 request 처리 자원이 부족하다면 request 처리를 거부할 수 있다.
예를 들어, cache set의 모든 way가 처리 중이거나 Pending Request Table ⑦의 공간이 없다면 request는 거부된다.
Pending Request Table ⑦은 추후에 자세히 설명하도록 하겠다.

