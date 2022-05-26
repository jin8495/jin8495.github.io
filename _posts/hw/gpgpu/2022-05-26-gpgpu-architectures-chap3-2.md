---
title: "GP-GPU 구조 시리즈: 챕터 3-2 - The SIMT Core: Instruction and Register Data Flow"
tags:
  - 컴퓨터구조
  - GPGPU
  - 시리즈
---



{%assign img_path = "/assets/images/hw/gpgpu/2022-05-16-gpgpu-architectures-chap3" %}


# Two-Loop Approximation

One-Loop Approximation에서는 한 번에 하나의 instruction만을 처리하도록 가정했다.
하지만 각 연산 코어가 처리해야할 warp의 수를 줄이면서 memory access latency를 숨기려면
  여러 instruction을 연속적으로 처리할 수 있어야 한다.
하지만 여러 instruction을 처리하면서 warp 간 dependency로 인한 hazard를 해결하려면
  추가적인 하드웨어가 필요하다.

따라서 two-loop approximation에서는 GPU에 instruction buffer를 도입했다.
여러 개의 scheduler가 instruction buffer를 살피며, dependency가 없는 instruction들을 꺼내
  각 연산 pipeline에 할당하게 된다.
게다가 GPU의 instruction buffer는 MSHR과 함께 사용되며 cache miss latency를 좀 더 숨길 수 있다.

Instruction buffer에서 instruction 간의 dependency를 탐지하는 방법은 크게 2가지가 있다.
기존 CPU의 reservation station과 scoreboard이다.

Reservation station은 name dependency를 효과적으로 제거할 수 있지만
  하드웨어 오버헤드가 큰 associative logic을 사용하기 때문에,
  GPU에선 scoreboard를 사용하는 것으로 추정된다.
Scoreboard의 경우 out-of-order 프로세서에서 매우 큰 오버헤드를 가지지만,
  in-order single threaded 프로세서에서는 매우 간단하게 구현될 수 있다.
그런데 GPU는 in-order multithreaded 프로세서로 볼 수 있다.
이때문에 scoreboard에 많은 수의 read port가 필요한데, 이를 해결하기 위해 다양한 연구가 있어왔다.

# Three-Loop Approximation

One-loop approximation에서는 한 번에 하나의 instruction만을 issue하는 간단한 GPU를 가정했다.
이 상황에서는 memory access latency를 숨기기 위해 많은 양의 warp가 필요했다.
하지만 한 번에 동작하는 warp의 수가 많아지면 register file의 크기가 커지게 된다.

이를 해결하고자 two-loop approximation에서는 warp들의 instruction dependency를 파악해
  여러 instruction들을 연속적으로 issue하는 구조를 도입했다.
이를 위해 instruction buffer와 MSHR, scoreboard와 같은 하드웨어가 추가되었다.
하지만 여전히 동작 가능한 warp의 수가 많아야 latency hiding이 가능하기 때문에,
  register file의 크기는 클 수밖에 없다.

Three-loop approximation에서는 register file 최적화에 초점을 둔다.


