---
title: "GP-GPU 구조 시리즈: 챕터 3 - The SIMT Core: Instruction and Register Data Flow"
tags:
  - 컴퓨터구조
  - GPGPU
  - 시리즈
---

현대의 GP-GPU는 수천~수만개의 쓰레드를 동시에 처리할 수 있다.
이를 가능케하기 위해 SIMT 코어는 다양한 방법을 채용했는데,
  이번 게시글에서는 이를 알아보도록 하겠다.

{%assign img_path = "/assets/images/hw/gpgpu/2022-05-16-gpgpu-architectures-chap3" %}

|<a name="Figure 1">![alt GPU 파이프라인]({{ img_path }}-fig1.jpg)</a>|
|:------|
|Figure 1. GPU 마이크로아키텍처의 파이프라인|

[Figure 1.](#Figure 1)과 같이, GPU 마이크로아키텍쳐의 파이프라인은 SIMT front-end와 SIMD back-end로 나뉜다.
파이프라인은 크게 3개의 스케쥴링 루프를 갖고 있다.
각 루프는 instruction fetch loop, instruction issue loop, register access scheduling loop라 부르는데,
  루프들은, 아래와 같이, 파이프라인의 스테이지들을 포함한다.
- **Instruction fetch loop**: Fetch, I-Cache, Decode, I-Buffer
- **Instruction issue loop**: I-Buffer, Scoreboard, Issue, SIMT stack
- **Register access scheduling loop**: Operand Collector, ALU, Memory

지금부터는 파이프라인의 각 스테이지들이 어떤 역할을 하는지 자세히 살펴볼 것이다.
N-Loop approximation이란 방법을 이용해, GPU의 동작을 설명할 것이다.
One-loop approximation에선, 최대한 단순화하여 GPU 파이프라인을 설명한 뒤,
  two, three-loop으로 가면서 세세한 부분을 설명할 예정이다.

-----

# One-Loop Approximation

먼저 다른 건 제쳐두고, 단일 스케쥴러를 가진 GPU를 가정해보자.
CUDA 프로그래밍 모델 매뉴얼과 차이가 있을 수는 있지만, 이해를 돕기 위해 많은 부분을 생략하고 설명할 예정이다.

먼저, 스케쥴링의 단위는 warp이다. 매 사이클마다 하드웨어는 warp를 선택해 스케쥴링한다.
이때 warp의 program counter를 이용해 instruction memory에서 warp가 실행할 다음 instruction을 접근한다.
Instruction fetch가 끝나면, decode 단계를 거친 뒤, register file에서 source operand register를 불러오게 된다.
register file에서 reigster를 불러오면서, 병렬적으로, SIMT execution mask가 결정된다.

Execution mask와 source register가 모두 준비가 완료되면, SIMD 방식으로 연산이 시작된다.
이때, SIMT execution mask를 바탕으로 실행해도 되는 lane이 결정되며, 그 lane에 해당하는 쓰레드만 실행되게 된다.
GPU의 function units은 CPU와 마찬가지로 heterogeneous하다. 즉, function unit은 instruction의 일부[^1]만 처리하게 된다.

모든 function unit은 내부에 lane을 갖고 있는데, 이 lane은 SIMT로 처리할 수 있는 쓰레드의 수만큼 존재한다.
예를 들어, NVIDIA의 warp는 32개의 쓰레드로 구성되어 있기에, 각 function unit은 총 32개의 lane을 갖고 있다.



## SIMT Execution Masking

CUDA 프로그램은 SIMT execution model이 있기 때문에, 프로그래머는 개별 쓰레드를 다루는 MIMD의 방식으로 프로그래밍이 가능하다.
이를 위해선 기존의 predication 만으로도 충분하지만,
  GPU의 SIMT execution model은 기존의 predication과 더불어, predicate mask stack (SIMT stack)을 함께 이용한다.
  
CPU에서는 predication을 위해 다수의 predicate register를 사용한다.
하지만 GPU에서 predicate register를 이용하면, 많은 수의 쓰레드를 감당할 수 없어, 하드웨어 오버헤드가 기하급수적으로 커진다.
따라서 SIMT stack을 이용해 하드웨어 오버헤드를 줄였다.

SIMT stack을 사용하면 두 가지 이슈를 간단히 해결할 수 있다.

1. **Nested control flow**: 하나의 브랜치가 다른 브랜치에 의존적인 경우. 이는 올바른 functionality를 위해 필수적이다.
2. **Skipping computation entirely**: Warp 내의 모든 쓰레드가 control flow path에 속하지 않는 경우. 이는 성능 향상에 큰 도움이 된다.


실제 SIMT stack에서는 특수한 instruction을 이용해 관리되지만, 이번 글에서는 하드웨어만으로 관리되는 SIMT stack을 소개하고자 한다.

아래 코드는 do-while loop 안에 두 개의 nested branch로 이루어진 CUDA C 코드이다. 이를 컴파일에 PTX Assembly로 만든 것이 그 아래 코드이다.

```c++
do {
  t1 = tid * N;         // A
  t2 = t1 + i;
  t3 = data1[t2];
  t4 = 0;
  if ( t3 != t3 ) {
    t5 = data2[t2];     // B
    if ( t5 != t4 ) {
      x += 1;           // C
    } else {
      y += 2;           // D
    }
  } else {
    z += 3;             // F
  }
  i++;                  // G
} while ( i < N );
```

```nasm
A:    mul.lo.u32      t1, tid, N;
      add.u32         t2, t1, i;
      ld.global.u32   t3, [t2];
      mov.u32         t4, 0;
      setp.eq.u32     p1, t3, t4;
@p1   bra             F;
B:    ld.global.u32   t5, [t2];
      setp.eq.u32     p2, t5, t4;
@p2   bra             D;
C:    add.u32         x, x, 1;
      bra             E;
D:    add.u32         y, y, 2;
E:    bra             G;
F:    add.u32         z, z, 3;
G:    add.u32         i, i, 1;
      setp.le.u32     p3, i, N;
@p3   bra             A;
```




---

# 참고 자료



---




[^1]: CPU의 function unit은 *load/store unit, floating-point unit, integer unit* 등으로 구성된다.
    GPU도 이와 마찬가지로 instruction의 종류별로 하드웨어 unit을 갖지 않고, insturcion의 일부만 처리하도록 설계되어 있다.
