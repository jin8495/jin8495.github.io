---
title: "GP-GPU 구조 시리즈: 챕터 2 - Programming Model"
tags:
  - 컴퓨터구조
  - GPGPU
  - 시리즈
---

GPU의 프로그래밍 모델은 GPU 하드웨어의 동작과 다르게 설계되어 있다.
이는 프로그래밍의 편리성을 증대시키기 위해서인데, 덕분에 프로그래머는 편리하게 GPU를 사용할 수 있게 되었다.

이번 게시글에서는 GPU의 프로그래밍 모델, 그 중에서도 CUDA 프로그래밍 모델에 대해 알아보도록 하겠다.

현대의 GPU는 SIMD 하드웨어를 이용하기 때문에, data-level parallelism을 활용한 어플리케이션을 가속한다.
하지만 GPU 프로그래밍 모델은 SIMD 하드웨어를 프로그래머에게 노출하지 않는다.
SIMD 하드웨어를 노출하게 되면, 하나의 instruction이 적용될 여러 data들을 프로그래머가 직접 지정해야 하는데,
  이는 프로그래밍 난이도를 증가시키기 때문이다.

대신 CUDA나 OpenCL 같은 프로그래밍 모델은 MIMD-like 프로그래밍 모델을 채용한다.
따라서 각 쓰레드에 대한 스칼라 프로그래밍이 가능하며, GPU에서 스칼라 쓰레드들이 개별적으로 구동하는 것처럼 보이게 한다.
GPU 하드웨어는 런-타임에 쓰레드를 warp[^1]라고 부르는 단위로 묶고,
  warp에 속한 여러 쓰레드를 SIMD와 같은 방식으로 동시에 처리한다.
이런 동작 모델을 SIMD와 구분짓기 위해 SIMT (Single-Instruction Multiple-Threads)라 부르게 되었다.

----

# Programming Model

CUDA는 C/C++을 기반의 API를 제공하기 때문에, 프로그래머가 입문하기 쉽다.
또한 위에서 언급한 바와 같이 스칼라 프로그래밍이 되기 때문에,
  컴파일러가 자동적으로 GPU에 쓰레드들을 warp 단위로 할당한다.

CUDA 프로그래밍 모델에는 크게 3가지의 특징이 있다.

1. CUDA blocks - 쓰레드들을 묶은 그룹
2. Shared Memory - 블록끼리 공유하는 메모리 영역
3. Synchronization Barriers - 모든 쓰레드들이 도달할 때까지 쓰레드의 진행을 가로막는 API

위와 같은 특징들 덕분에, 프로그래머들은 어렵지 않게 CUDA 프로그램을 짤 수 있다.
특징들은 이후 자세하게 설명할 예정이다.

그 전에, 코드 예시를 보고 CPU 코드와 GPU 코드의 차이에 대해 알아보자.

```c++
void saxpy_serial(int n, float a, float *x, float *y) {
  for (int i = 0; i < n; i++)
    y[i] = a*x[i] + y[i];
}

int main() {
  float *x, *y;
  int n;
  // omitted: allocated CPU memory for x and y and initialize contents
  saxpy_serial(n, 2.0, x, y);   // Invoke serial SAXPY kernel
  // omitted: use y on CPU, free memory pointed to by x and y
}
```

```c++
__global__ void saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a*x[i] + y[i];
}

int main() {
  float *h_x, *h_y;
  int n;
  // omitted: allocate CPU memory for h_x and h_y and initialize contents
  float *d_x, *d_y;
  cudaMalloc( &d_x, n * sizeof(float) );
  cudaMalloc( &d_y, n * sizeof(float) );
  cudaMemcpy( d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice );
  int nthreads = 256;
  int nblocks = (n + nthreads-1) / nthreads;
  saxpy<<<nblocks, nthreads>>>(n, 2.0, d_x, d_y);
  cudaMemcpy( h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost );
  // omitted: use h_y on CPU, free memory pointed to by h_x, h_y, d_x, and d_y
}
```

두 코드는 모두 **saxpy (single-precision scalar value A times vector value X plus vector value Y)**라 불리는 연산을 구현한 것이다.
위 코드는 CPU를 이용한 코드이기 때문에 직렬로 프로그래밍 되어 있고, 아래 코드는 GPU를 이용한 코드이기 때문에 병렬로 프로그래밍 되어 있다.

먼저 두 코드 모두 `main` 함수는 제외하고 `saxpy...` 함수만 살펴보자.

- `saxpy_serial`의 경우 for-loop을 돌면서 연산을 실행한다.
`x[i]`와 `y[i]`는 서로 간의 의존성이 없음에도 불구하고, 반복문으로 인해 연산 속도는 느려진다.
- `saxpy`의 경우 for-loop을 사용하지 않고 쓰레드의 좌표 값인 `i`만을 이용한다.
따라서 서로 간의 의존성이 없는 `x[i]`와 `y[i]`는 병렬 연산이 가능하고, 덕분에 빠른 연산이 가능하다.

이제 `main` 함수를 살펴보면 좀 더 뚜렷한 차이를 확인할 수 있다.

- 먼저 CPU 함수의 경우, 단순히 `saxpy_serial` 함수만 호출해 사용하면 된다.
- 하지만 GPU 커널은 조금 더 많은 사전/사후 작업이 필요하다.
먼저 host memory에서 사용할 포인터와 device memory에서 사용할 포인터, 두 가지를 구분해서 선언해야 한다.
관례적으로 host memory와 device memory는 prefix로 각각 `h_`, `d_`를 붙인다.
그리고 host memory에 메모리 공간과 값을 할당했다면, device memory 역시 메모리 공간과 값을 할당해준다.
할당을 위해 각각 `cudaMalloc`, `cudaMemcpy`라는 API를 사용한다.
이후 kernel을 호출하게 되는데,
  이때 CUDA block당 쓰레드의 개수인 nthreads, CUDA block의 개수인 nblocks를 `<<< >>>`에 지정해
  GPU에서 동작할 CUDA block의 개수와 크기를 결정한다.
마지막으로 연산이 끝난 device memory의 데이터를 CPU에서 사용하기 위해 `cudaMemcpy`로 다시금 불러온다.

## CUDA Blocks

GPU 컴퓨팅 어플리케이션의 커널은, [Figure 1.](#Figure 1)과 같이 Grid, Block[^2], Thread의 계층으로 구성된다.

{%assign img_path = "/assets/images/hw/gpgpu/2022-05-09-gpgpu-architectures-chap2" %}
|<a name="Figure 1">![alt CUDA 쓰레드 계층]({{ img_path }}-fig1.png)</a>|
|:-------|
|Figure 1. CUDA 프로그램의 쓰레드 계층 구조|

쓰레드를 계층 구조로 만들어 이용하면 몇 가지 장점이 있다.

첫 번째로, 병렬 연산 프로그래밍에 용이하다.
- 각 쓰레드는 `dim3`라는 자료 구조의 `blockDim`, `blockIdx`, `threadIdx`이란 변수들을 이용해 접근이 가능하다.
위 방식을 잘 이용하면 2D나 3D 행렬 연산 시, 각 요소의 위치를 쓰레드로 할당해 프로그래밍하기 훨씬 수월하게 만들어준다.
대부분의 과학 연산 어플리케이션은 행렬 연산을 주로 사용하기 때문에, 큰 이점이 있다.

두 번째로, GPU 하드웨어에 쓰레드를 할당하기 용이하다.
- 앞서 말했듯, SIMT를 이용하는 하드웨어와 달리, 프로그래밍 모델은 MIMD를 이용한다.
그렇기 때문에 프로그래밍 된 스칼라 쓰레드들을 효율적으로 GPU 하드웨어에 스케쥴링 할 수 있는 방법이 필요하다.
이때 쓰레드를 계층 구조로 구성하면, 컴파일러와 하드웨어만으로 SIMT 구조를 프로그래머에게 노출하지 않으면서, 쓰레드를 하드웨어에 스케쥴링 할 수 있다.
하나의 커널은 하나의 grid로 구성이 되어 있고, GPU 커널을 실행하게 되면 이 grid는 GPU에 할당이 된다.
Grid 내의 각 block들은 Thread Block Scheduler ([이전 챕터 Figure 3-1.](../gpgpu-architectures-chap1/#Figure 3-1))를 통해 SM으로 스케쥴링 되고,
  block 내의 thread들은 warp 단위로 실제 연산 코어인 CUDA core에 스케쥴링 된다.
위 과정은 [Figure 2.](#Figure 2)에서 좀 더 자세히 확인할 수 있다.

|<a name="Figure 2">![alt CUDA 쓰레드 스케쥴링]({{ img_path }}-fig2.jpg)</a>|
|:-------|
|Figure 2. CUDA 쓰레드 스케쥴링 과정. 쓰레드 계층은 일반 글씨체로, GPU 하드웨어는 굵은 글씨체로 표시되어 있다.|


## Memory Hierarchy and Shared Memory

NVIDIA GPU는 프로그래밍의 편의성을 위해, 메모리 구조 역시 하드웨어와 소프트웨어가 분리되어 있다.

### Memory spaces in SW perspective 

소프트웨어 관점에서는 크게 4가지의 메모리 영역이 존재한다.
1. Global Memory
2. Constant Memory
3. Shared Memory
4. Local Memory

이외에도 read-only 메모리인 Texture Memory가 존재하지만, 그래픽스 어플리케이션에 주로 사용되기 때문에 자세히 다루지는 않을 예정이다.

- Global Memory (Per-grid)

  모든 데이터는 따로 메모리 영역을 지정하지 않는 한 암묵적으로 global memory 영역에 저장된다.
  Global memory 영역에 할당된 데이터는 모든 쓰레드에서 접근이 가능하며, free를 시키지 않는 한 서로 다른 grid에서도 접근이 가능하다.
  Global memory 영역에 변수를 명시적으로 선언하고 싶다면 `__global__`이란 키워드를 데이터 타입 앞에 붙여주면 된다.

- Constant Memory (Per-grid)

  Global memory와 마찬가지로 constant memory 영역은 모든 쓰레드에서 접근이 가능하다.
  Read-only 메모리 영역이기 때문에 host에서 초기화를 시켜줘야 한다.
  Constant memory 영역에 변수를 선언하기 위해서는 `__constant__`란 키워드를 사용하면 된다.

- Shared Memory (Per-block)

  Shared memory 영역은 같은 block 내의 쓰레드들만 접근이 가능하다.
  프로그래머가 직접 지정 가능한 영역이며, shared memory 영역을 잘 활용하면 속도를 더더욱 향상시킬 수 있다.
  Shared memory 영역에 변수를 선언하기 위해서는 `__shared__`란 키워드를 사용한다.

- Local Memory (Per-thread)

  Local memory 영역은 쓰레드 내에서만 사용하는 변수를 할당하기 위한 영역이다.
  따로 지정할 필요 없이 지역 변수들은 local memory 영역에 할당된다.

### Memory hierarchy in HW perspective

하드웨어 관점에서 메모리 계층구조는 일반적인 컴퓨터 시스템의 메모리 계층구조와 비슷하다.

- Main memory

  DRAM이나 HBM으로 이뤄진 메인 메모리이다. 가장 크기가 크지만, 접근 속도가 가장 느리다.

- L2 Cache

  L2 캐시는 SRAM으로 이뤄진 메모리이다. 모든 SM이 접근 가능하며, main memory보다 접근 속도가 빠르다.
  캐시 메모리 중 가장 큰 용량을 가지고 있다.

- Read-only memory

  SM 마다 존재하는 메모리이다. Constant Cache ([이전 챕터 Figure 3-2.](../gpgpu-architectures-chap1/#Figure 3-2))가 read-only memory이다.
  SM 마다 존재하기 때문에, L2 Cache보다 접근 속도가 빠르다.
  하지만 크기가 제한적이다.
 
- L1 Cache / Shared memory (SMEM)

  L2 Cache와 마찬가지로 SRAM으로 이뤄진 메모리이다. 각 SM 마다 존재하기 때문에, 다른 SM에서는 접근이 불가능하다.
  프로그래머가 다룰 수 있는 메모리 영역 중에 가장 속도가 빠른 shared memory 영역이 여기에 속한다.
  같은 SM을 공유하는 모든 thread block은 같은 물리적 SMEM을 공유한다.
  이후 자세히 설명하겠지만, L1 Cache와 Shared memory는 같은 하드웨어를 공유한다.

- Registers

  SM 마다 존재하는 register file이다.
  Warp마다 최소 64개의 register (쓰레드 당 2개의 register)를 필요로하며, SM마다 최대 64개의 warp를 동작시킬 수 있다.
  따라서 상당 수의 register가 필요 (A100 기준 SM 당 64K의 32-bit register)하다.

하드웨어적 관점에서의 메모리 계층 구조는 이후 챕터에서 더욱 자세히 다룰 예정이므로, 간단히만 설명하고 넘어가도록 하겠다.

### Relation between memory spaces and memory hierarchy









메모리 영역이 위와 같이 나뉜 이유는 하드웨어 메모리 계층구조에 효율적으로 맵핑해 속도를 빠르게 만들기 위함이다.
여기까지만 읽었다면 어째서 속도가 빨라지는지 알기가 힘든데, 하드웨어 메모리 계층구조에 대한 설명을 먼저 한 뒤,
  소프트웨어 메모리 영역












각각의 SM들은 하나의 shared memory[^3]를 갖고 있다. 따라서 thread block은 shared memory를 통해 서로서로 데이터를 주고 받는다.
Shared memory의 크기는 SM 마다 16KB 매우 작았는데 (Fermi μ-archi. 기준), 최근에는 그 크기가 SM 당 164KB (Ampere μ-archi. 기준)까지 증가했다.
그리고 이 shared memory는 프로그래머에게는 별개의 메모리 영역으로 보이기 때문에, 프로그래머는 직접 이 영역을 사용할 수 있다.

이외에도 NVIDIA GPU에는 여러 종류의 메모리 영역을 가지고 있고, 






## Synchronization Barriers


---

# Instruction Set Architectures


---

결론...

# 참고 자료

- T. M. Aamodt et al., General-purpose graphics processor architectures. Morgan & Claypool Publishers, 2018.
- 
- [Fermi \(microarchitecture\) \- Wikipedia](https://en.wikipedia.org/wiki/Fermi_(microarchitecture))
- [NVIDIA Ampere GPU Architecture Tuning Guide :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)


---
[^1]: Warp는 NVIDIA의 용어이다. NVIDIA에서는 32개의 쓰레드를 묶어 warp라 부르며, AMD에서는 64개의 쓰레드를 묶어 wavefront라 부른다.  
[^2]: Cooperative thread array (CTA)란 용어로도 불린다.
[^3]: NVIDIA에서 사용하는 용어이다. AMD에서는 이러한 scratchpad memory (shared memory)를 local data sotre (LDS)라 부른다.


