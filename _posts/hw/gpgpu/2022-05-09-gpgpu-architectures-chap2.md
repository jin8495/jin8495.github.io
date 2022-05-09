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

대신 CUDA나 OpenCL 같은 GPU 컴퓨팅 API는 MIMD-like 프로그래밍 모델을 채용한다.
따라서 각 쓰레드에 대한 스칼라 프로그래밍이 가능하며, GPU에서 스칼라 쓰레드들이 개별적으로 구동하는 것처럼 보이게 한다.
GPU 하드웨어는 런-타임에 쓰레드를 워프라고 부르는 단위로 묶고 (워프는 NVIDIA에서 부르는 용어이다, AMD는 웨이브프론트라 부른다.),
  워프에 속한 여러 쓰레드를 SIMD와 같은 방식으로 동시에 처리한다.
이런 동작 모델을 SIMD와 구분짓기 위해 SIMT (Single-Instruction Multiple-Threads)라 부르게 되었다.

----

# Execution Model

CUDA는 C/C++을 기반의 API를 제공하기 때문에, 프로그래머가 입문하기 쉽다.
또한 위에서 언급한 바와 같이 스칼라 프로그래밍이 되기 때문에,
  컴파일러가 자동적으로 GPU에 쓰레드들을 워프 단위로 할당한다.

CUDA 프로그래밍 모델에는 크게 3가지의 특징이 있다.

1. CUDA blocks - 쓰레드들을 묶은 그룹
2. Shared Memory - 블록끼리 공유하는 메모리 영역
3. Synchronization Barriers - 모든 쓰레드들이 도달할 때까지 쓰레드의 진행을 가로막는 API

위와 같은 특징들 덕분에, 프로그래머들은 어렵지 않게 CUDA 프로그램을 짤 수 있다.
특징들은 이후 자세하게 설명할 예정이다.

그 전에, 코드 예시를 보고 CPU 코드와 GPU 코드의 차이에 대해 알아보자.

```
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

```
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

GPU 컴퓨팅 어플리케이션의 커널은, [Figure 1.](#Figure 1)과 같이 Grid, Block, Thread의 계층으로 구성된다.

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
그렇기 때문에 프로그래밍 된 스칼라 쓰레드들을 효율적으로 GPU 하드웨어에 이슈할 수 있는 방법이 필요하다.

TODO...


## Shared Memory and Memory Hierarchy

## Synchronization Barriers


---

# GPU Instruction Set Architectures







