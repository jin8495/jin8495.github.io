---
title: Name mangling
date: 2025-04-12
tags: 
  - 컴파일러
  - 시스템소프트웨어
toc: true
---

컴파일러에서는 이름으로 인한 충돌을 해결하기 위해 function, structure, class, 또는 data type 등에 unique name을 부여한다. 이 과정에서 name mangling이란 테크닉을 사용한다.

C++과 같은 언어에서는 네임 스페이스를 구분하거나, 함수 오버로딩 또는 클래스 내 동일 이름의 변수를 고유하게 만드는 데 사용되고, 파이썬에서는 프라이빗 속성이나 메서드의 이름을 숨기기 위해 사용된다.

# C++에서의 name mangling

**주요 상황**
- **클래스**에 같은 이름의 변수가 존재할 때, 이름을 고유하게 만들어줌.
- **오버로딩**된 함수들에 대해 서로 다른 이름을 가지게 만들어줌.

예를 들어, 두 클래스에서 같은 이름의 변수를 선언하면, C++ 컴파일러는 내부적으로 변수 이름을 클래스 이름과 함께 조합하여 고유한 이름으로 변환한다.

```cpp
class A {
private:
    int var; // 내부적으로 _ZN1A3variE로 변환될 수 있음
};

class B {
private:
    int var; // 내부적으로 _ZN1B3variE로 변환될 수 있음
};
```

**예시**
```cpp
#include <iostream>

class A {
public:
    void show() { std::cout << "Class A" << std::endl; }
};

class B {
public:
    void show() { std::cout << "Class B" << std::endl; }
};

int main() {
    A a;
    B b;
    a.show(); // _ZN1A4showEv로 변환
    b.show(); // _ZN1B4showEv로 변환
    return 0;
}
```
# Python에서의 name mangling

**주요 상황**
- **클래스 내부**의 변수나 메서드의 이름이 **프라이빗**으로 설정될 때 사용된다. 이름 앞에 **언더스코어 두 개**(`__`)를 붙여서 사용하면, 파이썬은 이를 내부적으로 변환하여 프라이빗 속성으로 취급한다.

```python
class MyClass:
    def __init__(self):
        self.__private_var = 42  # 내부적으로 _MyClass__private_var로 변환
```
- 이렇게 하면 클래스 외부에서 직접 접근하는 것이 어렵게 되어, **캡슐화**를 강화한다.

**예시**
```python
class MyClass:
    def __init__(self):
        self.__private = "I am private!"

    def __private_method(self):
        print("This is a private method")

obj = MyClass()
print(obj._MyClass__private)  # 직접 접근 가능하지만 일반적이지 않음
```


---
# Related
- [Name mangling - Wikipedia](https://en.wikipedia.org/wiki/Name_mangling)
