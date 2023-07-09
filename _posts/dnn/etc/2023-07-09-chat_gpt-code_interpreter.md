---
title: "Chat-GPT 4의 베타 버전으로 출시된 Code Interpreter 사용 후기"
tags:
  - 딥러닝
  - ChatGPT
---

최근에 Chat-GPT Plus를 결제하고 나서, 이것저것 실험을 해보고 있었다.

그러던 와중, 3일 전에 Code Interpreter라는 기능이 베타 버전으로 사용할 수 있게 됐다고 해서, 간단하게 사용해봤다.

{% assign img_dir = "/assets/images/dnn/etc/2023-07-09-chat_gpt_code_interpreter/" %}

---------

# Introduction

23년 7월 6일, Code Interpreter라는 기능이 Chat-GPT 4의 베타 버전으로 추가됐다.  
[ChatGPT — Release Notes](https://help.openai.com/en/articles/6825453-chatgpt-release-notes)

데이터를 넣으면 자동으로 분석한 뒤, 사용자가 입력한 질문에 맞게 재가공을 해준다.  
잠깐 사용해봤는데, 잘만 활용하면 진짜 도움이 많이 될 것 같았다.

# Code Interpreter

Kaggle에서 가져온 Amazon 주가 데이터를 csv로 받아서 업로드해봤다.
그리고 어느 정도 수준으로 처리가 가능한지 간단한 질문들을 몇 개 던져봤다.

일단, csv 파일을 받자마자 파일명과 column의 key 값을 보고 어떤 것에 대한 데이터인지 정확히 유추해냈다.
그리고 `pandas` 패키지를 이용한 파이썬 스크립트를 작성하고, 직접 돌려본 후 결과 값을 반환했다.

다음으로 그래프를 그려보라고 시킨 뒤, 특정 기간 동안 변동폭이 컸던 날짜들을 알려달라고 질문했다.
그래프도 나름 잘 그려주고 원하는 정보들을 정리해서 보여주기까지했다.

|<a name="Figure 1">![alt Example]({{ img_dir }}screencapture-chat-openai-2023-07-09-22_34_29.png)</a>|
|:---:|
|Figure 1. Chat-GPT4의 Code Interpreter 사용 예시|

위 그림이 실제로 Chat-GPT의 Code Interpreter를 사용해본 예시이다.

조금 더 고급 질문들은 던져보지 않았지만, 이 정도 간단한 수준의 질문만 잘 처리해주더라도 상당히 도움이 될 것으로 보인다.
아마 현재 마이크로소프트에서 MS office에 도입할 계획이라는 Co-pilot 역시 위와 비슷한 과정으로 데이터를 분석하고 처리해줄 것 같기는 하다.

아무튼, Code Interpreter라는 기능을 잘만 활용한다면 큰 도움이 될 것 같다. 어떻게 활용할지는 천천히 고민해봐야겠다.
