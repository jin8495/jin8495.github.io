---
title: "GP-GPU 구조 시리즈: 시작하기 전"
tags:
  - 컴퓨터구조
  - GPGPU
  - 시리즈
toc: false
---

최근 바쁘다보니 블로그 관리도 거의 하지 않았는데, 논문 리비전도 거의 마무리 지었고 슬슬 여유가 생기는 것 같아서,
  오래전부터 생각하고 있던 컴퓨터구조 게시물 연재를 시작하기로 했다.

어떤 주제를 먼저 연재할지 생각해보다가 내가 연구하고 있는 분야인 GP-GPU를 가장 먼저 연재하기로 결정했다.

{% assign img_path = "/assets/images/hw/gpgpu/gpgpu-architectures-series/2022-04-19-intro" %}
|<a name="썸네일">![alt 썸네일]({{ img_path }}-thumbnail.jpg){: width="200" }</a>|
|:-------|
|저자: Tor M. Aamodt 외 2인<br>출판사: Morgan & Claypool<br>출간일: 2018년 05월 21일|

이 책의 챕터들을 따라가며 내용을 정리하는 식으로 연재를 진행할 예정이다.<br>
책의 저자는 GP-GPU 연구 분야에서 가장 유명한 시뮬레이터인 GPGPU-Sim을 개발한 사람이다.
따라서 연재는 NVIDIA GP-GPU에 대한 이야기가 주가 될 것이다.


논외로 개인적으로 Morgan & Claypool 출판사의 책들을 좋아하는데,
  컴퓨터구조의 다양한 연구 분야들을 잘 소개해주기 때문이다.
판매가 되는 책들이다 보니 review, survey 논문들보다 퀄리티가 좋다.

아래 링크는 Morgan & Claypool에서 출간한 컴퓨터 구조 관련 책들 리스트이다.
<br>
[Morgan & Claypool Publisher - Computer Architecture](https://www.morganclaypoolpublishers.com/catalog_Orig/index.php?cPath=22&sort=2d&series=12&page=1)

**여유부리지 말고 부지런하게 연재해보자! 목표는 연재 종료!**

