# 인파기상청 — 유동인구가 밀집하면 그 지역의 미세 기후는 정말 변하는가
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://crowd-environment-analysis.streamlit.app/)
## 팀원
- 20319 이마루
- 20502 김경민

## 질문
"특정 구역의 유동인구가 변화하면 그 지역의 미세 기후 및 환경(온도, 습도, 소음 등등)은 그에 따라 변화하는가?"

## 왜 이 주제인가 (팀이 씁니다)
팀이 이 주제를 고른 이유 한 문단.
- 20319 이마루: 나는 인파가 몰리는 구역을 지날 때마다 평소보다 공기가 더 답답하고 온도가 높게 느껴지는 현상에 관심이 생겨서, 이 현상이 단순한 심리적 체감인지 아니면 인간의 체열과 활동이 실제로 그 지역의 환경 데이터 값을 변화시키는 물리적 요인인지 명확히 확인하고 싶었다. 이러한 가설을 머신 러닝을 통해 객관적인 수치로 증명해보고 싶어서 선택하였다.
- 20502 김경민: 나는 날씨와 도시 환경은 사람들의 이동 방식에 영향을 줄 것이다 라고 생각했다. 실제 데이터를 분석해 이러한 요인이 보행량과 어떤 관계가 있는지 알아보고 싶었다.

## 데이터
- 출처(주소):스마트서울 도시데이터 센서(S-DoT) 환경정보(https://data.seoul.go.kr/dataList/OA-15969/S/1/datasetView.do), 스마트서울 도시데이터 센서(S-DoT) 유동인구 측정 정보(https://data.seoul.go.kr/dataList/OA-15964/S/1/datasetView.do)
- 기간·행 수·열 수: 2020.05.06 ~ 현재 / 열 수(API 출력 항목)는 환경정보 58개, 유동인구 8개 / 행 수..?
사용한 열:
- 환경정보(IotVdata017): CGG(자치구), DONG(행정동), MSRMT_HR(측정시간) / 예측할 값 AVG_TP(온도), AVG_HUM(습도), AVG_NIS(소음)
- 유동인구(IotVdata018): AUTONOMOUS_DISTRICT(자치구), ADMINISTRATIVE_DISTRICT(행정동), SENSING_TIME(측정시간) / 입력값 VISITOR_COUNT(방문자 수)
- 병합 기준: 두 API의 시리얼 번호 체계가 달라, 자치구 + 행정동 + 1시간 단위 시간으로 병합
- 개인 단위 정보 없음 확인: 예 (사람의 수를 집계한 수치와 환경 센서 측정값만 제공되며, 개인 식별 정보는 없음.)
