# League of legends Match Prediction Using TabNet
Boostcamp AI Tech 2기 최종 팀 프로젝트입니다.

![image1](https://user-images.githubusercontent.com/63408791/147068876-e4ab71f7-ddf1-414f-9aa0-9b6aea0260a1.png)
(출처: https://www.youtube.com/watch?v=Pb4Hw2jteOg)

### 리그 오브 레전드에서 챔피언 밴픽은 게임의 승패에 중대한 영향을 끼칩니다.
- 게임 데이터로 모델을 학습시켜 플레이어 정보와 챔피언 정보로부터 게임의 승패를 예측하고자 하였습니다.
- 사용자는 챔피언 밴픽이 완료된 시점에서 간단하게 자신의 소환사 이름을 입력하고 예측 승률을 볼 수 있습니다.

# 전체 파이프라인
![그림1](https://user-images.githubusercontent.com/33981028/147061171-5232e4c4-5af2-4d83-b195-b7e198f2044e.png)

# TabNet 모델 (https://arxiv.org/pdf/1908.07442.pdf)
![그림2](https://user-images.githubusercontent.com/33981028/147062517-dbf9e408-35d0-4bcd-9a31-f0466c0527c1.png)
구현 시 참고한 코드:(https://github.com/dreamquark-ai/tabnet)

# 데이터 수집
Riot API를 이용해 챌린저, 그랜드마스터, 마스터 티어의 매치데이터 약 20만개를 수집했습니다.

<img width="841" alt="스크린샷 2021-12-22 오후 6 15 11" src="https://user-images.githubusercontent.com/68656752/147067891-dc0419d6-ad8e-4650-a3f5-e8e6bf8e12f8.png">
수집한 매치 데이터는 특정 유저, 특정 챔피언, 특정 라인으로 분류했으며 최대 20경기까지의 평균을 구하고 승/패를 예측할 수 있게 학습시켰습니다.

# Train
### 기본 학습
1. 하이퍼파라미터 및 모델 설정(arguments.py) 설정
2. 학습 시작 
```
python train.py
```
### AutoML
```
python tune.py
```

# Inference
```
python inference.py
```
# 주요 피처(메타) 확인 방법
ImportentFeature.ipynb를 통해 주요 피처를 시각화 할 수 있습니다.

# 서버 실행 방법








# 저장소 구조

```
final-project-level3-nlp-17/
├──code/
│  ├──src/
│  │  ├──modules/
│  │  │  ├──GhostBatchNorm.py
│  │  │  ├──FTabNet.py
│  │  │  └──activations.py
│  │  │  
│  │  └──data_loader.py
│  │
│  ├──ViewImportantFeature.ipynb
│  ├──arguments.py
│  ├──dataset.py
│  ├──false_data.csv
│  ├──inference.py
│  ├──inner_evaluation.py
│  ├──requirements.txt
│  ├──train.py
│  ├──true_data.csv
│  ├──requirements.txt
│  └──tune.py
│  
│──frontend/
│  ├──app/
│  │  ├──main.py
│  │  └──riot.py
│  │
│  ├──statics/
│  │  ├──assets/img/
│  │  │  ├──about.png
│  │  │  ├──main.png
│  │  │  └──result.jpeg
│  │  │
│  │  ├──css/
│  │  │  ├──style_404.css
│  │  │  └──styles.css
│  │  │
│  │  └──js/
│  │     └──styles.cssscripts.js
│  │
│  ├──templates/
│  │  ├──404.html
│  │  └──index.html
│  │
│  ├──Makefile
│  ├──poetry.lock
│  └──pyproject.toml
│
├──.gitignore    
└──README.md
```
