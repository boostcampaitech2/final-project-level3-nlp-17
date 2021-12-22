# League of legends Match Prediction Using TabNet
Boostcamp AI Tech 2기 최종 팀 프로젝트입니다.
<br>

# 전체 파이프라인
![그림1](https://user-images.githubusercontent.com/33981028/147061171-5232e4c4-5af2-4d83-b195-b7e198f2044e.png)

# TabNet 모델 (https://arxiv.org/pdf/1908.07442.pdf)
![그림2](https://user-images.githubusercontent.com/33981028/147062517-dbf9e408-35d0-4bcd-9a31-f0466c0527c1.png)
구현 시 참고한 코드:(https://github.com/dreamquark-ai/tabnet)

# 데이터 수집

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
