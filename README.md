# Fire Detection using PyTorch

본 프로젝트는 PyTorch를 활용하여 이미지 내 화재 발생 여부(**Fire / Non-Fire**)를 식별하는 딥러닝 분류 모델을 구현한 것입니다. Kaggle의 Fire Dataset을 사용하며, 사전 학습된 모델(Transfer Learning)과 직접 설계한 CNN 모델(Training from Scratch) 두 가지 방식을 비교 구현하였습니다.

## 개요

화재 이미지를 탐지하기 위해 두 가지 접근 방식을 사용했습니다.
* **Transfer Learning**: ImageNet으로 사전 학습된 **ResNet50** 모델을 미세 조정(Fine-tuning)하여 학습합니다.
* **Training from Scratch**: 독자적인 **CNN 아키텍처**를 정의하고 처음부터 학습합니다.


## 데이터셋

Kaggle의 `phylake1337/fire-dataset`을 사용합니다. 코드는 실행 시 Kaggle API를 통해 자동으로 데이터를 다운로드하도록 구성되어 있습니다.

**데이터셋 구조:**

* `./data/fire_dataset/fire_images`: 화재가 포함된 이미지 (Label: **Fire**)
* `./data/fire_dataset/non_fire_images`: 화재가 없는 이미지 (Label: **Non-Fire**)

**주의:** Kaggle API 사용을 위해 `kaggle.json` 파일이 설정되어 있어야 합니다.

## 파일 구성 및 상세 설명

### 1. `train.ipynb` (Transfer Learning)

사전 학습된 ResNet50 모델을 기반으로 화재 탐지 모델을 학습합니다.

* **모델:** `models.resnet18(weights='models.ResNet18_Weights.IMAGENET1K_V1')`
* **특징:**
    * ResNet18의 마지막 Fully Connected Layer를 2개의 클래스(Fire, Non-Fire)에 맞게 수정합니다.
    * 사전 학습된 가중치를 사용하여 적은 데이터와 에포크로도 높은 성능을 기대할 수 있습니다.

### 2. `train_scratch.ipynb` (Custom CNN)

모델 구조를 직접 정의하여 밑바닥부터 학습을 진행합니다.

* **모델:** 사용자 정의 CNN (Convolutional Neural Network)
* **특징:**
    * 외부 가중치 없이 초기화된 상태에서 학습을 시작합니다.
    * 데이터셋에 특화된 경량화된 모델 구조를 실험할 때 사용됩니다.

## 학습 설정 (Hyperparameters)

두 코드 공통적으로 적용된 주요 학습 파라미터는 다음과 같습니다.

* **이미지 전처리:**
    * Resize: 224x224
    * Augmentation: RandomHorizontalFlip, RandomRotation
    * Normalization: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]
* **Optimizer:** Adam (Learning Rate: 0.001)
* **Loss Function:** CrossEntropyLoss
* **Scheduler:** StepLR (10 에포크마다 학습률 0.1배 감소)
* **Epochs:** 30

## 결과 시각화

학습이 완료된 후 다음과 같은 시각화 결과를 제공합니다.

* **Loss & Accuracy Graph:** 학습(Train) 및 검증(Validation) 과정에서의 손실과 정확도 변화 추이.
* **Prediction Visualization:** 테스트 데이터셋에 대한 실제 라벨(T)과 예측 라벨(P)을 이미지와 함께 출력.
    * T: F: 실제 화재 (True: Fire)
    * P: NF: 예측 비화재 (Predicted: Non-Fire)