# Easyocr-Finetune-kordata
[Yujin의 Notion](https://kindhearted-whistle-34a.notion.site/OCR-with-2269e73ab0b542b6973b3926b7ef1240)에 보다 자세한 설명이 기재되어 있음
## 도입
- OCR 시 한글 필기체의 인식률이 저조함을 문제삼아, 메인모델로 정한 Easyocr의 pre-trained 모델을 한글 필기체 데이터로 Fine-tune해 그 성능을 높이고자 한다.

## 프로젝트 진행순서
### 1. 학습 데이터

학습 데이터는 AI hub의 '다양한 형태의 한글 문자 OCR'의 training data 605,741개, validation data 74,809개를 사용함

### 2. 학습 데이터 변환

ai hub에서 다운받은 데이터 *혹은 TextRecognitionDataGenerator로 만든 데이터는* 학습을 위한 [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) 프로젝트에서 요구하는 데이터 구조가 아니므로 바로 사용할 수 없음. So, 데이터 변환 필요

2-1. 'prepare_file_easyocr.ipynb' 파일을 이용해 AI hub의 데이터에서 ground truth 정보를 담은 txt파일 생성

2-2. [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) 프로젝트의 creat_lmdb_dataset.py에 image 데이터와 생성된 gt.txt 파일을 input으로 넣어 lmdb 파일 생성

### 3. 모델 학습

[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) 프로젝트의 train.py에 생성된 lmdb 파일을 input으로 넣어 pre-trained model 학습

## 학습 결과
기존 pre-trained model의 accuracy 55% 에서 Fine-tune한 custom model의 accuracy 92%로 성능 향상을 확인함
