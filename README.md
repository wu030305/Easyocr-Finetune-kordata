## 도입
- 한글 필기체의 인식률이 저조함을 문제삼아, 메인모델로 정한 Easyocr의 pre-trained 모델을 한글 필기체 데이터로 Fine-tune해 그 성능을 높이고자 한다.

## 프로젝트 진행순서
1. 학습 데이터

학습 데이터는 AI hub의 '다양한 형태의 한글 문자 OCR'의 training data 605,741개, validation data 74,809개를 사용하였다.

2. 학습 데이터 변환

ai hub에서 다운받은 데이터 *혹은 TextRecognitionDataGenerator로 만든 데이터는* 학습을 위한 [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) 프로젝트에서 요구하는 데이터 구조가 아니고, 바로 사용할 수 없음
그래서 데이터 변환이 필요함 

1. 'prepare_file_easyocr.ipynb' 파일을 이용해 AI hub의 데이터를 ' 

