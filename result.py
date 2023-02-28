import easyocr
import os
import cv2
import glob
import pandas as pd
import torch
from PIL import Image

class pre:
    def __init__(self, inputpath, savepath, ground_truth,model,choose):
        self.inputpath = inputpath  # 잘라진 이미지(ocr할 이미지)의 경로
        self.savepath = savepath  # 보정된 이미지가 저장될 경로
        self.ground_truth = ground_truth # 리스트 [] 형태로 주어야함
        self.model = model
        self.choose = choose

    # 이미지 보정
    def make_image(self):
        i = 0
        os.makedirs(self.savepath,exist_ok=True)
        os.makedirs("{0}/image".format(self.savepath),exist_ok=True)
        for filename in glob.glob(self.inputpath + "*.jpg"):
            img = cv2.imread(filename)
            img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)  # DPI 300으로 설정
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # black & white

            # avg_blur = cv2.blur(binary, (5, 5))
            # gau_blar = cv2.GaussianBlur(binary, (5,5), 0)
            median_blur = cv2.medianBlur(binary,5)
            erosion = cv2.erode(median_blur, (5,5), iterations=1)
            # delation = cv2.dilate(median_blur, (5, 5), iterations=1)
            RGBimage = cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB)
            PILimage = Image.fromarray(RGBimage)
            i = i + 1
            PILimage.save(self.savepath + 'image/' + f'{i}.jpg', dpi=(300.0, 300.0))

    # 보정된 이미지로 ocr 후 csv파일로 return
    def run_easyocr(self):
        val_img = os.listdir(self.savepath+ 'image/')
        #custom
        reader = easyocr.Reader(['ko'], gpu=True,
                                model_storage_directory=r'C:\Users\USER\PycharmProjects\ocr\demo\user_network_dir',
                                user_network_directory=r'C:\Users\USER\PycharmProjects\ocr\demo\user_network_dir',
                                recog_network='{0}'.format(self.model))
        # pre-trained
        # reader = easyocr.Reader(['ko'],gpu=True)
        list = []
        for f in val_img:
            result = reader.readtext(os.path.join(self.savepath +'image/', f), allowlist ='{0}'.format(self.choose))
            text = ''
            for i in range(len(result)):
                text += result[i][1]
            list.append(text)
        easyocr_df = pd.DataFrame(list, columns=['predict'])
        easyocr_df['ground truth'] = self.ground_truth
        easyocr_df.to_csv('easyocr_results.csv', index=False, encoding='utf-8-sig')

        # accuracy
        ground_truth_len = len(self.ground_truth)
        our_output_len = len(list)
        our_correct = 0

        for i in range(ground_truth_len):
            if self.ground_truth[i] == list[i]:
                our_correct += 1

        our_accuracy = our_correct / our_output_len
        print(our_accuracy)

a=pre('C:/Users/USER/PycharmProjects/ocr/demo/cropped/',
     'C:/Users/USER/PycharmProjects/ocr/demo/saveimage/',
      ['김혜림', '이미희', '김혜림', '82.11.15', '오송생명14로215.107동301호', '010-1234-5678', '여', '홍길동', '043-719-1234',
       '2022년12월26일', '김혜림', '82.11.15', '이미희', '김혜림', '2022.12.26', '홍길동', '청주시흥덕구오송생명14로215.107동301호',
       '010-1234-5678', '여', '국립중앙인체자원은행', '041-719-6531', '2022년12월26일', '김혜림'],
      'kor',
      ""
      )

a.make_image()
a.run_easyocr()