# Import Necessary Modules
import os
import cv2
import numpy as np
import pandas as pd
import easyocr
import glob
import torch
from PIL import ImageFont, ImageDraw, Image
import configparser
from pdf2image import convert_from_path, convert_from_bytes

# Convert PDF as JPG file and Crop an image using the coordinates of ROI
class Detection:
    def __init__(self, pdf_path, savepath, coordinates, model, choose):
        self.pdf_path = pdf_path  # PDF 파일의 경로
        self.coordinates = coordinates  # ini file을 통해 읽어낸 ROI의 좌표
        # ini file의 양식
        # [image_number]
        # roi = (x_pos, y_pos, width, height)
        self.savepath = savepath  # 보정된 이미지가 저장될 경로
        #  self.ground_truth = ground_truth # 리스트 [] 형태로 주어야함
        self.model = model
        self.choose = choose

    def crop_img(self):
        images = convert_from_path(self.pdf_path, fmt = 'jpg')

        # Make Folder to save JPG file
        os.makedirs(self.savepath + 'images', exist_ok = True)
        os.makedirs(self.savepath + 'cropped', exist_ok = True)

        for i in range(len(images)):
            images[i].save(self.savepath + 'images/' + 'images_' + str(i) + '.jpg')
        
        return images

    # Define the Fuction which removes the lines on the image
    def remove_line(self, img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        minLineLength = 70 
        maxLineGap = 50

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
        if str(type(lines)) == "<class 'NoneType'>":
            return img
        else:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                img_line_removed = cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 5)
    
        return img_line_removed

    # 이미지 보정
    def preprocess_image(self, img):
        img_resize = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)  # DPI 300으로 설정
        gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # black & white
        avg_blur = cv2.blur(binary, (5, 5))
        delation = cv2.dilate(avg_blur, (5, 5), iterations=1)
        RGBimage = cv2.cvtColor(delation, cv2.COLOR_BGR2RGB)

        return RGBimage

    def cropimg(self):
        images = {}
        for i in range(len(self.crop_img())):
            images['images_' + str(i)] = cv2.imread(self.savepath + 'images/' + 'images_' + str(i) + '.jpg')
        cropped_image = {}
        
        # Read the ini files of ROI coordinates
        config = configparser.ConfigParser()
        config.read(self.coordinates, encoding = 'UTF-8')

        # Get the section
        sections = dict(config.items())

        # Get a dictionary of all keys and values in the section
        coordinates = {}
        for section, values in sections.items():
            if section != 'DEFAULT':
                coordinates[section] = dict(config.items(section))
        
        for section, values in coordinates.items():
            cropped_image[section] = []
            image = images[section]
            for key, crd in coordinates[section].items():
                cropped_image[section].append(self.remove_line(self.preprocess_image(image[eval(crd)[1] : eval(crd)[1] + eval(crd)[3], eval(crd)[0] : eval(crd)[0] + eval(crd)[2]])))
        
        return cropped_image


    # 보정된 이미지로 ocr 후 csv파일로 return
    def run_easyocr(self):
        cropped_img = self.cropimg()
        #custom
        reader = easyocr.Reader(['ko'], gpu=True,
                                model_storage_directory=r'user_network_dir',
                                user_network_directory=r'user_network_dir',
                                recog_network='{0}'.format(self.model))
        # pre-trained
        # reader = easyocr.Reader(['ko'],gpu=True)
        list = []
        for section, values in cropped_img.items():
            for img in cropped_img[section]:
                result = reader.readtext(img, allowlist ='{0}'.format(self.choose))
                text = ''
                for i in range(len(result)):
                    text += result[i][1]
                list.append(text)
        easyocr_df = pd.DataFrame(list, columns=['predict'])
        # easyocr_df['ground truth'] = self.ground_truth
        easyocr_df.to_csv('easyocr_results.csv', index=False, encoding='utf-8-sig')


class CheckboxDetector:
    def __init__(self, image_path, coordinates, line_min_width=15):
        self.line_min_width = line_min_width
        self.image_path = image_path
        self.coordinates = coordinates
        self.image = cv2.imread(image_path)


    # Define the function to detect checkbox
    def detect_box(self):
        gray_scale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        th1,img_bin = cv2.threshold(gray_scale, 150, 225, cv2.THRESH_OTSU)
        kernal_h = np.ones((1, self.line_min_width), np.uint8)
        kernal_v = np.ones((self.line_min_width, 1), np.uint8)
        img_bin_h = cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal_h)
        img_bin_v = cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal_v)
        img_bin_final = img_bin_h | img_bin_v
        final_kernel = np.ones((3,3), np.uint8)
        img_bin_final = cv2.dilate(img_bin_final, final_kernel,iterations=1)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
        return stats, labels


    # Get the coordinates of checkbox
    def coordinates_checkbox(self, max_boxwidth=100, min_boxwidth=10):
        image = self.image
        stats, labels = self.detect_box()
        selected = []
        for x,y,w,h,area in stats[2:]:
            if w < max_boxwidth and w > min_boxwidth:
                bl = (x, y)
                tr = (x+w, y+h)
                selected.append((bl, tr))
        return selected
    

    # Draw the rectangles on detected box
    def draw_box(self, max_boxwidth=100, min_boxwidth=10):
        image = self.image
        selected = self.coordinates_checkbox(max_boxwidth, min_boxwidth)
        for bl, tr in selected:
            cv2.rectangle(image, bl, tr, (0, 255, 0), 3)
        return image


    # Detect whether checkbox is checked or not
    def detect_checked(self, threshold = 210):
        # Detect whether checkbox is checked or not
        cropped = {}
        i = 0
        selected = self.coordinates_checkbox()
        for bl, tr in selected:
            cropped[i] = self.image[bl[1]:tr[1], bl[0]:tr[0]]
            i += 1

        # Get the average pixel intensity of the checkbox region
        average_intensity = {}
        for i in range(len(selected)):
            average_intensity[i] = np.mean(cropped[i])


        for i in range(len(selected)):
            if average_intensity[i] < threshold:
                average_intensity[i] = [np.mean(cropped[i]), 1]  #  If Checkbox is checked, 1
            else:
                average_intensity[i] = [np.mean(cropped[i]), 0] #  If Checkbox is not checked, 0
        
        agreed = {}
        not_agreed = {}
        i = 0
        for bl, tr in selected:
            if bl[0] > 792 and tr[0] < 1088:
                agreed[i] = [selected[i], average_intensity[i]]
            i += 1

        i = 0
        for bl, tr in selected:
            if bl[0] > 1092 and tr[0] < 1388:
                not_agreed[i] = [selected[i], average_intensity[i]]
            i += 1

        return average_intensity, agreed, not_agreed
    
    # Detect words in box
    def detect_words(self):
        image = self.image
        selected = self.coordinates_checkbox(max_boxwidth=1000, min_boxwidth=10)
        words = {}
        reader = easyocr.Reader(['ko'], gpu = True)
        for i in range(len(selected)):
            result = reader.readtext(image[selected[i][0][1] : selected[i][1][1], 
            selected[i][0][0] : selected[i][1][0]])
            if len(result) != 0:
                words[i] = result[0][1]
        return words

    # Return the categories of checkbox and Detect whether checkbox is checked or not
    def checkbox_detection(self):
        
        config = configparser.ConfigParser()
        config.read(self.coordinates, encoding = 'UTF-8')
        sections = dict(config.items())
        data = {}
        for section, values in sections.items():
            data[section] = dict(config.items(section))
        
        categories = []
        for key, values in data['Categories'].items():
            coordinate = eval(values)
            categories.append([key, ((eval(values)[0], eval(values)[1]), 
            (eval(values)[0] + eval(values)[2] , eval(values)[1] + eval(values)[3]))])

        agreed = list(self.detect_checked()[1].values())
        not_agreed = list(self.detect_checked()[2].values())
        checkbox_detection = {}
        for i in range(len(agreed)):
            bl = agreed[i][0][0]
            tr = agreed[i][0][1]
            for j in range(len(categories)):
                bl_c = categories[j][1][0]
                tr_c = categories[j][1][1]
                if bl[1] > bl_c[1] and tr[1] < tr_c[1]:
                    checkbox_detection[categories[j][0]] = [agreed[i][1][1]]
        
        for i in range(len(not_agreed)):
            bl_n = not_agreed[i][0][0]
            tr_n = not_agreed[i][0][1]
            for j in range(len(categories)):
                bl_c = categories[j][1][0]
                tr_c = categories[j][1][1]
                if bl_n[1] > bl_c[1] and tr_n[1] < tr_c[1]:
                    checkbox_detection[categories[j][0]].append(not_agreed[i][1][1])
        
        return checkbox_detection # If checked 1, not 0



if __name__ == '__main__':
    detection = Detection(pdf_path = 'documents_ocr.pdf', savepath = '', coordinates = 'roi_coordinates.ini' , model = 'kor', choose= '')
    detection.run_easyocr()

