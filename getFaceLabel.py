import os
import cv2

labelPath = './inference_label_data/'
jpgPath = './inference_data/'
save_path = './faceImg/'

listd = os.listdir(labelPath)

for fileName in listd:
    f = open(labelPath+fileName)
    jpgFileName = fileName[:-4]
    img = cv2.imread(jpgPath+jpgFileName)
    print(jpgFileName)
    cnt = 0
    for coordinate in f.readlines():
        coordinate = coordinate.split()
        copyImg = img[int(coordinate[3]):int(coordinate[5]), int(coordinate[2]):int(coordinate[4])].copy()
        cv2.imwrite(save_path+f"{cnt}"+jpgFileName, copyImg)
        cnt+=1

        # img[]
        # print(j)


# 이미지잘라내려면?
# rectangle에서 