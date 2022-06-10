import cv2
import pandas as pd

# input_path = f'{path}/input'
# output_path = f'{path}/output'

path = '../files/device001/2022-06-02_152107.60'

input_path = f'{path}/input/photo.jpg'
csv_path   = f'{path}/output/pred.csv'
maskcolor = (0,255,255)
nomaskcolor = (0,255,0)
wrongcolor = (0,0,255)
font=cv2.FONT_HERSHEY_SIMPLEX

df = pd.read_csv(csv_path)

# print(df)

img = cv2.imread(input_path)
for i in range(len(df)):
    bbox = df['bbox_result'].iloc[i].split()
    text = df['pred_result'].iloc[i]
    x1, y1, x2, y2 = int(bbox[2]), int(bbox[3]), int(bbox[4]), int(bbox[5])

    if text == 'mask':
        img = cv2.putText(img, text, (x1,y1-3), font, fontScale=1, color=maskcolor, thickness=3)
        img = cv2.rectangle(img,(x1,y1),(x2,y2), maskcolor, thickness=2)
    elif text == 'nomask':
        img = cv2.putText(img, text, (x1,y1-3), font, fontScale=1, color=nomaskcolor, thickness=3)
        img = cv2.rectangle(img,(x1,y1),(x2,y2), nomaskcolor, thickness=2)
    elif text == 'wrong':
        img = cv2.putText(img, text, (x1,y1-3), font, fontScale=1, color=wrongcolor, thickness=3)
        img = cv2.rectangle(img,(x1,y1),(x2,y2), wrongcolor, thickness=2)

cv2.imwrite('./result.jpg',img)

