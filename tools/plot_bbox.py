import cv2

blue_color = (255,0,0)

# img = cv2.imread('./single2-100210014.jpg', cv2.IMREAD_COLOR)

# img = cv2.rectangle(img, (15, 56), (60, 131), color=blue_color, thickness=2)
# img = cv2.rectangle(img, (152, 35), (187, 105), color=blue_color, thickness=2)

img = cv2.imread('./tools/42_Car_Racing_Nascar_42_370.jpg', cv2.IMREAD_COLOR)

img = cv2.rectangle(img, (111, 46), (147, 83), color=blue_color, thickness=2)
img = cv2.rectangle(img, (39, 41), (71, 73), color=blue_color, thickness=2)
img = cv2.rectangle(img, (178, 35), (212, 67), color=blue_color, thickness=2)


cv2.imshow('holy', img)
cv2.waitKey(0)
