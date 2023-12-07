# import cv2
# import numpy as np

# img = cv2.imread("input/testrgb.tif")

# cv2.imshow('oxxostudio', img)

# contrast = 0    # 初始化要調整對比度的數值
# brightness = 0  # 初始化要調整亮度的數值
# cv2.imshow('oxxostudio', img)


# # 定義調整亮度對比的函式
# def adjust(i, c, b):
#     output = i * (c/100 + 1) - c + b    # 轉換公式
#     output = np.clip(output, 0, 255)
#     output = np.uint8(output)
#     cv2.imshow('oxxostudio', output)

# # 定義調整亮度函式
# def brightness_fn(val):
#     global img, contrast, brightness
#     brightness = val - 100
#     adjust(img, contrast, brightness)

# # 定義調整對比度函式
# def contrast_fn(val):
#     global img, contrast, brightness
#     contrast = val - 100
#     adjust(img, contrast, brightness)

# cv2.createTrackbar('brightness', 'oxxostudio', 0, 200, brightness_fn)  # 加入亮度調整滑桿
# cv2.setTrackbarPos('brightness', 'oxxostudio', 100)
# cv2.createTrackbar('contrast', 'oxxostudio', 0, 200, contrast_fn)      # 加入對比度調整滑桿
# cv2.setTrackbarPos('contrast', 'oxxostudio', 100)

# keycode = cv2.waitKey(0)
# cv2.destroyAllWindows()







import numpy as np
import cv2

class Control:
    def __init__(self, path) -> None:
        self.path = path
        
        # Load image
        im = cv2.imread(self.path)
        if im.shape[1] > 800 or im.shape[0] > 600:
            rfx = im.shape[1]//800
            rfy = im.shape[0]//600
            self.frame = cv2.resize(im, (0, 0), fx=1/rfx, fy=1/rfy) 
        else:
            self.frame = im
        # create variables
        self.lowerLimits = np.array([0, 0, 0])
        self.upperLimits = np.array([255, 255, 255])
        
        # create threshold
        self.threshold = [[0, 255], [0, 255], [0, 255]]


    # functions to modify the color ranges
    def setLowVal(self, val, col):
        self.lowerLimits[col] = val
        self.threshold[col][0] = val
        self.processImage()

    def setHighVal(self, val, col):
        self.upperLimits[col] = val
        self.threshold[col][1] = val
        self.processImage()

    def processImage(self):
        # treshold and mask image
        thresholded = cv2.inRange(self.frame, self.lowerLimits, self.upperLimits)
        outimage = cv2.bitwise_and(self.frame, self.frame, mask = thresholded)
        #show img
        cv2.imshow("Frame", outimage)

    def createpanel(self):
        #show initial image
        cv2.imshow("Frame", self.frame)


        # create trackbars
        cv2.namedWindow("Control")
        cv2.createTrackbar("lowRed", "Control", 0,255, lambda x: self.setLowVal(x,2))
        cv2.createTrackbar("highRed", "Control", 255,255, lambda x: self.setHighVal(x,2))
        cv2.createTrackbar("lowGreen", "Control", 0,255, lambda x: self.setLowVal(x,1))
        cv2.createTrackbar("highGreen", "Control", 255,255, lambda x: self.setHighVal(x,1))
        cv2.createTrackbar("lowBlue", "Control", 0,255, lambda x: self.setLowVal(x,0))
        cv2.createTrackbar("highBlue", "Control", 255,255, lambda x: self.setHighVal(x,0))
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return self.threshold


openimg = Control("input/Hum40-MAP2_GLT1_DAPI.tif")
threshold = openimg.createpanel()
print(threshold)