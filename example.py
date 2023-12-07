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
        self.frame = self.load(self.path)
        self.b_frame, self.g_frame, self.r_frame = self.rgbFrame(self.frame)

        # create variables
        self.lowerLimits = np.array([0, 0, 0])
        self.upperLimits = np.array([255, 255, 255])
        
        # create threshold
        self.threshold = [[0, 255], [0, 255], [0, 255]]


    def load(self, path):
        # Load image
        im = cv2.imread(path)
        if im.shape[1] > 800 or im.shape[0] > 600:
            rfx = im.shape[1]//400
            rfy = im.shape[0]//300
            frame = cv2.resize(im, (0, 0), fx=1/rfx, fy=1/rfy) 
        else:
            frame = im
        return frame


    def rgbFrame(self, img):
        
        b, g, r = cv2.split(img)
        zeros = np.zeros_like(img[:,:,0])
        b_frame = cv2.merge([b, zeros, zeros])
        g_frame = cv2.merge([zeros, g, zeros])
        r_frame = cv2.merge([zeros, zeros, r])

        return b_frame, g_frame, r_frame
        #cv2.imshow("rFrame", cv2.cvtColor(self.r_frame,cv2.COLOR_GRAY2RGB))
        #cv2.imshow("gFrame", cv2.cvtColor(self.g_frame,cv2.COLOR_GRAY2RGB))
        #cv2.imshow("bFrame", cv2.cvtColor(self.b_frame,cv2.COLOR_GRAY2RGB))
        pass




    # functions to modify the color ranges
    def setLowVal(self, val, col):
        self.lowerLimits[col] = val
        self.threshold[col][0] = val
        self.processImage(col)

    def setHighVal(self, val, col):
        self.upperLimits[col] = val
        self.threshold[col][1] = val
        self.processImage(col)

    def processImage(self, col):
        # treshold and mask image

        splitLowerLimits = np.array([self.threshold[0][0], self.threshold[1][0], self.threshold[2][0]])
        splitUpperLimits = np.array([self.threshold[0][1], self.threshold[1][1], self.threshold[2][1]])


        r_thresholded = cv2.inRange(self.r_frame, splitLowerLimits, splitUpperLimits)
        g_thresholded = cv2.inRange(self.g_frame, splitLowerLimits, splitUpperLimits)
        b_thresholded = cv2.inRange(self.b_frame, splitLowerLimits, splitUpperLimits)
        
        r_outimage = cv2.bitwise_and(self.r_frame, self.r_frame, mask = r_thresholded)
        g_outimage = cv2.bitwise_and(self.g_frame, self.g_frame, mask = g_thresholded)
        b_outimage = cv2.bitwise_and(self.b_frame, self.b_frame, mask = b_thresholded)

        thresholded = cv2.inRange(self.frame, self.lowerLimits, self.upperLimits)
        outimage = cv2.bitwise_and(self.frame, self.frame, mask = thresholded)
        

        #show img
        cv2.imshow("Frame", outimage)
        cv2.imshow("rFrame", r_outimage)
        cv2.imshow("gFrame", g_outimage)
        cv2.imshow("bFrame", b_outimage)
        cv2.moveWindow('rFrame',400,0)
        cv2.moveWindow('bFrame',400,350)
        cv2.moveWindow('gFrame',850,0)
        cv2.moveWindow('Frame',850,350)
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
        cv2.moveWindow('Control',50,200)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return self.threshold, self.lowerLimits, self.upperLimits


openimg = Control("input/Hum40-MAP2_GLT1_DAPI.tif")
threshold = openimg.createpanel()
print(threshold)