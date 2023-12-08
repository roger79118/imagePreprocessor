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
        self.origin = self.load(path)
        self.frame = self.resize(self.origin)

        # create subframe
        self.subframe = {0:None, 1:None, 2:None}
        self.subframe[0], self.subframe[1], self.subframe[2] = cv2.split(self.frame)  

        # create variables
        self.lowerLimits = np.array([0, 0, 0])
        self.upperLimits = np.array([255, 255, 255])
        
        # create threshold
        self.threshold = [[0, 255], [0, 255], [0, 255]]


    def load(self, path):
        return cv2.imread(path)
    

    def resize(self, img):
        zoomfactor = 300
        im = img.copy()
        if im.shape[1] > zoomfactor or im.shape[0] > zoomfactor:
            rfx = im.shape[1]//zoomfactor
            rfy = im.shape[0]//zoomfactor
            frame = cv2.resize(im, (0, 0), fx=1/(0.5+rfx), fy=1/(0.5+rfy)) 
        else:
            frame = im
        return frame


    def singleColerFrame(self, col, img):
        zeros = np.zeros_like(img)
        mergeList = []
        for i in range(3):
            if i == col:
                mergeList.append(img)
            else:
                mergeList.append(zeros)
        return cv2.merge(mergeList)


    def setLowVal(self, val, col):
        self.lowerLimits[col] = val
        val = int(val/65535*255)
        self.threshold[col][0] = val
        self.processImage()

    def setHighVal(self, val, col):
        self.upperLimits[col] = val
        val = int(val/65535*255)
        self.threshold[col][1] = val
        self.processImage()
    
    
    def processImage(self):
        b_outimage = self.setThreshold(self.subframe[0], 0, self.threshold)
        g_outimage = self.setThreshold(self.subframe[1], 1, self.threshold)
        r_outimage = self.setThreshold(self.subframe[2], 2, self.threshold)
        
        outimage = cv2.merge([b_outimage, g_outimage, r_outimage])
        b_outimage = self.singleColerFrame(0, b_outimage)
        g_outimage = self.singleColerFrame(1, g_outimage)
        r_outimage = self.singleColerFrame(2, r_outimage)

        cv2.imshow("bFrame", b_outimage)
        cv2.imshow("gFrame", g_outimage)
        cv2.imshow("rFrame", r_outimage)
        cv2.imshow("Frame", outimage)
        cv2.moveWindow('rFrame',400,0)
        cv2.moveWindow('bFrame',400,350)
        cv2.moveWindow('gFrame',850,0)
        cv2.moveWindow('Frame',850,350)


    def setThreshold(self, img, col, threshold):
        thres_img = img.copy()
        thres_img[img<=threshold[col][0]] = 0
        thres_img[img>=threshold[col][1]] = 0
        thres_img.astype(np.uint8)
        return thres_img


    def createpanel(self):
        # create trackbars
        cv2.namedWindow("Control")
        cv2.createTrackbar("lowRed", "Control", 0,65535, lambda x: self.setLowVal(x,2))
        cv2.createTrackbar("highRed", "Control", 65535,65535, lambda x: self.setHighVal(x,2))
        cv2.createTrackbar("lowGreen", "Control", 0,65535, lambda x: self.setLowVal(x,1))
        cv2.createTrackbar("highGreen", "Control", 65535,65535, lambda x: self.setHighVal(x,1))
        cv2.createTrackbar("lowBlue", "Control", 0,65535, lambda x: self.setLowVal(x,0))
        cv2.createTrackbar("highBlue", "Control", 65535,65535, lambda x: self.setHighVal(x,0))
        cv2.moveWindow('Control',50,200)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.threshold, self.lowerLimits, self.upperLimits


#openimg = Control("input/Hum40-MAP2_GLT1_DAPI.tif")
openimg = Control("input/testrgb.tif")
threshold = openimg.createpanel()
print(threshold)