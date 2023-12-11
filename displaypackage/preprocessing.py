import numpy as np
import cv2


class Imagedisplay:
    def __init__(self, path) -> None:
        self.path = path
        
        # Load image
        self.origin = self.load(path)
        self.frame = self.resize(self.origin)

        # create subframe
        self.subframe = {}
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
        #b_outimage = self.singleColerFrame(0, b_outimage)
        #g_outimage = self.singleColerFrame(1, g_outimage)
        #r_outimage = self.singleColerFrame(2, r_outimage)
        
        b_outimage = self.thresVisual(self.subframe[0], 0, self.threshold, "magenta")
        g_outimage = self.thresVisual(self.subframe[1], 1, self.threshold, "magenta")
        r_outimage = self.thresVisual(self.subframe[2], 2, self.threshold, "magenta")

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


    def thresVisual(self, img, col, threshold, color):
        labels = {"cyan":[255, 255, 10],
                  "white": [255, 255, 255],
                  "yellow": [100, 255, 255],
                  "magenta": [255, 10, 255]
                }
        renew = []
        for i in range(3):
            
            if labels[color][i]:
                thres_img = np.zeros_like(img)
                if i == col:
                    thres_img[img<threshold[col][0]] = img[img<threshold[col][0]]
                    thres_img[img>=threshold[col][0]] = labels[color][i]
                    thres_img[img>threshold[col][1]] = img[img>threshold[col][1]]
                else:
                    thres_img[img<threshold[col][0]] = 0
                    thres_img[img>=threshold[col][0]] = labels[color][i]
                    thres_img[img>threshold[col][1]] = 0
            else:
                if i == col:
                    thres_img = img.copy()
                else:
                    thres_img = np.zeros_like(img)

            renew.append(thres_img) 
        return cv2.merge(renew)


    def setThresFrame(self, img):
        return cv2.merge([img, img, img])


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


