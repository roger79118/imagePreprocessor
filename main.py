from displaypackage.preprocessing import Imagedisplay

if __name__ == '__main__':
    print("Update coming soon!")
    #openimg = Control("input/Hum40-MAP2_GLT1_DAPI.tif")
    openimg = Imagedisplay("input/testrgb.tif")
    threshold = openimg.createpanel()
    print(threshold)