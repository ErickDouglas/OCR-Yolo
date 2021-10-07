from src.modules.capaceteDetector import capaceteDetector
from src.modules.OCR_Detector import OCR_Detector
import cv2

def main():
    img = '/home/ekl/repo/OCR-Yolo/src/modules/IMG_20211001_135513_1.jpg'
    capacete = capaceteDetector()
    OCR = OCR_Detector()
    listOfCropedHelmet = capacete.detector(img)

    for imgCapacete in listOfCropedHelmet:
        imgOCR = OCR.detector(imgCapacete)
        cv2.imshow('image', imgOCR)
        #cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

    