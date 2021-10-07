from src.modules.capaceteDetector import capaceteDetector
from src.modules.OCR_Detector import OCR_Detector
import cv2

def main():
    #img = '/home/ekl/repo/OCR-Yolo/src/modules/IMG_20211001_135513_1.jpg'
    cap = cv2.VideoCapture('/home/ekl/repo/OCR-Yolo/src/modules/VID_20211005_152856.mp4')

    #imgReference = cv2.imread(img)
    
    capacete = capaceteDetector()
    OCR = OCR_Detector()
    
    # Read until video is completed
    while(cap.isOpened()):
        
        # Capture frame-by-frame
        ret, imgReference = cap.read()
        
        width = int(1080)
        height = int(720)
        dim = (width, height)
        imgReference = cv2.resize(imgReference, dim, interpolation = cv2.INTER_AREA)

        if ret == True:
            try:
                listOfCropedHelmet, imgReference = capacete.detector(imgReference)
            except:
                pass
            else:
                for imgCapacete in listOfCropedHelmet:
                    try:
                        imgOCR, letterSequence= OCR.detector(imgCapacete)
                    except:
                        pass
                    else:
                        imgReference[0:imgOCR.shape[0], 0:imgOCR.shape[1]] = imgOCR
                        cv2.putText(imgReference, str(letterSequence), (0, imgReference.shape[0]-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,0), 1,cv2.LINE_AA)
                        print(letterSequence)
            
            cv2.imshow('image', imgReference)
            #cv2.imshow("image", img)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else: 
            break

    cap.release()
    cv2.destroyAllWindows()

    

if __name__ == "__main__":
    main()
