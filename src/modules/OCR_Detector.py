import cv2
import numpy as np

class OCR_Detector():
    def __init__(self):
        self.cfgPath = '/home/ekl/repo/OCR-Yolo/src/yoloModels/OCR/yolov3-tiny-OCR.cfg'
        self.weightsPath = '/home/ekl/repo/OCR-Yolo/src/yoloModels/OCR/yolov3-tiny_final_OCR.weights'
        self.namesPath = '/home/ekl/repo/OCR-Yolo/src/yoloModels/OCR/obj.names'
        self.CONF_THRESH = 0.6
        self.NMS_THRESH = 0.4
        self.setIndex()
        self.loadNetwork()
        self.getOutputLayer()

    def setIndex(self):
        with open(self.namesPath, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def loadNetwork(self):
        # Load the network
        self.net = cv2.dnn.readNetFromDarknet(self.cfgPath, self.weightsPath)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def getOutputLayer(self):
        # Get the output layer from YOLO
        layers = self.net.getLayerNames()
        self.output_layers = [layers[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detector(self, img):
        # Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores
        #img = cv2.imread(img)
        height, width = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 0.00392, (150, 150), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)

        class_ids, confidences, b_boxes, center = [], [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.CONF_THRESH:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    center.append((center_x, center_y))
                    b_boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

        # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
        indices = cv2.dnn.NMSBoxes(b_boxes, confidences, self.CONF_THRESH, self.NMS_THRESH).flatten().tolist()

        colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        for index in indices:
            x, y, w, h = b_boxes[index]
            cv2.rectangle(img, (x, y), (x + w, y + h), colors[index], 2)
            cv2.putText(img, self.classes[class_ids[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colors[index], 2)
        

        letterSequence = self.getAdjustedSequenceLetter(b_boxes,  class_ids, center)
        if letterSequence != False:
            return img, letterSequence
        return False

    def getAdjustedSequenceLetter(self, b_boxes,  class_ids, center):
        distList = []
        for i, cls_id in enumerate(class_ids):
            if cls_id == 21:
                #key_x, key_y, key_w, key_h = b_boxes[i]
                center_x_key, center_y_key = center[i]
                break
        
        for i, cls_id in enumerate(class_ids):
            if cls_id != 21:
                #distBetweenCorners =  int((abs((b_boxes[i][0] + b_boxes[i][2]) - (key_x + key_w) )**2 + abs((b_boxes[i][1]) - (key_y + key_h) )**2)**(1/2))
                distToCenter =  int((abs(center[i][0] - center_x_key)**2 + abs(center[i][1] - center_y_key)**2)**(1/2))
                distList.append((distToCenter, self.classes[class_ids[i]]))
        distList.sort(key=lambda x: x[0])
        distList.reverse()

        letterList = []

        if len(distList) == 3:
            for letter in distList:
                letterList.append(letter[1])
            
            return ''.join(letterList)
        else:
            return False


if __name__ == "__main__":
    img = '/home/ekl/repo/OCR-Yolo/src/modules/cropedImage_screenshot_07.10.2021.png'
    OCR = OCR_Detector()
    img, letterSequence = OCR.detector(img)
    print(letterSequence)
    cv2.imshow('image', img)
    #cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()