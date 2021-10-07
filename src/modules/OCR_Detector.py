import cv2
import numpy as np

class OCR_Detector():
    def __init__(self):
        self.cfgPath = '/home/ekl/repo/OCR-Yolo/src/yoloModels/OCR/yolov3-tiny-OCR.cfg'
        self.weightsPath = '/home/ekl/repo/OCR-Yolo/src/yoloModels/OCR/yolov3-tiny_final_OCR.weights'
        self.namesPath = '/home/ekl/repo/OCR-Yolo/src/yoloModels/OCR/obj.names'
        self.CONF_THRESH = 0.8
        self.NMS_THRESH = 0.1
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

        class_ids, confidences, b_boxes = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.CONF_THRESH:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

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
        return img

if __name__ == "__main__":
    img = '/home/ekl/repo/OCR-Yolo/src/modules/cropedImage_screenshot_07.10.2021.png'
    OCR = OCR_Detector()
    img = OCR.detector(img)

    cv2.imshow('image', img)
    #cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()