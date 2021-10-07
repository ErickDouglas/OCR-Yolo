import cv2
import numpy as np

class capaceteDetector():
    def __init__(self):
        self.cfgPath = '/home/ekl/repo/OCR-Yolo/src/yoloModels/Capacete/yolov3-tiny_capacete.cfg'
        self.weightsPath = '/home/ekl/repo/OCR-Yolo/src/yoloModels/Capacete/yolov3-tiny_last_capacete.weights'
        self.namesPath = '/home/ekl/repo/OCR-Yolo/src/yoloModels/Capacete/obj.names'
        self.CONF_THRESH = 0.4
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

        blob = cv2.dnn.blobFromImage(img, 0.00392, (512, 288), swapRB=True, crop=False)
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
        
        cropedImageList = list()
        for index in indices:
            x, y, w, h = b_boxes[index]
            cv2.rectangle(img, (x, y), (x + w, y + h), colors[index], 2)
            cropedImageList.append(img[y:y + h, x:x + w])

        return cropedImageList, img

if __name__ == "__main__":
    img = '/home/ekl/repo/OCR-Yolo/src/modules/IMG_20211001_135513_1.jpg'
    capacete = capaceteDetector()
    listOfCropedImage = capacete.detector(img)

    for cropped_image in listOfCropedImage:
        cv2.imshow('cropedImage', cropped_image)
        #cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()