import imutils
import cv2
import numpy as np

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O",
         "P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9"]

class PlateReader:
    def __init__(self, model, sess, graph):

        self.scale = 12.4/5.7
        self.scale_ID = 12.4/18.08

        self.threshold = 68

        self.low_H = 102
        self.low_S = 50
        self.low_V = 9

        self.high_H = 138
        self.high_S = 255
        self.high_V = 255

        self.glow_H = 53
        self.glow_S = 6
        self.glow_V = 70

        self.ghigh_H = 140
        self.ghigh_S = 143
        self.ghigh_V = 99

        self.model = model
        self.sess = sess
        self.graph = graph
        
    def identify(self, image):

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        frame_threshold = cv2.inRange(img_hsv, (self.low_H, self.low_S, self.low_V), (self.high_H, self.high_S, self.high_V))

        # Reduce noise (Erosion -> Dilation)
        kernel = np.ones((5,5), np.uint8)
        img_erosion = cv2.erode(frame_threshold, kernel, iterations=1) 
        img_clean = cv2.dilate(img_erosion, kernel, iterations=1)

        _, blue_contours, h = cv2.findContours(img_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        blue_contours = sorted(blue_contours,key=cv2.contourArea, reverse = True)[:5]

        # Sum up all blue contours to capture whole car
        try:
            cnt = np.concatenate(blue_contours)
        except Exception:
            return ["NO_PLATE"], "", []

        x,y,w,h = cv2.boundingRect(cnt)
        
        mask = np.zeros(image.shape,np.uint8)
        mask[y:y+h,x:x+w] = image[y:y+h,x:x+w]
        mask = mask[:,:w//3*2]
        mask_hsv = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)

        plate_threshold = cv2.inRange(mask_hsv, (self.glow_H, self.glow_S, self.glow_V), (self.ghigh_H, self.ghigh_S, self.ghigh_V))

        # Reduce Noise (Dilation -> Erosion)
        plate_dilation = cv2.dilate(plate_threshold, kernel, iterations=1) 
        plate_clean = cv2.erode(plate_dilation, kernel, iterations=1)

        _, contours, h = cv2.findContours(plate_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours,key=cv2.contourArea, reverse = True)[:5]
        screenCnt = None

        for c in contours:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is None:
            detected = 0
        else:
            #cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 3)

            mask = np.zeros(grey.shape,np.uint8)
            new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
            new_image = cv2.bitwise_and(image,image,mask=mask)

            rect = np.zeros((4, 2), dtype = "float32")
            pts = np.transpose(approx, (1,0,2)).reshape((4,2))

            # Re-order the points from approx to [tl, tr, br, bl]
            # Find tl and br
            s = pts.sum(axis = 1)
            rect[0] = pts[np.argmin(s)] #tl
            tl = rect[0] 
            rect[2] = pts[np.argmax(s)] #br
            br = rect[2] 

            # Find bl and tr
            diff = np.diff(pts, axis = 1)
            rect[1] = pts[np.argmin(diff)] #tr
            tr = rect[1]
            rect[3] = pts[np.argmax(diff)] #bl
            bl = rect[3]

            # Extend to find parking ID too
            left_new = np.abs(tl[1] - bl[1]) * self.scale
            tl[1] = tl[1] - left_new

            right_new = np.abs(tr[1] - br[1]) * self.scale
            left_new = np.abs(tl[1] - bl[1]) * self.scale
            tr[1] = tr[1] - right_new

            # Find width of plate
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            # Find height of plate
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype = "float32")

            # compute the perspective transform matrix and then apply it
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(grey, M, (maxWidth, maxHeight))

            _, bin_transformed = cv2.threshold(warped, self.threshold, 255, cv2.THRESH_BINARY)
            
            new_height, new_width = bin_transformed.shape

            plate_ID = bin_transformed[0: int(new_height*self.scale_ID),: ]
            plate_number = bin_transformed[int(new_height*self.scale_ID):new_height, :]

            plate_res = cv2.resize(plate_number, dsize=(99*5, 280), interpolation=cv2.INTER_CUBIC)

            plate_set = []
            # Horizontal crop
            x0 = 0
            dx = 99

            # Vertical crop
            #y0 = int(h*0.55)
            y0 = 1250
            y1 = 1530
            #dy = int(h*0.243)

            plate_set.append(plate_res[:, x0:x0 + dx])
            plate_set.append(plate_res[:, x0 + dx:x0 + 2*dx])
            plate_set.append(plate_res[:, x0 + 3*dx:x0 + 4*dx])
            plate_set.append(plate_res[:, x0 + 4*dx:x0 + 5*dx])

            guess = ""
            probs = []

            for plate in plate_set:

                h, w = plate.shape
                plate = plate.reshape(1, h, w, 1)

                with self.graph.as_default():
                    set_session(self.sess)
                    predict = self.model.predict(plate)[0]
                    char = str(labels[np.argmax(predict)])

                    probs.append(max(predict))
                    guess += char


            
            return plate_set, guess, probs
            
            
        return ["NO_PLATE"], "", []

import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

from plate_reader import PlateReader
from imitator import Imitator

# Required setup for running the models in image callback
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

def main():
    
    license_plate_model = load_model('/home/andrew/ros_ws/src/2020T1_competition/controller/models/Pv0.h5')
    plate_reader = PlateReader(license_plate_model, sess, graph)
    image = cv2.imread("/home/andrew/ros_ws/src/2020T1_competition/controller/p2good.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plates, guess, probs = plate_reader.identify(image)

    print(guess)
    print(probs)
    cv2.imshow("test", cv2.hconcat(plates))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()