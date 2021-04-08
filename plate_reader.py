import imutils
import cv2
import numpy as np
import copy

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O",
         "P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9"]

class PlateReader:
    def __init__(self, plate_model, id_model, sess, graph):

        self.scale = 12.4/5.7
        self.scale_ID = 12.4/18.08

        self.threshold = 68

        self.lhsv_blue = (114, 118, 0)
        self.uhsv_blue = (180, 255, 255)

        # Grey threshold values working well for darker picture

        self.uhsv_dark = (140, 143, 99)
        self.lhsv_dark = (88, 6, 70)

        # Grey threshold values working well for lighter picture
        self.uhsv_lit = (124, 48, 255)
        self.lhsv_lit = (92, 0, 0)

        self.plate_model = plate_model
        self.id_model = id_model
        self.sess = sess
        self.graph = graph

    # Returns the normalized plate and id in a single image 
    def find(self, image_org):
        
        image = copy.copy(image_org)
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        frame_threshold = cv2.inRange(img_hsv, self.lhsv_blue, self.uhsv_blue)

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
            return "NO_PLATE", [], []

        x,y,w,h = cv2.boundingRect(cnt)
        
        mask = np.zeros(image.shape,np.uint8)
        mask[y:y+h,x:x+w] = image[y:y+h,x:x+w]
        mask_hsv = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)

        plate_threshold_lit = cv2.inRange(mask_hsv, self.lhsv_lit, self.uhsv_lit)
        plate_threshold_dark = cv2.inRange(mask_hsv, self.lhsv_dark, self.uhsv_dark)

        # Reduce Noise Darker (Dilation -> Erosion)
        plate_dilation_lit = cv2.dilate(plate_threshold_lit, kernel, iterations=1) 
        plate_clean_lit = cv2.erode(plate_dilation_lit, kernel, iterations=1)

        plate_dilation_dark = cv2.dilate(plate_threshold_dark, kernel, iterations=1) 
        plate_clean_dark = cv2.erode(plate_dilation_dark, kernel, iterations=1)

        _, contours_lit, h1 = cv2.findContours(plate_clean_lit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, contours_dark, h2 = cv2.findContours(plate_clean_dark, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_lit = sorted(contours_lit,key=cv2.contourArea, reverse = True)[:5]
        contours_dark = sorted(contours_dark,key=cv2.contourArea, reverse = True)[:5]
        screenCnt_lit = None
        screenCnt_dark = None

        detected_lit = 0
        detected_dark = 0

        for c in contours_lit:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx_lit = cv2.approxPolyDP(c, 0.018 * peri, True)
            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx_lit) == 4:
                screenCnt_lit = approx_lit
                break
        for c in contours_dark:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx_dark = cv2.approxPolyDP(c, 0.018 * peri, True)
            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx_dark) == 4:
                screenCnt_dark = approx_dark
                break

        if screenCnt_lit is None and screenCnt_dark is None:
            detected_dark = 0
            detected_lit = 0
            return "NO_PLATE", [], []
            
        if screenCnt_dark is None:
            detected_lit = 1
            
        if screenCnt_lit is None:
            detected_dark = 1 


        if detected_lit == 1:
            #cv2.drawContours(image, [screenCnt_lit], -1, (0, 0, 255), 3)
            approx = approx_lit
        elif detected_dark == 1:
            #cv2.drawContours(image, [screenCnt_dark], -1, (0, 0, 255), 3)
            approx = approx_dark
        else:
            return "NO_PLATE", [], []

        mask = np.zeros(grey.shape,np.uint8)
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
        scale = 12.4/5.7
        left_new = np.abs(tl[1] - bl[1]) * scale
        tl[1] = tl[1] - left_new

        right_new = np.abs(tr[1] - br[1]) * scale
        left_new = np.abs(tl[1] - bl[1]) * scale
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

       # Binarize image

        if detected_lit == 1:
         threshold = 130
        elif detected_dark == 1:
         threshold = 68
    
        _, bin_transformed = cv2.threshold(warped, threshold, 255, cv2.THRESH_BINARY)

        new_height, new_width = bin_transformed.shape

        scale_ID = 12.4/18.08

        plate_ID = bin_transformed[0: int(new_height*scale_ID),: ]
        plate_number = bin_transformed[int(new_height*scale_ID):new_height, :]
        plate_res = cv2.resize(plate_number, dsize=(99*5, 280), interpolation=cv2.INTER_CUBIC)
        
        #cleaning it up
        plate_res[0:55,:] = 255
        plate_res[-55:-1,:] = 255

        h, w = plate_res.shape
        plate_set = []

        # Horizontal crop
        x0 = int(w*0.08)
        dx = int(w*0.172)

        plate_set.append(plate_res[:, x0:x0 + dx])
        plate_set.append(plate_res[:, x0 + dx:x0 + 2*dx])
        plate_set.append(plate_res[:, x0 + 3*dx:x0 + 4*dx])
        plate_set.append(plate_res[:, x0 + 4*dx:x0 + 5*dx])

         # Horizontal crop
        plate_id = cv2.resize(plate_ID, dsize=(300, 240), interpolation=cv2.INTER_CUBIC)[:, 150:]
    
        plate_guess = ""
        probs = []
        id_guess = 0

        return warped, plate_set, plate_id

    def guess(self, image_org): 

        plate_guess = "NO_PLATE"
        plate_probs = []
        id_guess = 0
        id_prob = 0

        image = copy.copy(image_org)
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        frame_threshold = cv2.inRange(img_hsv, self.lhsv_blue, self.uhsv_blue)

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
            return plate_guess, plate_probs, id_guess, id_prob

        x,y,w,h = cv2.boundingRect(cnt)
        
        mask = np.zeros(image.shape,np.uint8)
        mask[y:y+h,x:x+w] = image[y:y+h,x:x+w]
        mask_hsv = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)

        plate_threshold_lit = cv2.inRange(mask_hsv, self.lhsv_lit, self.uhsv_lit)
        plate_threshold_dark = cv2.inRange(mask_hsv, self.lhsv_dark, self.uhsv_dark)

        # Reduce Noise Darker (Dilation -> Erosion)
        plate_dilation_lit = cv2.dilate(plate_threshold_lit, kernel, iterations=1) 
        plate_clean_lit = cv2.erode(plate_dilation_lit, kernel, iterations=1)

        plate_dilation_dark = cv2.dilate(plate_threshold_dark, kernel, iterations=1) 
        plate_clean_dark = cv2.erode(plate_dilation_dark, kernel, iterations=1)

        _, contours_lit, h1 = cv2.findContours(plate_clean_lit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, contours_dark, h2 = cv2.findContours(plate_clean_dark, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_lit = sorted(contours_lit,key=cv2.contourArea, reverse = True)[:5]
        contours_dark = sorted(contours_dark,key=cv2.contourArea, reverse = True)[:5]
        screenCnt_lit = None
        screenCnt_dark = None

        detected_lit = 0
        detected_dark = 0

        for c in contours_lit:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx_lit = cv2.approxPolyDP(c, 0.018 * peri, True)
            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx_lit) == 4:
                screenCnt_lit = approx_lit
                break
        for c in contours_dark:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx_dark = cv2.approxPolyDP(c, 0.018 * peri, True)
            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx_dark) == 4:
                screenCnt_dark = approx_dark
                break

        if screenCnt_lit is None and screenCnt_dark is None:
            detected_dark = 0
            detected_lit = 0
            return "NO_PLATE"
            
        if screenCnt_dark is None:
            detected_lit = 1
            
        if screenCnt_lit is None:
            detected_dark = 1 


        if detected_lit == 1:
            #cv2.drawContours(image, [screenCnt_lit], -1, (0, 0, 255), 3)
            approx = approx_lit
        elif detected_dark == 1:
            #cv2.drawContours(image, [screenCnt_dark], -1, (0, 0, 255), 3)
            approx = approx_dark
        else:
            return plate_guess, plate_probs, id_guess, id_prob

        mask = np.zeros(grey.shape,np.uint8)
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
        scale = 12.4/5.7
        left_new = np.abs(tl[1] - bl[1]) * scale
        tl[1] = tl[1] - left_new

        right_new = np.abs(tr[1] - br[1]) * scale
        left_new = np.abs(tl[1] - bl[1]) * scale
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

        if detected_lit == 1:
            threshold = 80
        elif detected_dark == 1:
            threshold = 68
        
        _, bin_transformed = cv2.threshold(warped, threshold, 255, cv2.THRESH_BINARY)

        new_height, new_width = bin_transformed.shape

        scale_ID = 12.4/18.08

        plate_ID = bin_transformed[0: int(new_height*scale_ID),: ]
        plate_number = bin_transformed[int(new_height*scale_ID):new_height, :]
        plate_res = cv2.resize(plate_number, dsize=(99*5, 280), interpolation=cv2.INTER_CUBIC)
        
        #cleaning it up
        plate_res[0:50,:] = 255
        plate_res[-50:-1,:] = 255

        h, w = plate_res.shape
        plate_set = []

        # Horizontal crop
        x0 = int(w*0.08)
        dx = int(w*0.172)

        plate_set.append(plate_res[:, x0:x0 + dx])
        plate_set.append(plate_res[:, x0 + dx:x0 + 2*dx])
        plate_set.append(plate_res[:, x0 + 3*dx:x0 + 4*dx])
        plate_set.append(plate_res[:, x0 + 4*dx:x0 + 5*dx])

         # Horizontal crop
        id_res = cv2.resize(plate_ID, dsize=(300, 240), interpolation=cv2.INTER_CUBIC)[:, 150:]
    
        plate_guess = ""
        probs = []
        id_guess = 0

        #start = time.time()
        with self.graph.as_default():
            set_session(self.sess)

            for plate in plate_set:
            
                plate = plate.reshape(1, 280, 85, 1)/255
            
                predict = self.plate_model.predict(plate)[0]
                char = str(labels[np.argmax(predict)])

                plate_probs.append(max(predict))
                plate_guess += char
        
            h, w = id_res.shape
            id_res = id_res.reshape(1, h, w, 1)/255
    
            s = time.time()
            predict = self.id_model.predict(id_res)
            print(time.time() - s)
            # max id index + 1 to get id value
            id_guess = np.argmax(predict) + 1
            id_prob = max(predict)

        return plate_guess, plate_probs, id_guess, id_prob

import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

from plate_reader import PlateReader
from imitator import Imitator

# Required setup for running the models in image callback
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

import time 

def main():
    
    license_plate_model = load_model('/home/andrew/ros_ws/src/2020T1_competition/controller/models/plate_number_model_v3.h5')
    id_model            = load_model('/home/andrew/ros_ws/src/2020T1_competition/controller/models/plate_id_model_v3.h5')
    plate_reader = PlateReader(license_plate_model, id_model, sess, graph)
    image = cv2.imread("/home/andrew/ros_ws/src/2020T1_competition/controller/p3_2.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    start = time.time()
    plate, plate_chars, plate_id = plate_reader.find(image)
    p, pprob, i, iprob = plate_reader.guess(image)
    stop = time.time()

    print("Time: {}".format(stop-start))
    
    try:
        cv2.imshow("test", plate)
        cv2.imshow("chars", cv2.hconcat(plate_chars))
        cv2.imshow("id", plate_id)
        print("Guessed: P{}_{}".format(i,p))
        print("ID Prob: {}".format(iprob))
        print("Plate Probs: {}".format(pprob))
    except Exception:
        print(p)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()