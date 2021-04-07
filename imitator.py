import cv2

import tensorflow as tf 
import numpy as np
import copy

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

class Imitator:
    def __init__(self, model, sess, graph):
        self.model = model
        self.sess = sess
        self.graph = graph
    
    def imitate(self, image_org):
        # Takes in the image from the bridge object, should be 
        # a BGR image in the form of an np array 

        # Returns the move (0-6) based on the input image

        image = copy.copy(image_org)
        ret, thresh = cv2.threshold(image,200,255,cv2.THRESH_BINARY)
        bw = cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY)
        res = cv2.resize(bw, dsize=(160,90))
        h, w = res.shape
        img_res = res.reshape(h, w, 1)/255
        img_aug = np.expand_dims(img_res, axis=0)

        move = 0

        # workaround for running model 
        with self.graph.as_default():
            set_session(self.sess)
            predict = self.model.predict(img_aug)[0]
            move = np.argmax(predict)
        
        return move