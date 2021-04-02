import cv2

import tensorflow as tf 
import numpy as np

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

class Imitator:
    def __init__(self, model):
        self.model = model
    
    def imitate(self, image, sess, graph):
        # Takes in the image from the bridge object, should be 
        # a BGR image in the form of an np array 

        # Returns the move (0-6) based on the input image

        bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        res = cv2.resize(bw, dsize=(320,180))
        h, w = res.shape
        img_res = res.reshape(h, w, 1)/255
        img_aug = np.expand_dims(img_res, axis=0)

        move = 0

        # workaround for running model 
        with graph.as_default():
            set_session(sess)
            predict = self.model.predict(img_aug)[0]
            move = np.argmax(predict)
        
        return move