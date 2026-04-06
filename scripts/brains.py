#!/usr/bin/env python3
import cv2
import numpy as np
import tensorflow as tf
import rospy
import os

class TFLiteBrain:
    def __init__(self, model_path):
        full_path = os.path.expanduser(model_path)
        try:
            self.interpreter = tf.lite.Interpreter(model_path=full_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            rospy.loginfo(f"Brain loaded: {model_path}")
        except Exception as e:
            rospy.logerr(f"Failed to load model at {full_path}. Error: {e}")

    def preprocess(self, cv_image):
        h, w = cv_image.shape[:2]
        img = cv_image[int(h * 0.33):h, :] 
        img = cv2.resize(img, (200, 66))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return np.expand_dims(img, axis=0).astype(np.float32)

    def get_command(self, cv_image):
        """Returns (linear_x, angular_z) based on the image"""
        input_tensor = self.preprocess(cv_image)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        lin_x = float(prediction[0][0])
        # Include your scaling and safety clipping here!
        ang_z = float(prediction[0][1]) * 2.2 
        ang_z = max(min(ang_z, 2.5), -2.5) 
        
        return lin_x, ang_z