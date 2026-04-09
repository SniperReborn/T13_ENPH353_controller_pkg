import tensorflow as tf
import cv2
import numpy as np
import os
import rospy

class TFLiteBrain:
    def __init__(self, model_path):
        full_path = os.path.expanduser(model_path)
        # Use 1 thread to avoid starving Gazebo of CPU
        self.interpreter = tf.lite.Interpreter(model_path=full_path, num_threads=1)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        rospy.loginfo(f"Brain Loaded: {model_path}")

    def preprocess(self, cv_image):
        h, w = cv_image.shape[:2]
        # 1. Exact same crop as training
        cv_image = cv_image[int(h * 0.33):h, :] 
        # 2. Exact same resize
        cv_image = cv2.resize(cv_image, (200, 66))
        # 3. BGR to RGB
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        #cv_image = (cv_image.astype(np.float32) / 127.5) - 1.0
        cv_image = cv_image.astype(np.float32)
        
        return np.expand_dims(cv_image, axis=0)

    def get_command(self, cv_image):
        input_tensor = self.preprocess(cv_image)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        lin_x = float(prediction[0][0])
        ang_z = float(prediction[0][1]) * 2.25
        
        # Start with 1.0 gain. Only increase if the robot is "lazy"
        # but 2.25 is likely way too high for stability.
        # print(f"Model Think: Lin: {lin_x:.2f} | Ang: {ang_z:.2f}")
        return lin_x, ang_z