#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import os
import tensorflow as tf
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class CNNDriver:
    def __init__(self):
        rospy.init_node('cnn_driver')
        
        # 1. Initialize the TFLite Interpreter
        model_path = os.path.expanduser('~/cnn_trainer/pavement_model.tflite')
        
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            rospy.loginfo("TFLite Model loaded successfully!")
        except ValueError as e:
            rospy.logerr(f"Failed to load TFLite model. Is the path correct? Error: {e}")
            return
            
        # 2. Get input and output tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.bridge = CvBridge()
        
        # 3. ROS Publishers and Subscribers (Check these namespaces for Fizzer!)
        self.image_sub = rospy.Subscriber("/B1/rrbot/camera1/image_raw", Image, self.image_callback)
        self.cmd_pub = rospy.Publisher("/B1/cmd_vel", Twist, queue_size=1)
        
        rospy.loginfo("CNN Driver Node Started. Ready to follow the line!")

    def preprocess(self, cv_image):
        h, w = cv_image.shape[:2]
        img = cv_image[int(h * 0.33):h, :] 
        img = cv2.resize(img, (200, 66))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # DELETE the img = img / 255.0 line!
        
        # Keep this, but note it's just converting 0-255 to floats
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # Preprocess the image
            input_tensor = self.preprocess(cv_image)
            
            # --- TFLITE INFERENCE ---
            # 1. Put the image into the input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            
            # 2. Run the math
            self.interpreter.invoke()
            
            # 3. Pull the results out of the output tensor
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            lin_x = float(prediction[0][0])
            ang_z = float(prediction[0][1])

            ang_z = ang_z * 2.2

            # Publish
            move_cmd = Twist()
            move_cmd.linear.x = lin_x
            move_cmd.angular.z = ang_z
            self.cmd_pub.publish(move_cmd)

        except Exception as e:
            rospy.logerr(f"Inference Error: {e}")

if __name__ == '__main__':
    try:
        driver = CNNDriver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass