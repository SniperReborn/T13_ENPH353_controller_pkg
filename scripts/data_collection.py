#!/usr/bin/env python3

import rospy
import cv2
import csv
import os
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

# rosservice call /gazebo/get_model_state "{model_name: 'B1', relative_entity_name: 'world'}"

# TELEPORT TO BEGINNING OF DIRT 
# rosservice call /gazebo/set_model_state "{model_state: {model_name: 'B1', pose: {position: {x: 0.564, y: -0.2703, z: 0.5}, orientation: {x: 0, y: 0, z: -0.733, w: -0.679}}, twist: { linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0} }, reference_frame: 'world'}}"

# TELEPORT TO BEFORE DIRT BRIDGE
# rosservice call /gazebo/set_model_state "{model_state: {model_name: 'B1', pose: {position: {x: 0.983, y: 2.36, z: 0.5}, orientation: {x: 0, y: 0, z: -0.999, w: 0.0453}}, twist: { linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0} }, reference_frame: 'world'}}"

# TELEPORT TO BEGINNING OF BABY YODA
# rosservice call /gazebo/set_model_state "{model_state: {model_name: 'B1', pose: {position: {x: -3.646, y: 0.663, z: 0.5}, orientation: {x: 0, y: 0, z: -0.797, w: 0.603}}, twist: { linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0} }, reference_frame: 'world'}}"

# TELEPORT TO BEGINNING OF MOUNTAIN A
# rosservice call /gazebo/set_model_state "{model_state: {model_name: 'B1', pose: {position: {x: -4.331, y: -2.27, z: 0.5}, orientation: {x: 0, y: 0, z: -0.361, w: 0.9326}}, twist: { linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0} }, reference_frame: 'world'}}"

# TELEPORT TO BEGINNING OF MOUNTAIN B
# rosservice call /gazebo/set_model_state "{model_state: {model_name: 'B1', pose: {position: {x: -4.324, y: -2.253, z: 0.5}, orientation: {x: 0, y: 0, z: -0.419, w: 0.907}}, twist: { linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0} }, reference_frame: 'world'}}"

# TELEPORT TO MOUNTAIN LEDGE
# rosservice call /gazebo/set_model_state "{model_state: {model_name: 'B1', pose: {position: {x: -2.424, y: -0.086, z: 1.586}, orientation: {x: 0.068, y: 0.0581, z: 0.756, w: -0.648}}, twist: { linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0} }, reference_frame: 'world'}}"

# rosrun 2025_controller data_collection.py _session_name:=dirt_13

# Zero out the speed
# rostopic pub -1 /B1/cmd_vel geometry_msgs/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

class DataCollector:
    def __init__(self):
        rospy.init_node('data_collection_node', anonymous=True)
        self.bridge = CvBridge()
        
        # --- SESSION CONFIGURATION ---
        # Gets name from terminal (e.g. _session_name:=lap1)
        self.session_name = rospy.get_param('~session_name', 'default_run')
        
        # Saves data inside your cnn_trainer folder for better organization
        self.base_dir = os.path.expanduser('~/cnn_trainer/robot_data_dirt')
        self.data_dir = os.path.join(self.base_dir, self.session_name)
        self.image_dir = os.path.join(self.data_dir, 'images')
        self.csv_path = os.path.join(self.data_dir, 'labels.csv')
        
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
            
        file_exists = os.path.isfile(self.csv_path)
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(['image_filename', 'linear_x', 'angular_z'])

        self.image_counter = 0

        # --- SUBSCRIBERS ---
        # Double check these topic names with 'rostopic list'
        image_sub = message_filters.Subscriber('/B1/rrbot/camera1/image_raw', Image)
        cmd_sub = message_filters.Subscriber('/B1/cmd_vel', Twist)
        
        # --- THE FIX: allow_headerless=True ---
        # This allows the synchronizer to work even though Twist messages 
        # don't have a timestamp header.
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, cmd_sub], 
            queue_size=10, 
            slop=0.1,
            allow_headerless=True
        )
        self.ts.registerCallback(self.sync_callback)
        
        rospy.loginfo(f"Collection started for session: {self.session_name}")

    def sync_callback(self, image_msg, cmd_msg):
        linear_x = cmd_msg.linear.x
        angular_z = cmd_msg.angular.z
        
        # Don't save if the robot is stationary
        if abs(linear_x) < 0.01 and abs(angular_z) < 0.01:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        filename = f"frame_{self.image_counter:05d}.jpg"
        image_filepath = os.path.join(self.image_dir, filename)
        cv2.imwrite(image_filepath, cv_image)

        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([filename, linear_x, angular_z])

        self.image_counter += 1
        if self.image_counter % 50 == 0:
            rospy.loginfo(f"Saved {self.image_counter} frames in {self.session_name}")

if __name__ == '__main__':
    try:
        collector = DataCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass