#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

def move_forward():
    # 1. Initialize the ROS node
    rospy.init_node('taiga_mover', anonymous=True)
    
    # 2. Create a publisher for the /cmd_vel topic
    # The message type is 'Twist'
    score_publisher = rospy.Publisher('/score_tracker', String, queue_size=1)

    while score_publisher.get_num_connections() == 0:
        rospy.loginfo("Waiting for scoreboard subscriber...")
        rospy.sleep(0.1)
    first_msg = "Team13,multi21,0,INIT"
    score_publisher.publish(first_msg)
    rospy.loginfo("Published first message")

    velocity_publisher = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=10)
    
    # 3. Define the movement parameters
    speed = 0.5        # Meters per second
    distance = 3.048   # 10 feet in meters
    duration = distance / speed  # Time = Distance / Speed
    
    # 4. Create the message object
    vel_msg = Twist()
    vel_msg.linear.x = speed  # Move forward along the x-axis
    vel_msg.angular.z = 0     # No turning
    
    # 5. Move for the calculated duration
    rospy.loginfo("Moving forward 10 feet...")
    start_time = rospy.Time.now().to_sec()
    
    while (rospy.Time.now().to_sec() - start_time) < duration:
        velocity_publisher.publish(vel_msg)
        rospy.sleep(0.1) # Small delay to avoid flooding the CPU
    
    # 6. Force the robot to stop
    vel_msg.linear.x = 0
    velocity_publisher.publish(vel_msg)
    rospy.loginfo("Goal reached. Stopping.")

if __name__ == '__main__':
    try:
        move_forward()
    except rospy.ROSInterruptException:
        pass