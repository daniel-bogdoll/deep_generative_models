#!/usr/bin/env python
import rospy
from generative_models.msg import Custom

def callback(data):
    rospy.loginfo("%s is age: %d" % (data.name, data.age))

def listener():
    rospy.init_node('custom_listener', anonymous=True)
    rospy.Subscriber("custom_chatter", Custom, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()