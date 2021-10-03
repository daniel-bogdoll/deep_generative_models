#!/usr/bin/env python

import rospy
from generative_models.msg import Custom

# -------------------------------
# find carla module
# -------------------------------
import glob
import os
import sys
try:
    sys.path.append(glob.glob('/opt/carla-nopng-0.9.6/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def talker():
    # -------------------------------
    # Initialize ROS
    # -------------------------------
    rospy.init_node('custom_talker', anonymous=True)
    rospy.logwarn("(%s) Initialize.", rospy.get_name())

    # fetch Carla host and port from the ~private namespace
    host    = rospy.get_param('~host', '127.0.0.1')
    port    = rospy.get_param('~port', 2000)
    rate    = rospy.get_param('~frequency', 10) #10hz
    delta_t = rospy.get_param('~fixed_delta_t', 1. / rate)

    # initialize publisher with custom message type 'Custom'
    pub = rospy.Publisher('custom_chatter', Custom)

    # -------------------------------
    # Initialize connection to Carla
    # -------------------------------
    rospy.loginfo("(%s) Connecting to Carla...", rospy.get_name())
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    rospy.logwarn("(%s) Connected to Carla host: '%s', port: %s.", rospy.get_name(), host, port)
    rospy.logwarn("(%s) Fixed synchronous time step is: %.3f seconds.", rospy.get_name(), delta_t)

    # -------------------------------
    # set Carla to synchronous mode
    # -------------------------------
    settings = world.get_settings()
    settings.fixed_delta_seconds = delta_t
    world.apply_settings(settings)

    rospy.logwarn("(%s) Ready.", rospy.get_name())

    # -------------------------------
    # main loop
    # -------------------------------
    loop_rate = rospy.Rate(rate)
    while not rospy.is_shutdown():
        # create msg and fill with content
        msg = Custom()
        msg.name = "ROS User"
        msg.age = 4
        rospy.loginfo(msg)
        pub.publish(msg)

        # synchronously step Carla
        world.tick()

        # -------------------------------
        # Example usage of Carla's PythonAPI:
        # get current list of carla actors
        # and filter for vehicles
        # -------------------------------        
        carla_actors = world.get_actors()
        carla_vehicles = carla_actors.filter('vehicle*')
        carla_walkers  = carla_actors.filter('walker*')
        rospy.loginfo("(%s) Currently %d vehicles and %d walkers in Carla.", rospy.get_name(), len(carla_vehicles), len(carla_walkers))

        # sleep
        loop_rate.sleep()

    # -------------------------------
    # Stop
    # -------------------------------
    rospy.logwarn("(%s) Terminating.", rospy.get_name())

    # unset synchronous mode in Carla
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: 
        pass