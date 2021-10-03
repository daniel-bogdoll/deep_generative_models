# Instructions for setting up the Carla-ROS-Bridge on the Workstations

## Moritz: Encoder Node Approach
1. Create new Package: 
	- catkin_create_pkg ros_package std_msgs roscpp rospy derived_object_msgs
	- Recompile: "catkin_make"
	- Make the workspace visible to the file system: "source devel/setup.bash"
2. Add new folder "scripts" for the python scripts: 
	- "mkdir scripts"
	- add encoder python script: "cp <location of .py file> <location of scripts folder>"
	- Change file permission: "chmod +x encoder.py"

# Test Encoder Node
1. Run Carla:
	- Open new terminal and run: "/opt/carla-simulator/CarlaUE4.sh"
2. Launch Carla-Ros-bridge:
	- Open new terminal and run: "roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch"
3. Run python Encoder Script:
	- Open new terminal
	- Cd to the scripts folder
	- Run python script: "python encoder.py"

# Helpful Links:
- https://medium.com/swlh/part-3-create-your-first-ros-publisher-and-subscriber-nodes-2e833dea7598
- https://medium.com/swlh/7-simple-steps-to-create-and-build-our-first-ros-package-7e3080d36faa

## Johannes: Install Carla-ROS-Bridge

Adapted from the [Carla-ROS-bridge docs](https://carla.readthedocs.io/projects/ros-bridge/en/latest/ros_installation_ros1/). The setup is intended for running Carla and the Carla-ROS-bridge via remote desktop and then connecting other nodes remotely from another machine.

1. Set up catkin ws
    1. `mkdir -p carla-ros-bridge/src`
    2. `git clone --recurse-submodules https://github.com/carla-simulator/ros-bridge.git catkin_ws/src/ros-bridge`
2. As we cannot install the dependencies with rosdep (no sudo), we install the required packages manually
    1. `git clone https://github.com/astuff/astuff_sensor_msgs catkin_ws/src/astuff_sensor_msgs`
    1. `pip install -U transforms3d pygame`
        -   TODO We currently do not know how to do this using a venv.
3. Build workspace
    1. `source /opt/ros/noetic/setup.bash`
    1. `cd catkin_ws`
    2. `catkin_make`
    - If you get an error for missing dependencies in a package within `astuff_sensor_msgs`, such as `ibeo_msgs`, just delete the folder (`rm -r ibeo_msgs`). We only need the delphi msgs.
4. Add Carla modules to your Python Path: `export PYTHONPATH=$PYTHONPATH:/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg:$CARLA_PYTHON_PATH`
    - If you will run this often, consider adding the export to your .bashrc
5. The following steps should be performed via remote desktop, because Carla will spawn a window, and streaming the x-window via ssh does not work    
    1. Run Carla: `/opt/carla-simulator/CarlaUE4.sh`
    6. Source Carla-ROS-bridge ws: `source ~/carla-ros-bridge/catkin_ws/devel/setup.bash`
    7. Run Carla-ROS-bridge: `roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch`
6. Now you can connect to the topics published by the Carla-ROS-bridge remotely (or locally, of course) via the hosts ip, port, and topic name
    - TODO add example subscription in Python