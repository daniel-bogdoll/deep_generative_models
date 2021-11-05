# Instructions for Setting Up and Running the Online Pipeline

## Brief Instructions
*If you already have a setup and have run everything at least once.*

Start without any python env sourced.
1. `cd THIS/FOLDER`
1. `source /opt/ros/noetic/setup.bash`
1. `catkin_make`
1. `source devel/setup.bash`
1. `source YOUR/PYTHON/ENV`
1. `roslaunch generative_models generative_models_image_vae.launch`    

## Installation and Setup Carla-ROS-Bridge

*Use the [Carla-ROS-bridge docs](https://carla.readthedocs.io/projects/ros-bridge/en/latest/ros_installation_ros1/) if you have sudo access. Otherwise, follow the instructions below.*

Required and not covered here:
- ROS Noetic
- Carla Simulator
- Python >=3.6

Adapted from the [Carla-ROS-bridge docs](https://carla.readthedocs.io/projects/ros-bridge/en/latest/ros_installation_ros1/). The setup is intended for running Carla and the Carla-ROS-bridge via remote desktop and then connecting other nodes remotely from another machine.

1. Set up catkin ws
    1. `mkdir -p ~/carla-ros-bridge/src`
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

## Run Online Pipeline
The following steps should be performed via remote desktop, because Carla will spawn a window, and streaming the x-window via ssh does not work well.

1. Run Carla: `/opt/carla-simulator/CarlaUE4.sh`
1. Source Carla-ROS-bridge ws: `source ~/carla-ros-bridge/catkin_ws/devel/setup.bash`
1. Run Carla-ROS-bridge: `roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch`
1. Now you can connect to the topics published by the Carla-ROS-bridge remotely (or locally, of course) via the hosts ip, port, and topic name.
    1. `export ROS_MASTER_URI=IP_OF_HOST:11311`
    1. Now you should see the host's ros-topics via `rostopic list`.
1. Optional: if you have installed the python packages into a python venv, activate the it now.
1. Run our `generative_models` package.
	1. Open new terminal
	1. `source catkin_ws/devel/setup.bash`
	1. Launch either the image compression, or the lidar compression: `roslaunch generative_models generative_models_image_gan.launch`
        - Alternative launchfiles: `generative_models_image_vae.launch`, `generative_models_lidar.launch`
        - Note: When using the GAN launch file: don't get confused by error messages ("... cannot find GAN..."). As the loading of the GAN model takes some time this is normal. After a couple of moments the model is successfully loaded and the error disappears.

## Important Note when Running the Lidar Online Pipeline
When using the VAE to compress/decompress the preprocessed Lidar pointclouds a slight adjustement in the VAE decompress method is necessary to view the real pointclouds:
1. Go to `deep_generative_models/vae/third_party/CompressAI/compressai/models/priors.py`
2. Remove the clamp() from line 305 in the decompress method of ScaleHyperprior from: So `x_hat = self.g_s(y_hat).clamp_(0, 1)` **to** `x_hat = self.g_s(y_hat)`. Otherwise the decompressed Lidar Output will be a cube.
3. Reinstall the compressAi package by `cd _compressAI_Path` and run `pip install .`
