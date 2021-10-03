# Get it running
Start without any python env sourced.
1. `source /opt/ros/noetic/setup.bash`
1. `catkin_make`
1. `source devel/setup.bash`
1. `source YOUR/PYTHON/ENV`
1. `roslaunch generative_models generative_models_image_vae.launch`
    1.  Note: When using the GAN launch file: don't get confused by error messages ("... cannot find GAN..."). As the loading of the GAN model takes some time this is normal. After a couple of minutes the model is successfully loaded and the error disappears.
