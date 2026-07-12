# DOLBi: Demonstration-Based Object-Centric Learning for Bimanual Manipulation

## Videos
#### UR - UR
<video src="./video/urur_pourwater_front_main_x5.mp4" width="100%" controls autoplay loop muted></video>

#### UR - Franka
<video src="./video/urfranka_pourwater_front_x5.mp4" width="100%" controls autoplay loop muted></video>

#### UR - Hand
<video src="./video/urhand_pourwater_front_x5.mp4" width="100%" controls autoplay loop muted></video>

#### Two_finger - Suction
<video src="./video/ursuction_pourwater_front_x5.mp4" width="100%" controls autoplay loop muted></video>



## Simulation
#### build
```
$ cd src
$ docker build -t video_to_action .

$ docker run --name vtob_anony \
	-it --runtime nvidia --privileged \
	-v /dev:/dev -v /tmp/.X11-unix:/tmp/.X11-unix \
	-v $HOME/.Xauthority:/root/.Xauthority:rw \
	-v /etc/nv_tegra_release:/etc/nv_tegra_release \
	-v {your_video_to_action_path}/:/workspace/video_to_action \
	-e DISPLAY=$DISPLAY \
	-e QT_X11_NO_MITSHM=1 \
	--net host \
	vtob_anony /bin/bash 

$ bash /workspace/video_to_action/dolbi_install.sh
```

#### Inference
```
$ conda activate vtob
$ cd /workspace/video_to_action/src/scripts
$ bash eval_policy_multi_dolbi.sh
```

#### Training
Comming Soon