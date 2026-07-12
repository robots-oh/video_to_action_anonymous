
# DOLBi: Demonstration-Based Object-Centric Learning for Bimanual Manipulation

## Videos
#### UR - UR
<video src="https://github.com/user-attachments/assets/7b99c2e5-cfeb-414f-838d-0f5875b61331" width="100%" controls autoplay loop muted></video>

#### UR - Franka
<video src="https://github.com/user-attachments/assets/94d41ca3-3dba-493a-823c-15e8bb6827da" width="100%" controls autoplay loop muted></video>

#### UR - Hand
<video src="https://github.com/user-attachments/assets/1d33f46e-3761-4dc0-8c4b-6a97124a6919" width="100%" controls autoplay loop muted></video>

#### Two_finger - Suction
<video src="https://github.com/user-attachments/assets/19252805-4ff6-4280-b269-ead4c3d88088" width="100%" controls autoplay loop muted></video>


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

#### Evaluation
```
$ conda activate vtob
$ cd /workspace/video_to_action/src/scripts
$ bash eval_policy_multi_dolbi.sh
```
