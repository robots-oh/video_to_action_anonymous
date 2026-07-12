## Video-To-Action for Bi-Manipulation
- We aim to generate bi-manipulation action from video


### build
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

$ conda activate vtob
$ cd /workspace/video_to_action/src/scripts
$ bash eval_policy_multi_dolbi.sh
```