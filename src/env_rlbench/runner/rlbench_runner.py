# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.env_runner.base_runner import BaseRunner

import glob
import os
import pickle
import numpy as np
from termcolor import cprint

from pyrep.const import RenderMode
from pyrep.objects.shape import Shape
from rlbench.backend import utils
from rlbench.backend.const import *
from rlbench.action_modes.arm_action_modes import (
    ArmActionMode,
    EndEffectorPoseViaIK,
    EndEffectorPoseViaPlanning,
    assert_action_shape,
)
from rlbench import ObservationConfig
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
from rlbench.backend import task as rlbench_tasks
from env_rlbench.policy.dp3_policy import RLBenchDP3Policy
from env_rlbench.policy.subgoal_policy import RLBenchSubGoalPolicy
from env_rlbench.runner.rl_bench_dataset import _create_obs_config, _get_action_mode
from env_rlbench.runner.rl_bench_env import CustomRLBenchEnv


from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
from env_rlbench_peract.utils.rlbench_utils_vtob import MyEnvironmentPeract  # 얘 보류

from diffusion_policy_3d.model.common.geodesic_loss import GeodesicLoss
from diffusion_policy_3d.model.clip.clip import build_model, load_clip, tokenize
from utils.pose_utils import euler_from_quaternion, get_average_pose
from PIL import Image
import torchvision.transforms as T_trans
from types import SimpleNamespace


class RLBenchRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 root_dir,
                 task_name,
                 has_lang_emb,
                 use_fp,
                 fp_cam_name,
                 pose_estimation_wrapper,
        ):
        super().__init__(output_dir)

        # ------------------------
        # |    Pose Estimation   |
        # ------------------------
        self.pose_estimation_wrapper = pose_estimation_wrapper

        # ------------------------
        # |        RLBench       |
        # ------------------------

        # create env
        self.task = task_name
        self.env = self._create_env(task=self.task, root_dir=root_dir, use_fp=use_fp, fp_cam_name=fp_cam_name, pose_estimation_wrapper=self.pose_estimation_wrapper, headless=False)
        self.env.launch()
        self.use_fp = use_fp
        
        # load demo for evalution
        self.root_dir = root_dir
        self.demo_path_list = glob.glob(os.path.join(self.root_dir, "episodes", "episode*", "low_dim_obs.pkl"))
        assert len(self.demo_path_list) > 0
        self.demo_path_list.sort()

        # language embeding
        self.has_lang_emb = has_lang_emb
        # if self.has_lang_emb:
        self._lang_token_embs = self.get_language_embedding()
        cprint(f"[RLbench Runner] has_lang_emb: {has_lang_emb}", "yellow")



    def _create_env(self, task, root_dir,  use_fp=False, fp_cam_name=None, pose_estimation_wrapper=None, headless=True):
        _camera_names = ["front", "left_shoulder", "right_shoulder", "wrist", "overhead"]
        _ds_img_size = 512
        # _ds_img_size = 256
        _data_raw_path = root_dir # !! This path does not do anything, but it needs to exist to prevent errors...
        
        observation_config = _create_obs_config(   
            _camera_names,
            [_ds_img_size, _ds_img_size],
        )

        action_mode=BimanualMoveArmThenGripper(BimanualEndEffectorPoseViaPlanning(), BimanualDiscrete()) 


        # single-arm tasks
        task_files_single = [
            t.replace(".py", "")
            for t in os.listdir(rlbench_tasks.TASKS_PATH)
            if t != "__init__.py" and t.endswith(".py")
        ]
        # bimanual tasks
        task_files_bimanual = [
            t.replace(".py", "")
            for t in os.listdir(rlbench_tasks.BIMANUAL_TASKS_PATH)
            if t != "__init__.py" and t.endswith(".py")
        ]
        if task not in task_files_single and task not in task_files_bimanual:
            raise ValueError("Task %s not recognised!." % task)
        
        is_bimanual = task in task_files_bimanual
        task_class = task_file_to_task_class(task, bimanual=is_bimanual)



        ####################################
        return CustomRLBenchEnv(
            task_class=task_class,
            observation_config=observation_config,
            action_mode=action_mode,
            episode_length=200,
            dataset_root=_data_raw_path,  
            headless=headless,
            time_in_state=True,
            use_fp=use_fp,
            fp_cam_name=fp_cam_name,
            pose_estimation_wrapper=pose_estimation_wrapper,
        )




    def get_language_embedding(self):
        # load clip model
        model, _ = load_clip("RN50", jit=False, device='cuda')
        clip_model = build_model(model.state_dict())
        clip_model.to('cuda')
        del model

        # load all descriptions                
        description_path = os.path.join(self.root_dir, "all_variation_descriptions.pkl")
        print(f"Loading demo from the path {description_path}")
        with open(description_path, "rb") as fin:
            descriptions = pickle.load(fin) # dict of VAR: DESC

        cur_task_id = 0 # cur_task_id is always zero as there is only one task

        # pre-generate all language embeddings
        _lang_token_embs = {}
        n_variation = len(descriptions)
        for var_id in range(n_variation):
            cur_description = descriptions[var_id]

            tokens = tokenize(cur_description).numpy()
            token_tensor = torch.from_numpy(tokens).to('cuda')
            sentence_emb, token_embs = clip_model.encode_text_with_embeddings(token_tensor)

            # print(cur_description)      # 5 sentence
            # print(token_embs.shape)     # [5, 77, 512]
            # print(sentence_emb.shape)   # [5, 1024]

            lang_token_embs = sentence_emb[0].float().detach().cpu().numpy()[None, :]

            _lang_token_embs[f"{cur_task_id}_{var_id}"] = lang_token_embs
        return _lang_token_embs
    

    def get_observation(self, obs=None):
        if obs is None:
            obs = self.env.env._scene.get_observation()
        return obs

    @staticmethod
    def toBimanualAction(left_action, right_action):
        # 앞 9개: 오른팔, 뒤 9개: 왼팔
        # 각 9개: 팔 포즈(7) + 그리퍼(1) + 충돌무시(1)
        action = np.zeros(18)
        action[:9] = right_action
        action[9:] = left_action
        return action
    
    
    def run(self, policy: BasePolicy=None, tag="latest", save_video=True):

        # get policy
        if policy is None: 
            subgoal_policy = RLBenchSubGoalPolicy(self.env)
        else:
            subgoal_policy = RLBenchDP3Policy(self.env, policy, self.use_fp)
            
        self.get_action = subgoal_policy.get_action
    

        success_list = []
        for demo_idx, demo_path in enumerate(self.demo_path_list[0:25]):
            self._estimator0 = None
            self._estimator1 = None
            self._camera_anchor_pose = None
            RUN_FINISHED = False
            N_TRIAL = 1
            success = False  # Initialize success before the trial loop
            for cur_trial in range(N_TRIAL):
                try:
                    # load demo
                    print(f"{demo_idx+1}/{len(self.demo_path_list)} Loading demo from the path {demo_path} (trial: {cur_trial})")
                    with open(demo_path, "rb") as fin:
                        demo = pickle.load(fin)
                    with open(demo_path.replace("low_dim_obs.pkl", "variation_number.pkl"), 'rb') as f:
                        variation_number = pickle.load(f)


                    #load image
                    episode_dir = os.path.dirname(demo_path)
                    goal_image0_path = os.path.join(episode_dir, 'grasp_obj0_rgb','0.png')
                    goal_image1_path = os.path.join(episode_dir, 'grasp_obj1_rgb','0.png')
                    image0 = Image.open(goal_image0_path).convert("RGB")
                    transform = T_trans.Compose([
                        T_trans.Resize((224, 224)),
                        T_trans.ToTensor(),
                    ])
                    image1 = Image.open(goal_image1_path).convert("RGB")
                    transform = T_trans.Compose([
                        T_trans.Resize((224, 224)),
                        T_trans.ToTensor(),
                    ])

                    image0_tensor = transform(image0).to(policy.device)
                    image1_tensor = transform(image1).to(policy.device)

                    # load language embeddings
                    lang_token_embs = None
                    cur_task_id = 0 # cur_task_id is always zero as there is only one task
                    if self.has_lang_emb:
                        var_id = variation_number
                        lang_token_embs = self._lang_token_embs[f"{cur_task_id}_{var_id}"]
                    else:
                        lang_token_embs = self._lang_token_embs[f"{0}_{0}"]

                    # reset environment        
                    self.env.reset_to_demo(demo)
                    self.env.set_variation(variation_number)

                    # !! set variation_index and reset again (no idea why we need this...)
                    self.env._rlbench_env._scene._variation_index = variation_number
                    self.env.reset_to_demo(demo)

                    print("variation_number", variation_number)

                    # reset policy
                    subgoal_policy.reset()
                        
                    # reset parameters
                    success = False
                    term = False

                    # set episode length
                    episode_length =60
                    # cheers : 45 ~50
                    # put object in crate : 15~ 25

                    # initialize observation
                    obs = self.get_observation()
            

                    if self.use_fp:
                        
                        # 2. misc에서 시각적 객체 이름 가져오기
                        obj0_name = obs.misc['visual_obj0_name']
                        obj1_name = obs.misc['visual_obj1_name']

                        # 3. Estimator가 없으면 생성 (최초 1회 캐싱)
                        if not hasattr(self, '_estimator0') or self._estimator0 is None:
                            cprint(f"[FP] Creating estimators for: {obj0_name}, {obj1_name}", "yellow")
                            self.pose_estimation_wrapper.update_grasp_obj_name(obj0_name)
                            self._estimator0 = self.pose_estimation_wrapper.create_estimator(debug_level=0)
                            
                            self.pose_estimation_wrapper.update_grasp_obj_name(obj1_name)
                            self._estimator1 = self.pose_estimation_wrapper.create_estimator(debug_level=0)

                        # 4. FP를 이용해 현재 카메라(cam_front) 이미지에서의 포즈 추정
                        # (여기서 _get_pose_est는 내부적으로 현재 obs 이미지를 사용한다고 가정합니다)
                        fp_pose0, _ = self.env._get_pose_est(obj0_name, self._estimator0)
                        fp_pose1, _ = self.env._get_pose_est(obj1_name, self._estimator1)

                        if fp_pose0 is not None and fp_pose1 is not None:
                            # 카메라 기준의 추정된 포즈로 앵커 계산
                            anchor_pose = get_average_pose(fp_pose0, fp_pose1)
                            cprint("[FP] Anchor pose calculated from FoundationPose.", "blue")
                        else:
                            cprint("[FP] Failed to estimate poses. Falling back to GT pose.", "red")
                    
                    else:
                        obj0_name = obs.misc['visual_obj0_name']
                        obj1_name = obs.misc['visual_obj1_name']
                        visual_obj0 = Shape(obj0_name)
                        visual_obj1 = Shape(obj1_name)

                        gt_pose0 = visual_obj0.get_pose()
                        gt_pose1 = visual_obj1.get_pose()
                        anchor_pose = get_average_pose(gt_pose0, gt_pose1)
                        cprint(f"[GT] Using Ground Truth pose for {obj0_name} and {obj1_name}", "cyan")
                        


                    # start running
                    for step_i in range(episode_length):

                        obs_as_object = SimpleNamespace(
                            right=obs.right,
                            left=obs.left,
                            misc=obs.misc,
                            grasp_object0_image=image0_tensor,
                            grasp_object1_image=image1_tensor
                        )
                        
                        # get action
                        if step_i < episode_length:
                            print(f"{step_i} out of {episode_length}")
                            left_action, right_action = self.get_action(obs_as_object, lang_token_embs=lang_token_embs, anchor_pose=anchor_pose)

                        else:
                            left_action, right_action = subgoal_policy._open_gripper()
                        
                        # apply action
                        if left_action is None and right_action is None:
                            # action이 None인 경우는 보통 'reach' 단계에서 데모
                            obs = self.get_observation()
                        else:                   
                            # Update the observation based on the predicted action
                            action = self.toBimanualAction(left_action, right_action)
                            ts = self.env.step(action, record=True, verbose=True)
                            obs = self.get_observation()
                            term = ts.terminal

                        success, _ = self.env.env._scene.task.success()

                        # end condition
                        if success:
                            cprint(f"[RLbench Runner] Task success (in {step_i} steps).", "green")
                            RUN_FINISHED = True
                            break # break from execution loop
                        if term:
                            cprint(f"[RLbench Runner] Task fails. Error occurs.", "red")
                            RUN_FINISHED = True
                            break # break from execution loop
                        if episode_length > -1 and step_i >= episode_length-1:
                            cprint(f"Task fails. Exceed maximum episode length ({episode_length}).", "red")
                            RUN_FINISHED = True
                            break # break from execution loop
                except Exception as e:
                    print(e)
                if success:
                    break   # break from trial loop


            if RUN_FINISHED:
                # log result
                success_list.append(success)
                
                if self.use_fp:
                    save_path = os.path.join(self.output_dir, f"eval_use_fp=true_{self.task}")
                else:
                    save_path = os.path.join(self.output_dir, f"eval_use_fp=false_{self.task}")

                os.makedirs(save_path, exist_ok=True)


                # save result (per epoch)
                log_data_temp = {
                    "success": np.array(success_list),
                    "mean_success_rates": np.mean(success_list),
                }
                json_path = os.path.join(save_path, f'{tag}_temp.json')
                from utils.io_utils import save_np_dict_to_json
                save_np_dict_to_json(log_data_temp, json_path)

                # save logger data
                env_logger = self.env.get_env_logger()


                # save video
                video_path = os.path.join(save_path, f"{tag}_{self.task}_{demo_idx}.mp4")
                if self.use_fp:
                    env_logger.save_data(["vis_pose"], video_path, type="video")
                    # env_logger.save_data(["obs_record_rgb"], video_path, type="video")
                else:
                    env_logger.save_data(["obs_record_rgb"], video_path, type="video")

                # get poses
                poses_est = env_logger.get_data("pose_est")
                poses_gt = env_logger.get_data("pose_gt")

                import utils.transform_utils as T
                def geodesic(quat1, quat2):
                    mat1 = torch.from_numpy(T.quat2mat(quat1)).unsqueeze(0)
                    mat2 = torch.from_numpy(T.quat2mat(quat1)).unsqueeze(0)
                    loss_fn = GeodesicLoss()
                    loss = loss_fn(mat1, mat2)
                    return loss.clone().cpu().numpy()
                def euler_dist(quat1, quat2):
                    euler1 = euler_from_quaternion(*quat1)
                    euler2 = euler_from_quaternion(*quat2)
                    return euler2-euler1

                # save pose diff
                import matplotlib.pyplot as plt
                import seaborn as sns

                pose_img_path = os.path.join(save_path, f"{tag}_{self.task}_{demo_idx}_pose_diff.png")
                trans_diff = [np.mean(np.abs(p1[:3] - p2[:3])) for p1, p2 in zip(poses_est, poses_gt)]
                # rot_diff = [np.mean(np.abs(euler_dist(p1[3:], p2[3:]))) for p1, p2 in zip(poses_est, poses_gt)]
                rot_diff = [np.mean(np.abs(geodesic(p1[3:], p2[3:]))) for p1, p2 in zip(poses_est, poses_gt)]
                
                fig, (ax1, ax2) = plt.subplots(2, 1)
                sns.lineplot(x=range(len(trans_diff)), y=trans_diff, ax=ax1)
                sns.lineplot(x=range(len(rot_diff)), y=rot_diff, ax=ax2)
                ax1.set_ylabel("Translation Error")
                ax2.set_ylabel("Rotation Error")
                plt.tight_layout()
                plt.savefig(pose_img_path)
                # plt.show()

        # convert every item to np array
        log_data = {
            "success": np.array(success_list),
            "mean_success_rates": np.mean(success_list),
        }
        return log_data