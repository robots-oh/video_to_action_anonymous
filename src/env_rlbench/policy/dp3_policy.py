# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import sys

from networkx import degree
sys.path.insert(0, os.getcwd())

import copy
import torch
import numpy as np
import einops

from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply

from pyrep.objects import Dummy, VisionSensor, Shape
from env_rlbench.policy.subgoal_policy import RLBenchSubGoalPolicy
from utils.pose_utils import calculate_action, calculate_goal_pose, get_rel_pose, euler_from_quaternion, quaternion_from_euler, relative_to_target_to_world


class RLBenchDP3Policy(RLBenchSubGoalPolicy):
    def __init__(self, env, sub_goal_policy: BasePolicy, use_fp):
        self.env = env

        # initial stage
        self.stage = "reach" 

        # timer for stages
        self.T_GRASP = 10 
        self.GRASP_TIMER = 0 # closing gripper -> moving
        self.MOVEUP_TIMER = 0

        # dp3
        self.sub_goal_policy = sub_goal_policy
        self.device = sub_goal_policy.device
        self.dtype = sub_goal_policy.dtype
        self.cur_subgoal_lg = None
        self.cur_subgoal_rg = None
        self.cur_subgoal_obj = None
        self.cur_progress = None
        self.history = [] # for dp3
        self.n_obs_steps = sub_goal_policy.n_obs_steps
        self.use_fp = use_fp
        self.cur_loop_num = 0
        self.max_loop_num = 1

        self.predict_type = 'rel'
        self.history_dist_threshold = 0.001 #0.05 # !! Must be greater than moving distance otherwise subgoal would not be updated
        
        # [Modified] Assert removed to allow n_obs_steps > 1
        # assert self.n_obs_steps == 1
    
    def reset_in_loop(self):
        # initial stage
        self.stage = "reach" 

        # timer for stages
        self.T_GRASP = 10 
        self.GRASP_TIMER = 0 # closing gripper -> moving
        self.MOVEUP_TIMER = 0

        # dp3
        self.cur_subgoal = None
        self.cur_subgoal_obj = None
        self.cur_subgoal_lg = None  
        self.cur_subgoal_rg = None
        self.cur_progress = None
        self.history = [] # for dp3

        self.subgoal_step_count = 0
        self.MAX_SUBGOAL_STEPS = 5

    def reset(self):
        self.reset_in_loop()

        # visualization
        self.ee_pose_history = []
        self.obj_pose_history = []
        self.target_obj_pose_history = []
        self.subgoal_obj_pose_history = []  # pred
        self.subgoal_ee_pose_history = []  # pred
        self.buffer_pose_history = []   # history buffer

        # reset loop for multi-stage task
        self.cur_loop_num = 0
        self.max_loop_num = 1

    def _get_current_pose(self, obs): #이거 use_fp 하면 obs 필요없음

    
        if self.use_fp:
            grasp_obj0_pose = self.env.current_pose_estimation_obj0
            grasp_obj1_pose = self.env.current_pose_estimation_obj1
        else:
            grasp_obj0_pose = obs.misc["grasp_obj0_pose"]
            grasp_obj1_pose = obs.misc["grasp_obj1_pose"]

        
        
        DEBUG = False
        if self.use_fp and DEBUG:
            from scipy.spatial.transform import Rotation as R
            def get_euler(pose):
                rot = R.from_quat(pose[3:]) # (x, y, z, w)
                euler = rot.as_euler('xyz', degrees=True)
                return euler
            def get_rotvec(pose):
                rot = R.from_quat(pose[3:]) # (x, y, z, w)
                rotvec = rot.as_rotvec('xyz')
                return rotvec
            
            grasp_obj_pose = self.env.current_pose_estimation
            euler_fp = get_euler(grasp_obj_pose)
            grasp_obj_pose_gt = obs.misc["grasp_obj_pose"]
            euler_gt = get_euler(grasp_obj_pose_gt)
            print("fp", grasp_obj_pose, euler_fp)
            print("gt", grasp_obj_pose_gt, euler_gt)

            debug_dir = "./debug"

            # save mesh in simulation's global frame (converted from FP) - #*the mesh should align with fp_pcd
            world_mat = np.eye(4)
            world_mat[:3, 3] = grasp_obj_pose[:3]
            world_mat[:3, :3] = R.from_quat(grasp_obj_pose[3:]).as_matrix()
            m = self.env.pose_estimator.mesh.copy()
            m.apply_transform(world_mat)
            m.export(f'{debug_dir}/model_tf_world_fp.obj')

            # save mesh in simulation's global frame (ground truth) - #*the mesh should align with fp_pcd
            world_mat = np.eye(4)
            world_mat[:3, 3] = grasp_obj_pose_gt[:3]
            world_mat[:3, :3] = R.from_quat(grasp_obj_pose_gt[3:]).as_matrix()
            m = self.env.pose_estimator.mesh.copy()
            m.apply_transform(world_mat)
            m.export(f'{debug_dir}/model_tf_world_gt.obj')

            exit()
        return grasp_obj0_pose, grasp_obj1_pose

    def _get_sub_goal(self, obs, lang_token_embs=None, anchor_pose=None):
        # get the position and quaternion of object
        scene = self.env.env._scene
        np_obs_dict = dict()
        
   
        np_obs_dict['obj0_anchor_pos'] = np.array(self.history)[:, :7]
        np_obs_dict['obj1_anchor_pos'] = np.array(self.history)[:, 7:]
        if lang_token_embs is not None:
            np_obs_dict['lang_token_embs'] = lang_token_embs

        if hasattr(obs, 'grasp_object0_image') and hasattr(obs,'grasp_object1_image'):
            np_obs_dict['obj0_image'] = obs.grasp_object0_image
            np_obs_dict['obj1_image'] = obs.grasp_object1_image
        
        obs_dict = dict_apply(np_obs_dict,
                            lambda x: torch.from_numpy(x).to(
                                device=self.device) if isinstance(x, np.ndarray) else x)

        # [Modified] Use index -1 to reference the most recent observation in the history
        # When n_obs_steps=1, [0] and [-1] are the same. When >1, [-1] is correct for "current".
        with torch.no_grad():
            obs_dict_input = {
                'obs_anchor': {
                    'obj0_anchor_pos': obs_dict['obj0_anchor_pos'].unsqueeze(0).float(),
                    'obj1_anchor_pos': obs_dict['obj1_anchor_pos'].unsqueeze(0).float(),
                    'obj0_image': obs_dict['obj0_image'].to(self.device).unsqueeze(0),
                    'obj1_image': obs_dict['obj1_image'].to(self.device).unsqueeze(0),
                    'lang_token_embs': obs_dict['lang_token_embs'].to(self.device).unsqueeze(0).float(),
                }, 
            'obj0_to_obj1_pos': torch.from_numpy(get_rel_pose(np_obs_dict['obj1_anchor_pos'][-1], np_obs_dict['obj0_anchor_pos'][-1])).unsqueeze(0).unsqueeze(0).float().to(self.device),
            'obj1_to_obj0_pos': torch.from_numpy(get_rel_pose(np_obs_dict['obj0_anchor_pos'][-1], np_obs_dict['obj1_anchor_pos'][-1])).unsqueeze(0).unsqueeze(0).float().to(self.device)
            }

            action_dict = self.sub_goal_policy.predict_action(obs_dict_input)

        np_action_dict = dict_apply(action_dict,
                                    lambda x: x.detach().to('cpu').numpy())
        pred = np_action_dict['action_combined'].squeeze(0)
        select_pred = pred[0]

        select_action = select_pred[:14]
        select_obj0_action = select_action[:7]  #
        select_obj1_action = select_action[7:] 
        select_progress = select_pred[14]

        
        pose_combined = np.array(self.history[-1])
        anchor_obj0_pose = pose_combined[:7]
        anchor_obj1_pose = pose_combined[7:]


        # 예측한 action 변화량을 anchor에서 object가 다음 state가 어떻게 될지 계산
        obj0_subgoal = calculate_goal_pose(anchor_obj0_pose, select_obj0_action) # AO
        obj1_subgoal = calculate_goal_pose(anchor_obj1_pose, select_obj1_action) # AO


        ##### subgoal 은 left pose , right pose 합친거
        subgoal_actions = select_action # 물체 pos, ori 변화량

        # object의 다음 state 에 맞는 gripper의 state 예측
        # gripper가 anchor 기준으로 움직이게 됨
        subgoal_left_gripper, subgoal_right_gripper = self._subgoal_actions_to_subgoal_gripper(obj0_subgoal, obj1_subgoal, anchor_pose)
        # anchor_pose : World 기준 Anchor
        
        return subgoal_left_gripper, subgoal_right_gripper, subgoal_actions, select_progress  # WGl, WGr , delta AO, progress

    def _update_history_buffer(self, obs, anchor_pose): # anchor 기준으로 object pose 나와.
        # get the position and quaternion of object
        # get current pose
        grasp_obj0_pose, grasp_obj1_pose = self._get_current_pose(obs)
        anchor_obj0_pose = get_rel_pose(anchor_pose, grasp_obj0_pose)
        anchor_obj1_pose = get_rel_pose(anchor_pose, grasp_obj1_pose)
        obj_anchorpose_combined = np.concatenate([anchor_obj0_pose, anchor_obj1_pose])
        #obj0_pose_relative_to_obj1 = get_rel_pose(grasp_obj1_pose, grasp_obj0_pose)

        # # convert from quaternion to euler
        # obj_pose_relative_to_target_euler = np.zeros(6)
        # obj_pose_relative_to_target_euler[:3] = obj_pose_relative_to_target[:3]
        # obj_pose_relative_to_target_euler[3:] = euler_from_quaternion(*obj_pose_relative_to_target[3:])

        assert self.n_obs_steps > 0
        if len(self.history) < self.n_obs_steps:
            # [Modified] Fill buffer up to n_obs_steps
            self.history.extend([obj_anchorpose_combined for _ in range(self.n_obs_steps-len(self.history))])
        else:
            prev_pose_combined = self.history[-1]
            dist0 = np.linalg.norm(obj_anchorpose_combined[:3] - prev_pose_combined[:3])
            dist1 = np.linalg.norm(obj_anchorpose_combined[7:10] - prev_pose_combined[7:10])
            dist = max(dist0, dist1)

            if dist >= self.history_dist_threshold:
                # [Modified] Generalized history update (FIFO)
                self.history.append(obj_anchorpose_combined)
                if len(self.history) > self.n_obs_steps:
                    self.history.pop(0)

    def get_action(self, obs, lang_token_embs=None, anchor_pose=None):
        assert self.stage in ["reach", "move", "leave", "end"]
        if self.env._start_tracking:
            grasp_obj0_pose, grasp_obj1_pose = self._get_current_pose(obs)
            left_gripper_pose = self.env._task._robot.left_arm.get_tip().get_pose()
            right_gripper_pose = self.env._task._robot.right_arm.get_tip().get_pose()
            self.T_obj0_to_left_gripper = get_rel_pose(grasp_obj0_pose, left_gripper_pose) # O-> G
            self.T_obj1_to_right_gripper = get_rel_pose(grasp_obj1_pose, right_gripper_pose)
        task_name = self.env.env._scene.task.get_name()
        
        # get action
        if self.stage == "reach":
            self.env.run_demo_until_n_waypoint(task_name)
            self.env.start_tracking()
            self.env._rlbench_env._scene.step()
            self.stage = "move"
            return None, None
        
        
        elif self.stage == "move":
            print(f"progress: {self.cur_progress}")
            # if (self.cur_progress is not None) and (self.cur_progress > 0.9) and (task_name not in ["pour_water_to_cup", "wipe_dish_with_sponge"]):
            if (self.cur_progress is not None) and (self.cur_progress > 0.9) and (task_name not in ["pour_water_to_cup", "pour_water_to_cup_mirror", "wipe_dish_with_sponge", "wipe_dish_with_sponge_mirror", "cheers_coke_easy","cheers_coke_easy_mirror", "vase_and_flower", "vase_and_flower_mirror"]):
                # move to next stage
                self.stage = "leave"
                return None, None
                
            subgoal_threshold = 0.05 # distance between current EE pose and subgoal pose 
            if (self.cur_subgoal_lg is not None) and (self.cur_subgoal_rg is not None):   # 왼팔 오른팔 둘다 subgoal이 있을때
                left_gripper_pose = self.env._task._robot.left_arm.get_tip().get_pose()
                right_gripper_pose = self.env._task._robot.right_arm.get_tip().get_pose()
                lg_dist = np.linalg.norm(self.cur_subgoal_lg[:3] - left_gripper_pose[:3])
                lg_angle_diff = np.linalg.norm(self.cur_subgoal_lg[3:] - left_gripper_pose[3:])
                lg_angle_diff2 = np.linalg.norm(-self.cur_subgoal_lg[3:] - left_gripper_pose[3:])
                gripper_lg_goal_far = lg_dist > subgoal_threshold or min(lg_angle_diff, lg_angle_diff2) > 0.001

                rg_dist = np.linalg.norm(self.cur_subgoal_rg[:3] - right_gripper_pose[:3])
                rg_angle_diff = np.linalg.norm(self.cur_subgoal_rg[3:] - right_gripper_pose[3:])
                rg_angle_diff2 = np.linalg.norm(-self.cur_subgoal_rg[3:] - right_gripper_pose[3:])
                gripper_rg_goal_far = rg_dist > subgoal_threshold or min(rg_angle_diff, rg_angle_diff2) > 0.001 # BUG: This was gripper_rg_goal_far = ...


            is_timeout = self.subgoal_step_count >= self.MAX_SUBGOAL_STEPS

            # get first subgoal
            if (self.cur_subgoal_lg is None) and (self.cur_subgoal_rg is None):
                
                self._update_history_buffer(obs, anchor_pose)
                # self._update_history_buffer_gt()
                self.cur_subgoal_lg, self.cur_subgoal_rg, self.cur_subgoal_obj, self.cur_progress = self._get_sub_goal(obs, lang_token_embs, anchor_pose)
            # get new/next subgoal
            elif not gripper_lg_goal_far or not gripper_rg_goal_far or is_timeout:
                if is_timeout:
                    print(f"Warning: Subgoal timeout! Applying random jitter to escape deadlock.")
                
                    # 1. 1~2cm(0.01~0.02m) 수준의 랜덤 노이즈 생성
                    jitter_pos_lg = np.random.uniform(-0.06, 0.06, size=3)
                    jitter_pos_rg = np.random.uniform(-0.06, 0.06, size=3)
                    
                    # (선택사항) orientation에도 약간의 노이즈를 주고 싶다면 추가 가능하지만, 
                    # 보통 position만 살짝 흔들어줘도 충분히 빠져나옵니다.

                    # 2. 현재 그리퍼 위치에 노이즈를 더해 임시 목표(Jitter Goal) 생성
                    temp_goal_lg = left_gripper_pose.copy()
                    temp_goal_lg[:3] += jitter_pos_lg
                    
                    temp_goal_rg = right_gripper_pose.copy()
                    temp_goal_rg[:3] += jitter_pos_rg

                    # 3. 상태 초기화 (다음 스텝에서 _get_sub_goal이 호출되도록 함)
                    self.cur_subgoal_lg = None
                    self.cur_subgoal_rg = None
                    self.subgoal_step_count = 0
                    
                    # 4. 살짝 흔드는 액션 반환
                    return self._move_to(temp_goal_lg, temp_goal_rg, close_gripper=True, object_centric=True)

                self._update_history_buffer(obs, anchor_pose)
                # self._update_history_buffer_gt()
                self.cur_subgoal_lg, self.cur_subgoal_rg, self.cur_subgoal_obj, self.cur_progress = self._get_sub_goal(obs, lang_token_embs, anchor_pose)
                self.subgoal_step_count = 0
            # use previous subgoal
            else:
                print(f"Far | left: {gripper_lg_goal_far} | right: {gripper_rg_goal_far}")
                self.subgoal_step_count += 1

            lg_goal_pose = self.cur_subgoal_lg # anchor 기준으로 움직여야 하는 gripper goal pose
            rg_goal_pose = self.cur_subgoal_rg # anchor 기준으로 움직여야 하는 gripper goal pose
            #goal_obj_pose = self.cur_subgoal_obj
            return self._move_to(lg_goal_pose, rg_goal_pose, close_gripper=True, object_centric=True)

        elif self.stage == "leave":
            self.env.end_tracking()
            if self.GRASP_TIMER < self.T_GRASP:
                self.GRASP_TIMER += 1
                return self._open_gripper()
            else:
                self.GRASP_TIMER = 0 # reset timer
                self.stage = "end"
                return self._move_away(axis='z', dist=0.1)
            
        elif self.stage == "end":
           
            return None, None


        else:
            raise NotImplementedError(f"Unknown stage: {self.stage}")