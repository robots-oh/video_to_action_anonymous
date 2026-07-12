# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from utils.pose_utils import get_rel_pose, relative_to_target_to_world


class RLBenchSubGoalPolicy:
    def __init__(self, env):
        pass
    
    def _open_gripper(self):
               
        gripper_left = self.env._rlbench_env._scene.robot.left_gripper
        gripper_right = self.env._rlbench_env._scene.robot.right_gripper
        # 나중에 task 바뀌면 right도 건드려야할수도 있음
        scene = self.env._rlbench_env._scene
        gripper_left.release() # remove object from the list gripper._grasped_objects # origin의 경우 left
        # gripper_right.release() # mirror인 경우 right
    
        done = False
        i = 0
        vel = 0.04
        open_amount = 1.0 #if gripper_name == 'Robotiq85Gripper' else 0.8
        while not done:
            done = gripper_left.actuate(open_amount, velocity=vel) # origin task
            # done = gripper_right.actuate(open_amount, velocity=vel) # mirror task
            scene.step()
            i += 1
            if i > 1000:
                self.fail('Took too many steps to open')
        return None, None
    
    def _move_away(self, axis='z', dist=0.2):
        gripper_left_pose = self.env._task._robot.left_arm.get_tip().get_pose()
        gripper_right_pose = self.env._task._robot.right_arm.get_tip().get_pose()

        action_left = np.zeros(9)
        action_left[7] = 1 # open gripper
        action_left[8] = 0 # ignore collision
        action_left[:7] = gripper_left_pose
        if 'x' in axis:
            action_left[0] += dist[0] if isinstance(dist, list) else dist
        if 'y' in axis:
            action_left[1] += dist[1] if isinstance(dist, list) else dist
        if 'z' in axis:
            action_left[2] += dist[2] if isinstance(dist, list) else dist

        action_right = np.zeros(9)
        action_right[7] = 1 # open gripper
        action_right[8] = 0 # ignore collision
        action_right[:7] = gripper_right_pose
        if 'x' in axis:
            action_right[0] += dist[0] if isinstance(dist, list) else dist
        if 'y' in axis:
            action_right[1] += dist[1] if isinstance(dist, list) else dist
        if 'z' in axis:
            action_right[2] += dist[2] if isinstance(dist, list) else dist
        
        return action_left, action_right
    
    def _move_to(self, lg_goal_pose, rg_goal_pose, close_gripper=True, object_centric=False):
        action_left = np.zeros(9)
        action_left[7] = 0 # close gripper
        action_left[8] = 0 # ignore collision
        action_left[:7] = lg_goal_pose # assume the action mode is EndEffectorPoseViaIK or EndEffectorPoseViaPlanning

        action_right = np.zeros(9)
        action_right[7] = 0 # close gripper
        action_right[8] = 0 # ignore collision
        action_right[:7] = rg_goal_pose # assume the action mode
        return action_left, action_right


    def _subgoal_actions_to_subgoal_gripper(self, obj0_subgoal, obj1_subgoal, anchor_pose):

        # Calculate gripper's goal pose relative to the anchor
        subgoal_left_gripper_in_anchor = relative_to_target_to_world(self.T_obj0_to_left_gripper, obj0_subgoal) # AO * OG -> AG
        subgoal_right_gripper_in_anchor = relative_to_target_to_world(self.T_obj1_to_right_gripper, obj1_subgoal) 

        # Convert the anchor-relative gripper pose to the world coordinate system 
        subgoal_left_gripper = relative_to_target_to_world(subgoal_left_gripper_in_anchor, anchor_pose) # WA * AG -> WG
        subgoal_right_gripper = relative_to_target_to_world(subgoal_right_gripper_in_anchor, anchor_pose)

        return subgoal_left_gripper, subgoal_right_gripper