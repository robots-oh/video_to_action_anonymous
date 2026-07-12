
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

import re
import logging
from typing import List, Type
import copy

import numpy as np
from pyrep.const import RenderMode
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.objects import Dummy, VisionSensor, Shape
from rlbench import ActionMode, ObservationConfig
from rlbench.backend.exceptions import InvalidActionError, BoundaryError, WaypointError
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task
from yarr.agents.agent import VideoSummary
from yarr.envs.rlbench_env import RLBenchEnv
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition
from foundation_pose import Utils

# from env_rlbench_peract.utils.rlbench_utils import MyEnvironmentPeract, get_mask, get_rgb_depth, get_seg_mask
# from env_rlbench.runner.rl_bench_camera import get_camera
# from env_rlbench_peract.utils.rlbench_objects import task_object_dict


from env_rlbench_peract.utils.rlbench_utils_vtob import MyEnvironmentPeract, get_mask, get_rgb_depth, get_seg_mask
from env_rlbench.runner.rl_bench_camera import get_camera
from env_rlbench_peract.utils.rlbench_objects_vtob import task_object_dict


import utils.transform_utils as T
from utils.logger_utils import EnvLogger
from utils.vis_utils import get_vis_pose

RECORD_EVERY = 20


class CustomRLBenchEnv(RLBenchEnv):
    def __init__(
        self,
        task_class: Type[Task],
        observation_config: ObservationConfig,
        action_mode: ActionMode,
        episode_length: int,
        dataset_root: str = "",
        channels_last: bool = False,
        reward_scale=100.0,
        headless: bool = True,
        time_in_state: bool = False,
        use_fp: bool = False,
        fp_cam_name: str = None,
        pose_estimation_wrapper=None,
    ):
        super(CustomRLBenchEnv, self).__init__(
            task_class,
            observation_config,
            action_mode,
            dataset_root,
            channels_last,
            headless=headless,
        )
        self._observation_config.robot_name = (
            "bimanual"  # YARR's _extract_obs_bimanual expects this
        )
        self._reward_scale = reward_scale
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._episode_length = episode_length
        self._time_in_state = time_in_state
        self._i = 0

        self._last_mile = False
        self._start_tracking = False

        # support object state observation
        self._rlbench_env = MyEnvironmentPeract(
            action_mode=action_mode,
            obs_config=observation_config,
            dataset_root=dataset_root,
            headless=headless,
        )
        task_name = re.sub(
            '(?<!^)(?=[A-Z])', '_', task_class.__name__).lower()
        self.grasp_obj_name = None
        self.target_obj_name = None

        # ------------------------
        # |    Pose Estimation   |
        # ------------------------
        self.use_fp = use_fp
        self.fp_cam_name = fp_cam_name
        self.pose_estimation_wrapper = pose_estimation_wrapper
        self.pose_estimator_obj0 = None
        self.pose_estimator_obj1 = None
        self.current_pose_estimation_obj0 = None
        self.current_pose_estimation_obj1 = None
        
        # 데이터 수집 시 FP 사용 여부 및 카메라 설정
        self._collect_use_fp = False
        self._fp_cam_for_collect = None

        # ------------------------
        # |        Logging       |
        # ------------------------
        self.env_logger = EnvLogger()
        self.env_logger.add_data_type("obs_record_rgb", "rgb")
        self.env_logger.add_data_type("obs_wrist_rgb", "rgb")

        self.env_logger.add_data_type("obs_fp_rgb", "rgb")
        self.env_logger.add_data_type("obs_fp_depth", "depth")
        self.env_logger.add_data_type("obs_fp_mask", "mask")

        self.env_logger.add_data_type("vis_pose", "rgb")        
        self.env_logger.add_data_type("pose_est", "list")
        self.env_logger.add_data_type("pose_gt", "list")


        self.debug = False
        self._count = 0 # debug only

#####################################################################################
    # @property
    # def observation_elements(self) -> List[ObservationElement]:
    #     obs_elems = super(CustomRLBenchEnv, self).observation_elements
    #     for oe in obs_elems:
    #         if oe.name == "low_dim_state":
    #             oe.shape = (
    #                 oe.shape[0] - 7 * 2 + int(self._time_in_state),
    #             )  # remove pose and joint velocities as they will not be included
    #             self.low_dim_state_len = oe.shape[0]
    #     obs_elems.append(ObservationElement("gripper_pose", (7,), np.float32))
    #     return obs_elems

    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super(CustomRLBenchEnv, self).observation_elements
        for oe in obs_elems:
            if oe.name == "right_low_dim_state":
                oe.shape = (
                    oe.shape[0] - 7 * 2 + int(self._time_in_state),
                )  # remove pose and joint velocities as they will not be included
                self.low_dim_state_len = oe.shape[0]
        obs_elems.append(ObservationElement("gripper_pose", (7,), np.float32))
        return obs_elems


    def extract_obs(self, obs: Observation, t=None, prev_action=None):
        # For bimanual, yarr's _extract_obs_bimanual is called in super().extract_obs
        # It requires obs_config.robot_name to be set.
        # We set it to 'bimanual' in __init__.

        # We don't want joint velocities in the observation
        if obs.is_bimanual:
            obs.right.joint_velocities = None
            obs.left.joint_velocities = None
        else:
            # 단일 팔 로봇의 경우
            obs.joint_velocities = None

        # Store original values to restore them later
        if obs.is_bimanual:
            # 양팔 로봇의 경우, 각 팔에서 값을 가져옵니다.
            # YARR의 _extract_obs_bimanual은 gripper_matrix 등을 사용하지 않으므로,
            # 여기서는 None으로 설정하고 나중에 복원할 필요가 없습니다.
            obs.right.gripper_matrix = None
            obs.left.gripper_matrix = None
            obs.right.wrist_camera_matrix = None
            obs.left.wrist_camera_matrix = None
            obs.right.joint_positions = None
            obs.left.joint_positions = None
        else:
            # 단일 팔 로봇의 경우 기존 로직을 유지합니다.
            grip_mat = obs.gripper_matrix
            grip_pose = obs.gripper_pose
            joint_pos = obs.joint_positions
            wrist_cam_mat = obs.wrist_camera_matrix if hasattr(obs, 'wrist_camera_matrix') else None
            obs.gripper_matrix = None
            obs.wrist_camera_matrix = None
            obs.joint_positions = None

        # This will call _extract_obs_bimanual if obs is BimanualObservation
        obs_dict = super(CustomRLBenchEnv, self).extract_obs(obs)

        # Add gripper_pose to obs_dict, as it's used by the policy
        if not obs.is_bimanual:
             # Restore original values for unimanual
             obs.gripper_matrix = grip_mat
             obs.joint_positions = joint_pos
             if wrist_cam_mat is not None:
                 obs.wrist_camera_matrix = wrist_cam_mat
             obs_dict["gripper_pose"] = grip_pose

        if self._time_in_state:
            time = (1.0 - ((self._i if t is None else t) / float(self._episode_length - 1))) * 2.0 - 1.0
            # For bimanual, low_dim_state is split into right and left
            if 'right_low_dim_state' in obs_dict:
                obs_dict['right_low_dim_state'] = np.concatenate([obs_dict['right_low_dim_state'], [time]]).astype(np.float32)
                obs_dict['left_low_dim_state'] = np.concatenate([obs_dict['left_low_dim_state'], [time]]).astype(np.float32)
            else: # unimanual
                obs_dict["low_dim_state"] = np.concatenate([obs_dict["low_dim_state"], [time]]).astype(np.float32)

        return obs_dict

    def launch(self):
        super(CustomRLBenchEnv, self).launch()
        self._task._scene.register_step_callback(self._my_callback)

        self._record_cam = VisionSensor.create([512, 288])
        self._record_cam.set_explicit_handling(True)
        pose = np.array(VisionSensor('cam_front').get_pose())
        self._record_cam.set_pose(list(pose))
        self._record_cam.set_render_mode(RenderMode.OPENGL)

        self._wrist_cam_right = VisionSensor('cam_wrist_right')
        self._wrist_cam_right.set_resolution([512, 288])
        self._wrist_cam_right.set_explicit_handling(True)
        self._wrist_cam_right.set_render_mode(RenderMode.OPENGL)

        self._wrist_cam_left = VisionSensor('cam_wrist_left')
        self._wrist_cam_left.set_resolution([288, 288])
        self._wrist_cam_left.set_explicit_handling(True)
        self._wrist_cam_left.set_render_mode(RenderMode.OPENGL)

        self._overhead_cam = VisionSensor('cam_overhead')
        self._overhead_cam.set_resolution([288, 288])
        self._overhead_cam.set_explicit_handling(True)
        self._overhead_cam.set_render_mode(RenderMode.OPENGL)

        # get fp camera
        if self.use_fp and self.fp_cam_name is not None:
            if self.fp_cam_name == "cam_front":
                self._fp_cam = VisionSensor('cam_front') 
                self._fp_cam.set_render_mode(RenderMode.OPENGL)
                self._fp_cam.set_explicit_handling(True)

                self._fp_cam_mask = VisionSensor('cam_front_mask')
                self._fp_cam_mask.set_render_mode(RenderMode.OPENGL_COLOR_CODED)
                self._fp_cam_mask.set_explicit_handling(True)

                rgb_pose = self._fp_cam.get_pose()
                self._fp_cam_mask.set_pose(rgb_pose)
            else:
                raise Exception("fp_cam_name Error")


    def reset(self) -> dict:
        self._previous_obs_dict = super(CustomRLBenchEnv, self).reset()

        self._i = 0
        self._cur_stage = 0

        self._last_mile = False
        self._start_tracking = False

        self.env_logger.clear()
        
        # get pose estimator
        if self.pose_estimation_wrapper is not None:
            self.end_tracking() 
        
        return self._previous_obs_dict
    
    def reset_to_demo(self, demo):
        self._task.reset_to_demo(demo)  # TaskEnvironment().reset_to_demo()

        self._i = 0
        self._cur_stage = 0

        self._last_mile = False
        self._start_tracking = False
        
        self.env_logger.clear()
        
        # get pose estimator
        if self.pose_estimation_wrapper is not None:
            self.end_tracking() # pose_estimator들을 None으로 초기화

        # MyScene에 캐시된 FoundationPose 관련 변수들을 명시적으로 초기화합니다.
        # 이렇게 하면 새 데모를 시작할 때마다 estimator와 앵커 포즈가 새로 계산됩니다.
        scene = self._task._scene
        if hasattr(scene, 'pose_estimation_wrapper') and scene.pose_estimation_wrapper is not None:
            scene._estimator0 = None
            scene._estimator1 = None
            scene._fp_grasp_obj0_pose = None
            scene._fp_grasp_obj1_pose = None
            scene._camera_anchor_pose = None
    
    def set_variation(self, variation_number):
        self._task.set_variation(variation_number)
        task_name = self._task.get_name()



        self.grasp_obj0_name = task_object_dict[task_name]["grasp_object0_name"] if isinstance(task_object_dict[task_name]["grasp_object0_name"], str) else task_object_dict[task_name]["grasp_object0_name"][variation_number]
        self.grasp_obj1_name = task_object_dict[task_name]["grasp_object1_name"] if isinstance(task_object_dict[task_name]["grasp_object1_name"], str) else task_object_dict[task_name]["grasp_object1_name"][variation_number]

        self.visual_obj0_name = task_object_dict[task_name]["visual_object0_name"] if isinstance(task_object_dict[task_name]["visual_object0_name"], str) else task_object_dict[task_name]["visual_object0_name"][variation_number]
        self.visual_obj1_name = task_object_dict[task_name]["visual_object1_name"] if isinstance(task_object_dict[task_name]["visual_object1_name"], str) else task_object_dict[task_name]["visual_object1_name"][variation_number]


        # 데이터 수집을 위한 FoundationPose Estimator 설정
        if self._collect_use_fp and self.pose_estimation_wrapper is not None:
            self.end_tracking() # 이전 estimator 정리
            # obj0에 대한 estimator 생성
            self.pose_estimation_wrapper.update_grasp_obj_name(self.visual_obj0_name)
            self.pose_estimator_obj0 = self.pose_estimation_wrapper.create_estimator(debug_level=0)
            # obj1에 대한 estimator 생성
            self.pose_estimation_wrapper.update_grasp_obj_name(self.visual_obj1_name)
            self.pose_estimator_obj1 = self.pose_estimation_wrapper.create_estimator(debug_level=0)
            # 사용할 카메라 설정
            self._fp_cam_for_collect = VisionSensor(self.fp_cam_name)

    def register_callback(self, func):
        self._task._scene.register_step_callback(func)

    def _get_pose_est(self, obj_name: str, pose_estimator):
        def fp_frame_to_world_frame(fp_mat, R):
            # flip x- and y-axis
            # tf_flip_x_y = np.eye(4)
            # tf_flip_x_y[0, 0] = -1   # flip x-axis
            # tf_flip_x_y[1, 1] = -1   # flip y-axis

            from scipy.spatial.transform import Rotation
            r = Rotation.from_euler('zyx', [np.pi, 0, 0])
            tf_flip_x_y = np.eye(4)
            tf_flip_x_y[:3, :3] = r.as_matrix()

            # to world frame
            tf_camera_to_world = R

            fp_mat_flipped = np.matmul(tf_flip_x_y, fp_mat)              # flip x- and y-axis
            world_mat = np.matmul(tf_camera_to_world, fp_mat_flipped)    # to world frame
            pose = np.concatenate([world_mat[:3, 3], T.mat2quat(world_mat[:3, :3])])
            return pose, world_mat
        
        def world_frame_to_fp_frame(world_pose, R):
            # flip x- and y-axis
            # tf_flip_x_y = np.eye(4)
            # tf_flip_x_y[0, 0] = -1   # flip x-axis
            # tf_flip_x_y[1, 1] = -1   # flip y-axis

            from scipy.spatial.transform import Rotation
            r = Rotation.from_euler('zyx', [np.pi, 0, 0])
            tf_flip_x_y = np.eye(4)
            tf_flip_x_y[:3, :3] = r.as_matrix()

            # to world frame
            tf_camera_to_world = R

            world_mat = np.eye(4)
            world_mat[:3, 3] = world_pose[:3]
            world_mat[:3, :3] = T.quat2mat(world_pose[3:])
            fp_mat_flipped = np.matmul(np.linalg.inv(tf_camera_to_world), world_mat)    # to world frame
            fp_mat = np.matmul(np.linalg.inv(tf_flip_x_y), fp_mat_flipped)
            return fp_mat
        

        # 사용할 카메라 결정 (eval 중이면 _fp_cam, collect 중이면 _fp_cam_for_collect)
        cam_to_use = self._fp_cam if self._fp_cam is not None else self._fp_cam_for_collect
        cam_mask_to_use = self._fp_cam_mask if self._fp_cam_mask is not None else VisionSensor(self.fp_cam_name + '_mask')

        fp_rgb, fp_depth, fp_pcd = get_rgb_depth(
            sensor=cam_to_use,
            get_rgb=True, get_depth=True, get_pcd=True, 
            rgb_noise=None, depth_noise=None, 
            depth_in_meters=True,
        )

        # get mask
        mask = get_mask(sensor=cam_mask_to_use, masks_as_one_channel=True)
        fp_mask = get_seg_mask([Shape(obj_name)], mask)
            
        # get new pose estimation
        K = cam_to_use.get_intrinsic_matrix()
        R = cam_to_use.get_matrix()
        K[:2, :2] *= -1 # !! convert into positive
        color = fp_rgb
        depth = fp_depth
        select_seg_id = 1 # !! 0 is background
        ob_mask = (fp_mask == select_seg_id).astype(bool).astype(np.uint8)

        if pose_estimator.pose_last is None or self._i % 5 == 0:
            fp_mat = pose_estimator.register(K=K, rgb=color, depth=depth, ob_mask=ob_mask, iteration=5)
            self._count += 1
        else:
            if pose_estimator.pose_last is None:
                fp_mat = pose_estimator.register(K=K, rgb=color, depth=depth, ob_mask=ob_mask, iteration=5)
            else:
                fp_mat = pose_estimator.track_one(rgb=color, depth=depth, K=K, iteration=2)
            self._count += 1

        # convert pose estimation to world frame
        pose_est, world_mat = fp_frame_to_world_frame(fp_mat, R)
        
        if self.debug:
            from scipy.spatial.transform import Rotation
            def get_euler(pose):
                rot = Rotation.from_quat(pose[3:]) # (x, y, z, w)
                euler = rot.as_euler('xyz', degrees=True)
                return euler
            def get_rotvec(pose):
                rot = Rotation.from_quat(pose[3:]) # (x, y, z, w)
                rotvec = rot.as_rotvec('xyz')
                return rotvec
            
            from foundation_pose.Utils import depth2xyzmap, toOpen3dCloud
            import open3d as o3d

            # save intrinsics
            debug_dir = "./debug"
            np.savetxt(os.path.join(debug_dir, "K.txt"), K)

            # save rgb
            import cv2
            cv2.imwrite(os.path.join(debug_dir, "fp_rgb.png"), cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
            
            # save depth
            cv2.imwrite(os.path.join(debug_dir, "fp_depth.png"), (fp_depth * 1000).astype(np.uint16))
            print(np.max(depth), np.min(depth))
            # load depth
            depth_map = cv2.imread(os.path.join(debug_dir, "fp_depth.png"), cv2.IMREAD_ANYDEPTH) / 1000.
            print(np.max(depth_map), np.min(depth_map))

            # save mask
            cv2.imwrite(os.path.join(debug_dir, "fp_mask.png"), (ob_mask*255.0).clip(0,255))

            # save pointcloud in simulation's global frame (ground truth)
            pcd = toOpen3dCloud(fp_pcd.reshape(-1, 3), fp_rgb.reshape(-1, 3))
            o3d.io.write_point_cloud(f'{debug_dir}/fp_pcd.ply', pcd)    # global frame

            # save mesh in simulation's global frame (converted from FP) - #*the mesh should align with fp_pcd
            m = self.pose_estimator.mesh.copy()
            m.apply_transform(world_mat)
            m.export(f'{debug_dir}/model_tf_world_fp.obj')
        
            # save pointcloud in camera frame (directly from FP)
            xyz_map = depth2xyzmap(depth, K)
            valid = depth>=0.001
            pcd = toOpen3dCloud(xyz_map[valid], color[valid])
            o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)        # camera frame

            # save mesh in camera frame (directly from FP) - #*the mesh should align with xyz_map
            m = pose_estimator.mesh.copy()
            m.apply_transform(fp_mat)
            m.export(f'{debug_dir}/model_tf.obj')

            # check determinant (1 if in right-hand coordinate)
            print("fp_mat det", np.linalg.det(fp_mat[:3, :3]))
            print("world_mat det", np.linalg.det(world_mat[:3, :3]))

            pose_gt = Shape(obj_name).get_pose()
            print("pose_est", pose_est)
            print("pose_gt", pose_gt)
            print("pose_est_euler", get_euler(pose_est))
            print("pose_gt_euler", get_euler(pose_gt))

            # save mesh in simulation's global frame (converted from FP) - #*the mesh should align with fp_pcd
            world_mat = np.eye(4)
            world_mat[:3, 3] = pose_est[:3]
            world_mat[:3, :3] = Rotation.from_quat(pose_est[3:]).as_matrix() # type: ignore
            m = pose_estimator.mesh.copy()
            m.apply_transform(world_mat)
            m.export(f'{debug_dir}/model_tf_world_fp.obj')

            # save mesh in simulation's global frame (ground truth) - #*the mesh should align with fp_pcd
            world_mat = np.eye(4)
            world_mat[:3, 3] = pose_gt[:3]
            world_mat[:3, :3] = Rotation.from_quat(pose_gt[3:]).as_matrix() # type: ignore
            m = pose_estimator.mesh.copy()
            m.apply_transform(world_mat)
            m.export(f'{debug_dir}/model_tf_world_gt.obj')

            exit()

        # logging
        self.env_logger.add_data("obs_fp_rgb", fp_rgb)
        self.env_logger.add_data("obs_fp_depth", fp_depth)
        self.env_logger.add_data("obs_fp_mask", fp_mask)

        vis_pose = get_vis_pose(
            pose=fp_mat, 
            color=color, 
            K=K, 
            mesh=pose_estimator.mesh
        )

        pose_gt = Shape(obj_name).get_pose()
        self.env_logger.add_data("pose_est", pose_est)
        self.env_logger.add_data("pose_gt", pose_gt)
        
        return pose_est, vis_pose

  

    def _my_callback(self):

        # last mile
        grasp_obj0_pose = Shape(self.visual_obj0_name).get_pose()
        grasp_obj1_pose = Shape(self.visual_obj1_name).get_pose()
        if np.linalg.norm(grasp_obj0_pose[:3] - grasp_obj1_pose[:3]) < 0.15:
            self._last_mile = True

        # fp pose
        if self.use_fp:
            if self._start_tracking and (self.current_pose_estimation_obj0 is None or self._i % 1 == 0):
                self.current_pose_estimation_obj0, vis_pose0 = self._get_pose_est(self.visual_obj0_name, self.pose_estimator_obj0)

                self.current_pose_estimation_obj1, vis_pose1 = self._get_pose_est(self.visual_obj1_name, self.pose_estimator_obj1)
                
                
                # 두 시각화 이미지를 가로로 합쳐서 저장
                vis_pose_combined = np.concatenate([vis_pose0, vis_pose1], axis=1)
                self.env_logger.add_data("vis_pose", vis_pose_combined)
            else:
                self._fp_cam.handle_explicitly()
                cap_fp = (self._fp_cam.capture_rgb() * 255).astype(np.uint8)
                # 두 객체에 대한 시각화가 없으므로, 동일한 이미지를 두 번 합쳐서 형태를 맞춥니다.
                vis_pose_combined = np.concatenate([cap_fp, cap_fp], axis=1)
                self.env_logger.add_data("vis_pose", vis_pose_combined)


        # record cam (vis only)
        self._record_cam.handle_explicitly()
        self._overhead_cam.handle_explicitly()
        cap_record = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        cap_overhead = (self._overhead_cam.capture_rgb() * 255).astype(np.uint8)
        self.env_logger.add_data("obs_record_rgb", np.concatenate([cap_record, cap_overhead], axis=1))

        # wrist cam (vis only) for bimanual robot
        self._wrist_cam_right.handle_explicitly()
        self._wrist_cam_left.handle_explicitly()
        cap_wrist_right = (self._wrist_cam_right.capture_rgb() * 255).astype(np.uint8)
        cap_wrist_left = (self._wrist_cam_left.capture_rgb() * 255).astype(np.uint8)
        
        # 두 손목 카메라 이미지를 가로로 합칩니다.
        if cap_wrist_right is not None and cap_wrist_left is not None:
            cap_wrist_combined = np.concatenate([cap_wrist_left, cap_wrist_right], axis=1)
            self.env_logger.add_data("obs_wrist_rgb", cap_wrist_combined)
        elif cap_wrist_right is not None:
            self.env_logger.add_data("obs_wrist_rgb", cap_wrist_right)

    def step(self, action: np.ndarray, record: bool = False, verbose: bool = False) -> Transition:
        success = False
        obs = self._previous_obs_dict  # in case action fails.

        ik_error = False
        try:
            obs, reward, terminal = self._task.step(action)
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            if verbose:
                print(e)
            terminal = True
            reward = 0.0
            ik_error = True

        self._i += 1

        return Transition(obs, reward, terminal)


    def _execute_waypoint(self, point, scene, arm, gripper, robot_shapes):
        from pyrep.const import ObjectType
        from rlbench.backend.exceptions import DemoError

        self._ignore_collisions_for_current_waypoint = False
        self._ignore_collisions_for_current_waypoint = point._ignore_collisions
        point.start_of_path()
        if point.skip:
            raise NotImplementedError("skip 하면 안되는데 구현 안함")

        grasped_object = gripper.get_grasped_objects()
        colliding_shapes = []
        for s in self._rlbench_env._pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE):
            if s in grasped_object:
                continue
            # if s in self._robot_shapes:
            #    continue
            if not s.is_collidable():
                continue
            if arm.check_arm_collision(s):
                colliding_shapes.append(s)
        logging.debug("got list of colliding objects: %s", ", ".join([s.get_name() for s in colliding_shapes]))
        [s.set_collidable(False) for s in colliding_shapes]
        try:
            path = point.get_path()
        except ConfigurationPathError as e:
            logging.error("Unable to get path %s", e)
            raise DemoError(f'Could not get a path for waypoint {point.name}.', task=self.task) from e
        finally:
            [s.set_collidable(True) for s in colliding_shapes]

        ext = point.get_ext()
        path.visualize()
        done = False
        while not path.step():
            scene.step()
        point.end_of_path()
        path.clear_visualization()

        # 그리퍼 제어 로직
        if len(ext) > 0:
            contains_param = False
            
            # 그리퍼 이름(left/right)을 결정하는 로직 수정
            if 'left' in ext:
                name = 'left'
            elif 'right' in ext:
                name = 'right'
            else:
                name = arm.get_name().split('_')[1] # 'Panda_leftArm' -> 'left'

            if 'open_gripper(' in ext:
                gripper.release()
                start_of_bracket = ext.index('open_gripper(') + 13
                contains_param = ext[start_of_bracket] != ')'
                if not contains_param:
                    done = False
                    while not done:
                        done = self._rlbench_env._scene.robot.actutate_gripper(1.0, 0.04,name)
                        scene.step()
            elif 'close_gripper(' in ext:
                start_of_bracket = ext.index('close_gripper(') + 14
                contains_param = ext[start_of_bracket] != ')'
                if not contains_param:
                    done = False
                    while not done:
                        done = self._rlbench_env._scene.robot.actutate_gripper(0.0, 0.04,name)
                        scene.step()

            if 'close_gripper(' in ext:
                for g_obj in scene.task.get_graspable_objects():
                    self._rlbench_env._scene.robot.grasp(g_obj,name)

    def run_demo_until_n_waypoint(self, task_name):
        # from pyrep.const import ObjectType
        # from pyrep.errors import ConfigurationPathError
        # from rlbench.backend.exceptions import WaypointError, BoundaryError, NoWaypointsError, DemoError
   
        # original
        WAYPOINT_MAP = {
            "vase_and_flower": {"right": [0, 2], "left": [1, 3]},
            "pour_water_to_cup": {"right": [1, 3], "left": [0, 2]},
            "put_object_in_crate": {"right": [1, 2], "left": [0, 4]},
            "wipe_dish_with_sponge": {"right": [0, 2], "left": [1, 3]},
            "put_object_in_box" : {"right": [0, 1], "left": [3, 4]},  
            "close_the_pot_lid" : {"right": [0, 1], "left": [3, 4]},
            "cheers_coke_easy" : {"right": [0, 1], "left": [3, 4]},
            "cheers_coke_hard" : {"right": [0, 1], "left": [3, 4]},
            "ring_the_bell_with_mallet" : {"right": [0, 1], "left": [2, 3]},
            "place_the_figure" : {"right": [0, 1], "left": [4, 5]},
            "plating_the_grilled_meat" : {"right":[2, 3], "left":[0, 1]},
            "hang_cup_top" : {"right": [2, 3], "left": [0, 1]},
            "scan_the_bottle" : {"right": [0, 1], "left": [4, 5]},
            "screw_the_bottle_cap" : {"right": [0, 1], "left": [4, 5]},
            
            "pour_water_to_cup_real": {"right": [0, 1], "left": [2, 3]},
            "wipe_dish_with_sponge_real": {"right": [0, 1], "left": [2, 3]},
            "put_truck_toy_in_basket_real": {"right": [2, 3], "left": [0, 1]},

        }

        # # mirror 
        # WAYPOINT_MAP = {
        #     "vase_and_flower_mirror": {"left": [0, 2], "right": [1, 3]},
        #     "pour_water_to_cup_mirror": {"left": [1, 3], "right": [0, 2]},
        #     "put_object_in_crate_mirror": {"left": [1, 2], "right": [0, 4]},
        #     "wipe_dish_with_sponge_mirror": {"left": [0, 2], "right": [1, 3]},
        #     "put_object_in_box_mirror" : {"left": [0, 1], "right": [3, 4]},
        #     "close_the_pot_lid_mirror" : {"left": [0, 1], "right": [3, 4]},
        #     "cheers_coke_easy_mirror" : {"left": [0, 1], "right": [3, 4]},
        #     "cheers_coke_hard_mirror" : {"left": [0, 1], "right": [3, 4]},
        #     "ring_the_bell_with_mallet_mirror" : {"left": [0, 1], "right": [2, 3]},
        #     "place_the_figure_mirror" : {"left": [0, 1], "right": [4, 5]},
        #     "plating_the_grilled_meat_mirror" : {"left":[2, 3], "right":[0, 1]},
        #     "hang_cup_top_mirror" : {"left": [2, 3], "right": [0, 1]},
        #     "scan_the_bottle_mirror" : {"left": [0, 1], "right": [4, 5]},
        #     "screw_the_bottle_cap_mirror" : {"left": [0, 1], "right": [4, 5]},
        # }
                
        scene = self._rlbench_env._scene
        if not scene._has_init_task: scene.init_task()
        if not scene._has_init_episode: scene.init_episode(scene._variation_index, randomly_place=False)

        all_waypoints = self.env._scene.task.get_waypoints()
        task_indices = WAYPOINT_MAP.get(task_name, {})
        right_indices = task_indices.get("right", [])
        left_indices = task_indices.get("left", [])
        right_waypoints = [all_waypoints[i] for i in right_indices]
        left_waypoints = [all_waypoints[i] for i in left_indices]
        
        right_arm, right_gripper = scene.robot.right_arm, scene.robot.right_gripper
        left_arm, left_gripper = scene.robot.left_arm, scene.robot.left_gripper
        robot_shapes = scene._robot_shapes 

        print("--- 오른팔 데모 실행 ---")
        for point in right_waypoints:
            self._execute_waypoint(point, scene, right_arm, right_gripper, robot_shapes)

        print("--- 왼팔 데모 실행 ---")
        for point in left_waypoints:
            self._execute_waypoint(point, scene, left_arm, left_gripper, robot_shapes)



    def start_tracking(self):
        assert self.pose_estimation_wrapper is not None

        if self.use_fp:
            # obj0에 대한 estimator 생성
            self.pose_estimation_wrapper.update_grasp_obj_name(self.visual_obj0_name)
            print("create FP estimator for obj0...", self.visual_obj0_name)
            self.pose_estimator_obj0 = self.pose_estimation_wrapper.create_estimator(debug_level=0)
            self.current_pose_estimation_obj0 = None

            # obj1에 대한 estimator 생성
            self.pose_estimation_wrapper.update_grasp_obj_name(self.visual_obj1_name)
            print("create FP estimator for obj1...", self.visual_obj1_name)
            self.pose_estimator_obj1 = self.pose_estimation_wrapper.create_estimator(debug_level=0)
            self.current_pose_estimation_obj1 = None

            self._start_tracking = True
        self._start_tracking = True

    def end_tracking(self):
        assert self.pose_estimation_wrapper is not None
        self.pose_estimator_obj0 = None
        self.pose_estimator_obj1 = None
        self.current_pose_estimation_obj0 = None
        self.current_pose_estimation_obj1 = None
        self._start_tracking = False


    def get_env_logger(self):
        return self.env_logger
        