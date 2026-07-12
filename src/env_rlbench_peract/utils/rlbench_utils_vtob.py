# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""
Wrappers for collecting object poses from demos
"""
from functools import partial
from os.path import exists, dirname, abspath, join
from typing import List, Callable

import numpy as np
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.object import Object, object_type_to_class
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.robots.arms.dual_panda import PandaLeft
from pyrep.robots.arms.dual_panda import PandaRight
from pyrep.robots.end_effectors.dual_panda_gripper import PandaGripperRight
from pyrep.robots.end_effectors.dual_panda_gripper import PandaGripperLeft
from rlbench.sim2real.domain_randomization_scene import DomainRandomizationScene


# from rlbench import ObservationConfig
from rlbench.backend.const import *
from rlbench.const import SUPPORTED_ROBOTS
from rlbench.backend.robot import Robot
from rlbench.backend.robot import BimanualRobot
from rlbench.backend.robot import UnimanualRobot

from rlbench.backend.scene import Scene
from rlbench.environment import Environment, DIR_PATH

from env_rlbench_peract.utils.rlbench_objects_vtob import task_object_dict
import logging


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
# (다른 import 구문...)


class MyScene(Scene):
    def __init__(self, *args, **kwargs):
        pose_wrapper = kwargs.pop('pose_estimation_wrapper', None)

        # 2. (선택사항) 꺼내온 값을 이 인스턴스의 속성으로 저장하여 나중에 쓸 수 있게 합니다.
        self.pose_estimation_wrapper = pose_wrapper
        super().__init__(*args, **kwargs)
      
    
    def _get_misc(self):
        def _get_cam_data(cam: VisionSensor, name: str):
            d = {}
            if cam.still_exists():
                d = {
                    '%s_extrinsics' % name: cam.get_matrix(),
                    '%s_intrinsics' % name: cam.get_intrinsic_matrix(),
                    '%s_near' % name: cam.get_near_clipping_plane(),
                    '%s_far' % name: cam.get_far_clipping_plane(),
                }
            return d
        misc = _get_cam_data(self._cam_over_shoulder_left, 'left_shoulder_camera')
        misc.update(_get_cam_data(self._cam_over_shoulder_right, 'right_shoulder_camera'))
        misc.update(_get_cam_data(self._cam_overhead, 'overhead_camera'))
        misc.update(_get_cam_data(self._cam_front, 'front_camera'))
        misc.update(_get_cam_data(self._cam_wrist_left, 'wrist_left_camera'))
        misc.update(_get_cam_data(self._cam_wrist_right, 'wrist_right_camera'))

        misc.update({"variation_index": self._variation_index})



        ee_pose_right = self.robot.right_arm.get_tip().get_pose()
        misc.update({f'ee_pose_right': ee_pose_right})

        ee_pose_left = self.robot.left_arm.get_tip().get_pose()
        misc.update({f'ee_pose_left': ee_pose_left})


        # add obj0 object pose
        task_name = self.task.get_name()
        task_variation_number = self._variation_index

        # add obj0 object pose


        grasp_obj0_name = task_object_dict[task_name]["grasp_object0_name"] if isinstance(task_object_dict[task_name]["grasp_object0_name"], str) else task_object_dict[task_name]["grasp_object0_name"][task_variation_number]
        grasp_obj0 = Shape(grasp_obj0_name)
        misc.update({f'grasp_obj0_name': grasp_obj0_name})
        misc.update({f'grasp_obj0_handle': grasp_obj0.get_handle()})
        # grasp_obj0_pose = grasp_obj0.get_pose()
        # misc.update({'grasp_obj0_pose': grasp_obj0_pose})

        # add obj1 object pose
        grasp_obj1_name = task_object_dict[task_name]["grasp_object1_name"] if isinstance(task_object_dict[task_name]["grasp_object1_name"], str) else task_object_dict[task_name]["grasp_object1_name"][task_variation_number]
        grasp_obj1 = Shape(grasp_obj1_name)
        misc.update({f'grasp_obj1_name': grasp_obj1_name})
        misc.update({f'grasp_obj1_handle': grasp_obj1.get_handle()})
        # grasp_obj1_pose = grasp_obj1.get_pose()
        # misc.update({'grasp_obj1_pose': grasp_obj1_pose})


        visual_obj0_name = grasp_obj0_name + '_visual'
        try:
            visual_obj0 = Shape(visual_obj0_name)
            if visual_obj0.still_exists():
                misc.update({f'visual_obj0_handle': visual_obj0.get_handle()})
                grasp_obj0_pose = visual_obj0.get_pose()
                misc.update({'grasp_obj0_pose': grasp_obj0_pose})
                misc.update({'visual_obj0_name': visual_obj0_name})
            else: # visual 객체가 없으면 grasp 객체 핸들을 사용
                misc.update({f'visual_obj0_handle': grasp_obj0.get_handle()})
                grasp_obj0_pose = grasp_obj0.get_pose()
                misc.update({'grasp_obj0_pose': grasp_obj0_pose})
        except: # Shape(visual_obj0_name)에서 오류 발생 시
            NotImplementedError("Not implemented")


        visual_obj1_name = grasp_obj1_name + '_visual'
        try:
            visual_obj1 = Shape(visual_obj1_name)
            if visual_obj1.still_exists():
                misc.update({f'visual_obj1_handle': visual_obj1.get_handle()})
                grasp_obj1_pose = visual_obj1.get_pose()
                misc.update({'grasp_obj1_pose': grasp_obj1_pose})
                misc.update({'visual_obj1_name': visual_obj1_name})
            else: # visual 객체가 없으면 grasp 객체 핸들을 사용
                misc.update({f'visual_obj1_handle': grasp_obj1.get_handle()})
                grasp_obj1_pose = grasp_obj1.get_pose()
                misc.update({'grasp_obj1_pose': grasp_obj1_pose})

        except: # Shape(visual_obj1_name)에서 오류 발생 시
            NotImplementedError("Not implemented")



        from utils.pose_utils import get_rel_pose, euler_from_quaternion, get_average_pose , pose_quaternion_to_pose_euler
        from scipy.spatial.transform import Rotation as R
        if (not hasattr(self, '_world_anchor_pose') or self._world_anchor_pose is None):
            logging.info("Calculating and caching WORLD anchor pose...")
            self._world_anchor_pose = get_average_pose(grasp_obj0_pose, grasp_obj1_pose)
        
        # 3. 매 프레임 캐시된 앵커를 사용합니다.
        object_anchor = self._world_anchor_pose
        misc.update({'object_anchor_pose': object_anchor})
        
        # 4. 이제 object_anchor는 "고정된" 월드 앵커입니다.
        obj0_pose_in_anchor = get_rel_pose(object_anchor, grasp_obj0_pose)  # WA(fixed) -> WO : AO
        obj1_pose_in_anchor = get_rel_pose(object_anchor, grasp_obj1_pose)
        misc.update({'grasp_obj0_anchor_world':obj0_pose_in_anchor})
        misc.update({'grasp_obj1_anchor_world':obj1_pose_in_anchor})

        # FoundationPose 추정치 추가
        
        import utils.transform_utils as T
        if self.pose_estimation_wrapper is not None:

            # 1. 헬퍼 함수를 if문 바깥으로 이동시킵니다 (get_misc 메서드 내부에는 둡니다).
            def _get_pose_est(obj_name: str, pose_estimator, cam_name: str):
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
                
                def fp_frame_to_cam_frame(fp_mat):
                    from scipy.spatial.transform import Rotation
                    
                    # 1. FP raw 좌표계(OpenCV/GL) -> 시뮬레이터 카메라 좌표계 보정
                    r = Rotation.from_euler('zyx', [np.pi, 0, 0])
                    tf_flip = np.eye(4)
                    tf_flip[:3, :3] = r.as_matrix()
                    fp_mat_cam_frame = np.matmul(tf_flip, fp_mat) # 이것이 올바른 C->O 행렬

                    # 2. 4x4 행렬 -> 7D 포즈 [x,y,z, qx,qy,qz,qw]
                    pose = np.concatenate([fp_mat_cam_frame[:3, 3], T.mat2quat(fp_mat_cam_frame[:3, :3])])
                    return pose
                
                cam = VisionSensor(cam_name)
                cam_mask = VisionSensor(cam_name + '_mask')
                
                fp_rgb, fp_depth, fp_pcd = get_rgb_depth(
                    sensor=cam,
                    get_rgb=True, get_depth=True, get_pcd=True, 
                    rgb_noise=None, depth_noise=None, 
                    depth_in_meters=True,
                    )
                mask = get_mask(cam_mask, masks_as_one_channel=True)
                fp_mask = get_seg_mask([Shape(obj_name)], mask)
                
                K = cam.get_intrinsic_matrix()
                R = cam.get_matrix()
                K[:2, :2] *= -1 # !! FoundationPose 형식에 맞게 변환
                color = fp_rgb
                depth = fp_depth
                select_seg_id = 1
                ob_mask = (fp_mask == select_seg_id).astype(bool).astype(np.uint8)

                if np.sum(ob_mask) < 10: return None
                
                # FoundationPose 실행
                # 매 프레임 추적을 위해 register/track 로직을 유지합니다.
               
                
                if pose_estimator.pose_last is None:
                    fp_mat = pose_estimator.register(K=K, rgb=color, depth=depth, ob_mask=ob_mask, iteration=5)
                else:
                    fp_mat = pose_estimator.track_one(rgb=color, depth=depth, K=K, iteration=2)
                    
                
                pose_est, world_mat = fp_frame_to_world_frame(fp_mat, R)
                pose_est_in_cam_frame = fp_frame_to_cam_frame(fp_mat)

                #return pose_est_in_cam_frame
                return pose_est_in_cam_frame
                
            # --- [수정] task_name을 get_name() 메소드로 가져옵니다. ---
            # current_task_name = self.task.get_name()

            # tasks_using_visual_name = {
            #     'vase_and_flower', 
            #     'cheers_coke_easy', 
            #     'cheers_coke_hard',
            #     'ring_the_bell_with_mallet',
            #     'place_the_figure',
            #     'put_obejct_in_crate'       
            # }

            # if current_task_name in tasks_using_visual_name:
            obj0_name_for_fp = visual_obj0_name
            obj1_name_for_fp = visual_obj1_name
            logging.info(f"Task is '{self.task.get_name()}'. Using '{visual_obj0_name}' for FP estimator 0.")
            logging.info(f"Task is '{self.task.get_name()}'. Using '{visual_obj1_name}' for FP estimator 1.")
            # else:
                # obj0_name_for_fp = grasp_obj0_name
                # obj1_name_for_fp = grasp_obj1_name

            # 2. Estimator를 self에 캐싱합니다. (첫 프레임에만 생성)
            # (이 변수들이 __init__에서 None으로 초기화되었다고 가정합니다)
            if not hasattr(self, '_estimator0') or self._estimator0 is None:
                logging.info("Creating and caching FoundationPose estimators...")
                self.pose_estimation_wrapper.update_grasp_obj_name(obj0_name_for_fp)
                self._estimator0 = self.pose_estimation_wrapper.create_estimator(debug_level=0)
                
                self.pose_estimation_wrapper.update_grasp_obj_name(obj1_name_for_fp)
                self._estimator1 = self.pose_estimation_wrapper.create_estimator(debug_level=0)

            # 3. 매 프레임 포즈를 추정합니다.
            fp_grasp_obj0_pose = _get_pose_est(obj0_name_for_fp, self._estimator0, 'cam_front')
            #fp_grasp_obj0_pose_overhead = _get_pose_est(grasp_obj0_name, self._estimator0, 'cam_overhead')
            fp_grasp_obj1_pose = _get_pose_est(obj1_name_for_fp, self._estimator1, 'cam_front')

            # 4. 첫 프레임에만 앵커를 계산하고 캐싱합니다.
            # (self._camera_anchor_pose가 __init__에서 None으로 초기화되었다고 가정)
            if (not hasattr(self, '_camera_anchor_pose') or self._camera_anchor_pose is None):
                if fp_grasp_obj0_pose is not None and fp_grasp_obj1_pose is not None:
                    logging.info("Calculating and caching camera anchor pose...")
                    self._camera_anchor_pose = get_average_pose(fp_grasp_obj0_pose, fp_grasp_obj1_pose)#up_vector_for_edge_case=np.array([0.0, 1.0, 0.0]))

            # 5. 매 프레임 최신 포즈와 캐시된 앵커를 사용해 misc에 추가
            if fp_grasp_obj0_pose is not None:
                misc.update({'fp_grasp_obj0_pose': fp_grasp_obj0_pose})
            if fp_grasp_obj1_pose is not None:
                misc.update({'fp_grasp_obj1_pose': fp_grasp_obj1_pose})
                
            if self._camera_anchor_pose is not None:
                misc.update({'camera_anchor_pose': self._camera_anchor_pose})
                
                # 앵커 기준 상대좌표 계산 (최신 포즈 + 고정된 앵커)
                if fp_grasp_obj0_pose is not None and fp_grasp_obj1_pose is not None:
                    fp_grasp_anchor_obj0_pose = get_rel_pose(self._camera_anchor_pose, fp_grasp_obj0_pose)
                    fp_grasp_anchor_obj1_pose = get_rel_pose(self._camera_anchor_pose, fp_grasp_obj1_pose)
                    misc.update({'fp_grasp_anchor_obj0_pose': fp_grasp_anchor_obj0_pose})
                    misc.update({'fp_grasp_anchor_obj1_pose': fp_grasp_anchor_obj1_pose})
                    #print(f"WORLD ANCHOR :{object_anchor}, CAMERA ANCHOR : {self._camera_anchor_pose}")

                    #######################################################################################
                    left_grasped = self.robot.left_gripper.get_grasped_objects()
                    right_grasped = self.robot.right_gripper.get_grasped_objects()
                    is_left_grasping = len(left_grasped) > 0
                    is_right_grasping = len(right_grasped) > 0

                    # 2. GT(Shape)가 참조하는 객체 이름 확인
                    gt_obj_name = "None"
                    try:
                        # grasp_obj0는 이 if 블록보다 이전에 정의되었습니다.
                        gt_obj_name = grasp_obj0.get_name()
                    except Exception as e:
                        gt_obj_name = f"Error: {e}"

                    print("--- DEBUG FRAME ---")
                    print(f"Gripper Status: Left={is_left_grasping}, Right={is_right_grasping}")
                    print(f"GT (Shape) is referencing: {gt_obj_name} (Expected: {grasp_obj0_name})")
                    
                    # 3. 사용자 요청 print문 (기존)
                    fp_grasp_anchor_obj0_pose_euler = pose_quaternion_to_pose_euler(fp_grasp_anchor_obj0_pose)
                    obj0_pose_in_anchor_euler = pose_quaternion_to_pose_euler(obj0_pose_in_anchor)

                    print(f" camera 로부터 anchor 기준 : {fp_grasp_anchor_obj0_pose_euler}")
                    print(f" World 로부터 anchor 기준 : {obj0_pose_in_anchor_euler}")
                    print("-------------------")
    
                    #print(f" camera 로부터 anchor 기준 : {fp_grasp_anchor_obj0_pose}, World 로부터 anchor 기준 : {obj0_pose_in_anchor}")
                    #######################################################################################
                    #print(grasp_obj0_pose)


        # obj_pose_relative_to_init = get_rel_pose_init(init_pose, grasped_obj)
        obj1_pose_relative_to_obj0 = get_rel_pose(grasp_obj0_pose, grasp_obj1_pose) # !! be careful about the order (target, grasp_object)
        misc.update({'obj1_pose_relative_to_obj0': obj1_pose_relative_to_obj0})

        obj0_pose_relative_to_obj1 = get_rel_pose(grasp_obj1_pose, grasp_obj0_pose)
        misc.update({'obj0_pose_relative_to_obj1': obj0_pose_relative_to_obj1})

        obj0_pose_relative_to_obj0 = get_rel_pose(grasp_obj0_pose, grasp_obj0_pose) # (0,0,0) 만드는 과정 (obj0 을 기준으로 기준점 잡을때.)
        misc.update({'obj0_pose_relative_to_obj0': obj0_pose_relative_to_obj0}) 

        obj1_pose_relative_to_obj1 = get_rel_pose(grasp_obj1_pose, grasp_obj1_pose) # (0,0,0) (obj1 기준)
        misc.update({'obj1_pose_relative_to_obj1': obj1_pose_relative_to_obj1})



        # convert from quaternion to euler
        obj1_pose_relative_to_obj0_euler = np.zeros(6)
        obj1_pose_relative_to_obj0_euler[:3] = obj1_pose_relative_to_obj0[:3]
        obj1_pose_relative_to_obj0_euler[3:] = euler_from_quaternion(*obj1_pose_relative_to_obj0[3:])
        misc.update({'obj1_pose_relative_to_obj0_euler': obj1_pose_relative_to_obj0_euler})

        obj0_pose_relative_to_obj1_euler = np.zeros(6)
        obj0_pose_relative_to_obj1_euler[:3] = obj0_pose_relative_to_obj1[:3]
        obj0_pose_relative_to_obj1_euler[3:] = euler_from_quaternion(*obj0_pose_relative_to_obj1[3:])
        misc.update({'obj0_pose_relative_to_obj1_euler': obj0_pose_relative_to_obj1_euler})

        obj0_pose_relative_to_obj0_euler = np.zeros(6)
        obj0_pose_relative_to_obj0_euler[:3] = obj0_pose_relative_to_obj0[:3]
        obj0_pose_relative_to_obj0_euler[3:] = euler_from_quaternion(*obj0_pose_relative_to_obj0[3:])
        misc.update({'obj0_pose_relative_to_obj0_euler': obj0_pose_relative_to_obj0_euler})

        obj1_pose_relative_to_obj1_euler = np.zeros(6)
        obj1_pose_relative_to_obj1_euler[:3] = obj1_pose_relative_to_obj1[:3]
        obj1_pose_relative_to_obj1_euler[3:] = euler_from_quaternion(*obj1_pose_relative_to_obj1[3:])
        misc.update({'obj1_pose_relative_to_obj1_euler': obj1_pose_relative_to_obj1_euler})
        

        return misc


class MyEnvironmentPeract(Environment):
    def __init__(self, *args, **kwargs):
        pose_wrapper = kwargs.pop('pose_estimation_wrapper', None)
        self.pose_estimation_wrapper = pose_wrapper
        super().__init__(*args, **kwargs)

    def launch(self):
        if self._pyrep is not None:
            raise RuntimeError('Already called launch!')
        self._pyrep = PyRep()
        if self._robot_setup == 'dual_panda':
            self._pyrep.launch(join(DIR_PATH, BIMANUAL_TTT_FILE), headless=self._headless)
        else:
            self._pyrep.launch(join(DIR_PATH, TTT_FILE), headless=self._headless)

        arm_class, gripper_class, _ = SUPPORTED_ROBOTS[
            self._robot_setup]


        if self._robot_setup == 'dual_panda':

            logging.info("Using dual panda robot")
           
            #panda_arm = Panda()
            #panda_pos = panda_arm.get_position()
            #panda_arm.remove()

            right_arm = PandaRight()
            left_arm = PandaLeft()
            right_gripper = PandaGripperRight()
            left_gripper = PandaGripperLeft()

            # ..not updating position as we assume that the scene already contains two pandas which are placed correctly     
            #relative_left_position = left_arm.get_position(relative_to=right_arm)            
            #right_arm.set_position(panda_pos)
            #left_arm.set_position(relative_left_position, relative_to=right_arm)

            self._robot = BimanualRobot(right_arm, right_gripper, left_arm, left_gripper)

        # We assume the panda is already loaded in the scene.
        elif self._robot_setup != 'panda':
            # Remove the panda from the scene
            panda_arm = Panda()
            panda_pos = panda_arm.get_position()
            panda_arm.remove()
            arm_path = join(DIR_PATH, 'robot_ttms', self._robot_setup + '.ttm')
            self._pyrep.import_model(arm_path)
            arm, gripper = arm_class(), gripper_class()
            arm.set_position(panda_pos)
            self._robot = UnimanualRobot(arm, gripper)
        else:
            arm, gripper = arm_class(), gripper_class()
            self._robot = UnimanualRobot(arm, gripper)


        if self._randomize_every is None:
            self._scene = MyScene(
                self._pyrep, self._robot, self._obs_config, self._robot_setup,
                pose_estimation_wrapper=self.pose_estimation_wrapper)
        else:
            self._scene = DomainRandomizationScene(
                self._pyrep, self._robot, self._obs_config, self._robot_setup,
                self._randomize_every, self._frequency,
                self._visual_randomization_config,
                self._dynamics_randomization_config)

        self._action_mode.arm_action_mode.set_control_mode(self._robot)


from rlbench.noise_model import NoiseModel
from rlbench.backend.utils import image_to_float_array, rgb_handles_to_mask


def get_rgb_depth(sensor: VisionSensor, get_rgb: bool, get_depth: bool,
                get_pcd: bool, rgb_noise: NoiseModel,
                depth_noise: NoiseModel, depth_in_meters: bool):
    rgb = depth = pcd = None
    if sensor is not None and (get_rgb or get_depth):
        sensor.handle_explicitly()
        if get_rgb:
            rgb = sensor.capture_rgb()
            if rgb_noise is not None:
                rgb = rgb_noise.apply(rgb)
            rgb = np.clip((rgb * 255.).astype(np.uint8), 0, 255)
        if get_depth or get_pcd:
            depth = sensor.capture_depth(depth_in_meters)
            if depth_noise is not None:
                depth = depth_noise.apply(depth)
        if get_pcd:
            depth_m = depth
            if not depth_in_meters:
                near = sensor.get_near_clipping_plane()
                far = sensor.get_far_clipping_plane()
                depth_m = near + depth * (far - near)
            pcd = sensor.pointcloud_from_depth(depth_m)
            if not get_depth:
                depth = None
    return rgb, depth, pcd

def get_mask(sensor: VisionSensor, masks_as_one_channel=True):
    masks_as_one_channel = True
    mask_fn = rgb_handles_to_mask if masks_as_one_channel else lambda x: x

    mask = None
    if sensor is not None:
        sensor.handle_explicitly()
        mask = mask_fn(sensor.capture_rgb())
    return mask


def get_seg_mask(obj_list, mask):
    mask_id_list = np.unique(mask)

    select_id_list = []
    for obj in obj_list:
        id = obj.get_handle()
        name = obj.get_object_name(obj.get_handle())
        if name == "cup3":
            id = 97 # !! not sure why we need this
        if id not in mask_id_list and (id + 1) in mask_id_list:
            id = id + 1 # !! It solves the inconsisencies between object id and mask id (my guess is the object id is rounded down while the mask id could be rounded up/down)
        # if id not in mask_id_list:
        #     print(f"object {name} ({id}) is not found in {mask_id_list}")
        select_id_list.append(id)
    
    # print("select_id_list", select_id_list)

    seg_mask = np.zeros_like(mask)
    for seg_id, obj_id in enumerate(select_id_list):
        seg_mask[mask == obj_id] = seg_id + 1 # start from 1
    return seg_mask
