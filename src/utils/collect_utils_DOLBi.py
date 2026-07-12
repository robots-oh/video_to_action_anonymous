# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from math import e
import os, shutil
import zarr
import numpy as np
from termcolor import cprint
from scipy.spatial.transform import Rotation as R
from utils.pose_utils import get_rel_pose, euler_from_quaternion, quaternion_from_euler, relative_to_target_to_world, calculate_action, calculate_goal_pose, apply_action_to_pose, get_rel_pose2


def check_pose_diff(pose1, pose2, trans_threshold, rot_threshold):
    action = calculate_action(pose1, pose2)
    dist = np.linalg.norm(action[:3])
    
    rot = R.from_quat(pose1[3:]) # (x, y, z, w)
    rotvec1 = rot.as_rotvec()
    rot = R.from_quat(pose2[3:]) # (x, y, z, w)
    rotvec2 = rot.as_rotvec()
    angle_diff = rotvec2 - rotvec1
    
    return dist >= trans_threshold or np.max(angle_diff) >= rot_threshold



# clean keyframe
# def clean_keyframe(key_frame_idx_list, obj_pose_relative_to_target_list, dist_threshold=0.01):
#     new_frame_idx_list = []
#     for idx in range(len(key_frame_idx_list)-1):
#         previous_frame_idx = key_frame_idx_list[idx-1]
#         cur_frame_idx = key_frame_idx_list[idx]
#         next_frame_idx = key_frame_idx_list[idx+1]

#         dist = (obj_pose_relative_to_target_list[next_frame_idx] - obj_pose_relative_to_target_list[cur_frame_idx])
#         dist = np.linalg.norm(dist[:3])
#         # print(dist)

#         if dist >= dist_threshold:
#             new_frame_idx_list.append(cur_frame_idx)
#     return new_frame_idx_list

def clean_keyframe(key_frame_idx_list, obj_anchor_pose,trans_threshold=0.05, rot_threshold=np.radians(20)):
    new_frame_idx_list = []
    prev_frame_idx = key_frame_idx_list[0]
    new_frame_idx_list.append(key_frame_idx_list[0])    # add initial frame index

    
    for idx in range(1, len(key_frame_idx_list)): # ignore first and last frame
        cur_frame_idx = key_frame_idx_list[idx]
        far = check_pose_diff(
            obj_anchor_pose[prev_frame_idx], obj_anchor_pose[cur_frame_idx],
            trans_threshold=trans_threshold, rot_threshold=rot_threshold,
        )
        
        if far:
            new_frame_idx_list.append(cur_frame_idx)    # add current frame index
            prev_frame_idx = cur_frame_idx  # update previous frame index for next iterarion


    # # check if add final frame
    # far3 = check_pose_diff(
    #     obj_world_pose[new_frame_idx_list[-1]], obj_world_pose[key_frame_idx_list[-1]],
    #     trans_threshold=trans_threshold*0.8, rot_threshold=rot_threshold*0.8,
    # )
    # if far3:   # only add if not too close to next keypoint
    #     new_frame_idx_list.append(key_frame_idx_list[-1])    # add last frame index_list
    if len(key_frame_idx_list) > 1 and new_frame_idx_list[-1] != key_frame_idx_list[-1]:
        new_frame_idx_list.append(key_frame_idx_list[-1])

    return new_frame_idx_list


def collect_narr_function(poses_dict_list, lang, intermediate_frame_length=1):
    # per demo        
    # img_arrays_sub = []
    # point_cloud_arrays_sub = []
    # depth_arrays_sub = []
    # state_arrays_sub = []
    # state_in_world_arrays_sub = []
    # state_next_arrays_sub = []
    # goal_arrays_sub = []
    # action_arrays_sub = []
    # total_count_sub = 0

    arrays_sub_dict_list = []
    total_count_sub_list = []


    for _, poses_dict in enumerate(poses_dict_list):
        arrays_sub_dict = {
            "obj0_to_obj1_state": [],                # obj0 current grasp obj pose
            #"obj0_state_in_world": [],       # current grasp obj0 pose in world coordinate
            "obj0_state_in_anchor": [],

            "obj1_to_obj0_state": [],                # obj1 current grasp obj pose
            #"obj1_state_in_world": [],       # current grasp obj1 pose in world coordinate
            "obj1_state_in_anchor": [],

            "obj0_to_obj1_state_next": [],           # obj0 next grasp obj pose
            #"obj0_state_next_in_world": [],  # next grasp obj0 pose in world coordinate
            "obj0_state_next_in_anchor": [],

            "obj1_to_obj0_state_next": [],           # obj1 next grasp obj pose
            #"obj1_state_next_in_world": [],  # next grasp obj1 pose in world coordinate
            "obj1_state_next_in_anchor": [],

            "obj0_to_obj1_action": [],               # action (from current grasp obj pose to next grasp obj pose)
            "obj1_to_obj0_action": [],               # action (from current grasp obj pose to next grasp obj pose)

            "obj0_anchor_action": [],
            "obj1_anchor_action": [],
            
            "progress": [],
            "progress_binary": [],

        }
        total_count_sub = 0



        obj0_pose_in_anchor_list = poses_dict["grasp_anchor_obj0_pose"]
        obj1_pose_in_anchor_list = poses_dict["grasp_anchor_obj1_pose"]
        obj0_anchor_pose_seq_length = len(obj0_pose_in_anchor_list) # 1. 여기서 뽑힌거 lyength 비교해서 사진이랑 비교
        obj1_anchor_pose_seq_length = len(obj1_pose_in_anchor_list)  
        obj0_anchor_key_frame_idx_list = range(0, obj0_anchor_pose_seq_length, intermediate_frame_length)
        obj1_anchor_key_frame_idx_list = range(0, obj1_anchor_pose_seq_length, intermediate_frame_length)

        assert len(obj0_anchor_key_frame_idx_list) == len(obj0_pose_in_anchor_list)  
        assert len(obj1_anchor_key_frame_idx_list) == len(obj1_pose_in_anchor_list)

        obj0_anchor_key_frame_idx_list = clean_keyframe(obj0_anchor_key_frame_idx_list, obj0_pose_in_anchor_list)
        obj1_anchor_key_frame_idx_list = clean_keyframe(obj1_anchor_key_frame_idx_list, obj1_pose_in_anchor_list)

        anchor_key_frame_idx_list = sorted(list(set(obj0_anchor_key_frame_idx_list ) | set(obj1_anchor_key_frame_idx_list)))

        # get each frame by key frame idx
        action = None
        for idx in range(len(anchor_key_frame_idx_list)-1):

            previous_frame_idx = anchor_key_frame_idx_list[idx-1]
            cur_frame_idx = anchor_key_frame_idx_list[idx]
            next_frame_idx = anchor_key_frame_idx_list[idx+1]

            obj0_cur_state_in_anchor = poses_dict["grasp_anchor_obj0_pose"][cur_frame_idx]  # Anchor 기준 obj0 의 pose
            obj1_cur_state_in_anchor = poses_dict["grasp_anchor_obj1_pose"][cur_frame_idx]  # Anchor 기준 obj1 의 pose

            obj0_cur_state_next_in_anchor = poses_dict["grasp_anchor_obj0_pose"][next_frame_idx]  # Anchor 기준 obj0 의 next pose
            obj1_cur_state_next_in_anchor = poses_dict["grasp_anchor_obj1_pose"][next_frame_idx]  # Anchor 기준 obj0 의 next pose

            obj0_to_obj1_cur_state = poses_dict["grasp_obj0_pose_relative_to_grasp_obj1"][cur_frame_idx] # current pose
            obj0_to_obj1_cur_state_next = poses_dict["grasp_obj0_pose_relative_to_grasp_obj1"][next_frame_idx] # next pose
           
            obj1_to_obj0_cur_state = poses_dict["grasp_obj1_pose_relative_to_grasp_obj0"][cur_frame_idx] # current pose
            obj1_to_obj0_cur_state_next = poses_dict["grasp_obj1_pose_relative_to_grasp_obj0"][next_frame_idx] # next pose
            

            # quaternion
            obj0_to_obj1_cur_action = calculate_action(obj0_to_obj1_cur_state, obj0_to_obj1_cur_state_next)
            obj1_to_obj0_cur_action = calculate_action(obj1_to_obj0_cur_state, obj1_to_obj0_cur_state_next)
            
            obj0_anchor_cur_action = calculate_action(obj0_cur_state_in_anchor, obj0_cur_state_next_in_anchor) # Anchor(world) 기준 위치
            obj1_anchor_cur_action = calculate_action(obj1_cur_state_in_anchor, obj1_cur_state_next_in_anchor)
            

            # save data
            total_count_sub += 1

            arrays_sub_dict["obj0_to_obj1_state"].append(obj0_to_obj1_cur_state)
            arrays_sub_dict["obj0_state_in_anchor"].append(obj0_cur_state_in_anchor)

            arrays_sub_dict["obj1_to_obj0_state"].append(obj1_to_obj0_cur_state)
            arrays_sub_dict["obj1_state_in_anchor"].append(obj1_cur_state_in_anchor)

            arrays_sub_dict["obj0_to_obj1_state_next"].append(obj0_to_obj1_cur_state_next)
            arrays_sub_dict["obj0_state_next_in_anchor"].append(obj0_cur_state_next_in_anchor)

            arrays_sub_dict["obj1_to_obj0_state_next"].append(obj1_to_obj0_cur_state_next)
            arrays_sub_dict["obj1_state_next_in_anchor"].append(obj1_cur_state_next_in_anchor)

            arrays_sub_dict["obj0_to_obj1_action"].append(obj0_to_obj1_cur_action)
            arrays_sub_dict["obj1_to_obj0_action"].append(obj1_to_obj0_cur_action)

            arrays_sub_dict["obj0_anchor_action"].append(obj0_anchor_cur_action)
            arrays_sub_dict["obj1_anchor_action"].append(obj1_anchor_cur_action)



    

        # add final frame
        cur_frame_idx = -1
        obj0_cur_state_in_anchor = poses_dict["grasp_anchor_obj0_pose"][-1]
        obj1_cur_state_in_anchor = poses_dict["grasp_anchor_obj1_pose"][-1]


        obj0_cur_state_next_in_anchor = obj0_cur_state_in_anchor
        obj1_cur_state_next_in_anchor = obj1_cur_state_in_anchor
        obj0_to_obj1_cur_state = poses_dict["grasp_obj0_pose_relative_to_grasp_obj1"][-1] # current pose
        obj0_to_obj1_cur_state_next = obj0_to_obj1_cur_state
        obj1_to_obj0_cur_state = poses_dict["grasp_obj1_pose_relative_to_grasp_obj0"][-1] # current pose
        obj1_to_obj0_cur_state_next = obj1_to_obj0_cur_state
        obj0_to_obj1_cur_action = calculate_action(obj0_to_obj1_cur_state, obj0_to_obj1_cur_state_next)
        obj1_to_obj0_cur_action = calculate_action(obj1_to_obj0_cur_state, obj1_to_obj0_cur_state_next)
        obj0_anchor_cur_action = calculate_action(obj0_cur_state_in_anchor, obj0_cur_state_next_in_anchor)
        obj1_anchor_cur_action = calculate_action(obj1_cur_state_in_anchor, obj1_cur_state_next_in_anchor)


        # save final frame data
        total_count_sub += 1

        arrays_sub_dict["obj0_to_obj1_state"].append(obj0_to_obj1_cur_state)
        #arrays_sub_dict["obj0_state_in_world"].append(obj0_cur_state_in_world)
        arrays_sub_dict["obj0_state_in_anchor"].append(obj0_cur_state_in_anchor)

        arrays_sub_dict["obj1_to_obj0_state"].append(obj1_to_obj0_cur_state)
        #arrays_sub_dict["obj1_state_in_world"].append(obj1_cur_state_in_world)
        arrays_sub_dict["obj1_state_in_anchor"].append(obj1_cur_state_in_anchor)

        arrays_sub_dict["obj0_to_obj1_state_next"].append(obj0_to_obj1_cur_state_next)
        #arrays_sub_dict["obj0_state_next_in_world"].append(obj0_cur_state_next_in_world)
        arrays_sub_dict["obj0_state_next_in_anchor"].append(obj0_cur_state_next_in_anchor)

        arrays_sub_dict["obj1_to_obj0_state_next"].append(obj1_to_obj0_cur_state_next)
        #arrays_sub_dict["obj1_state_next_in_world"].append(obj1_cur_state_next_in_world)
        arrays_sub_dict["obj1_state_next_in_anchor"].append(obj1_cur_state_next_in_anchor)

        arrays_sub_dict["obj0_to_obj1_action"].append(obj0_to_obj1_cur_action)
        arrays_sub_dict["obj1_to_obj0_action"].append(obj1_to_obj0_cur_action)

        #arrays_sub_dict["obj0_world_action"].append(obj0_world_cur_action)
        #arrays_sub_dict["obj1_world_action"].append(obj1_world_cur_action)

        arrays_sub_dict["obj0_anchor_action"].append(obj0_anchor_cur_action)
        arrays_sub_dict["obj1_anchor_action"].append(obj1_anchor_cur_action)

        # point_cloud_arrays_sub.append(time_step.observation_pointcloud)
        # depth_arrays_sub.append(time_step.observation_depth)

        # add prgoress
        progress = np.linspace(0., 1., num=len(anchor_key_frame_idx_list)).reshape(-1, 1)
        arrays_sub_dict["progress"].extend(progress)
        progress_binary = np.zeros_like(progress)
        progress_binary[-1] = 1.
        progress_binary[-2] = 0.9
        progress_binary[-3] = 0.8
        if len(progress_binary) >= 4:
            progress_binary[-4] = 0.7
        if len(progress_binary) >= 5:
            progress_binary[-5] = 0.6
        if len(progress_binary) >= 6:
            progress_binary[-6] = 0.5
        arrays_sub_dict["progress_binary"].extend(progress_binary)


        arrays_sub_dict_list.append(arrays_sub_dict)
        total_count_sub_list.append(total_count_sub)

    return arrays_sub_dict_list, total_count_sub_list


def save_zarr(
        save_dir,
        obj0_to_obj1_state_arrays,
        obj0_state_in_anchor_arrays,
        #obj0_state_in_world_arrays,

        obj0_to_obj1_state_next_arrays,
        obj0_state_next_in_anchor_arrays,
        #obj0_state_next_in_world_arrays,

        obj1_to_obj0_state_arrays,
        obj1_state_in_anchor_arrays,
        #obj1_state_in_world_arrays,

        obj1_to_obj0_state_next_arrays,
        obj1_state_next_in_anchor_arrays,
        #obj1_state_next_in_world_arrays,

        #obj0_world_action_arrays,
        #obj1_world_action_arrays,
        obj0_anchor_action_arrays,
        obj1_anchor_action_arrays,

        progress_arrays,
        progress_binary_arrays,
        variation_arrays,
        episode_ends_arrays,
    ):
    # img_arrays = np.stack(img_arrays, axis=0)
    # if img_arrays.shape[1] == 3: # make channel last
    #     img_arrays = np.transpose(img_arrays, (0,2,3,1))
    obj0_to_obj1_state_arrays = np.stack(obj0_to_obj1_state_arrays, axis=0)
    obj0_state_in_anchor_arrays = np.stack(obj0_state_in_anchor_arrays, axis=0)
    #obj0_state_in_world_arrays = np.stack(obj0_state_in_world_arrays, axis=0)

    obj1_to_obj0_state_arrays = np.stack(obj1_to_obj0_state_arrays, axis=0)
    obj1_state_in_anchor_arrays = np.stack(obj1_state_in_anchor_arrays, axis=0)
    #obj1_state_in_world_arrays = np.stack(obj1_state_in_world_arrays, axis=0)


    obj0_to_obj1_state_next_arrays = np.stack(obj0_to_obj1_state_next_arrays, axis=0)
    obj0_state_next_in_anchor_arrays = np.stack(obj0_state_next_in_anchor_arrays, axis=0)
    #obj0_state_next_in_world_arrays = np.stack(obj0_state_next_in_world_arrays, axis=0)

    obj1_to_obj0_state_next_arrays = np.stack(obj1_to_obj0_state_next_arrays, axis=0)   
    obj1_state_next_in_anchor_arrays = np.stack(obj1_state_next_in_anchor_arrays, axis=0)
    #obj1_state_next_in_world_arrays = np.stack(obj1_state_next_in_world_arrays, axis=0)

    # obj0_to_obj1_action_arrays = np.stack(obj0_to_obj1_action_arrays, axis=0)
    # obj1_to_obj0_action_arrays = np.stack(obj1_to_obj0_action_arrays, axis=0)
    #obj0_world_action_arrays = np.stack(obj0_world_action_arrays, axis=0)
    #obj1_world_action_arrays = np.stack(obj1_world_action_arrays, axis=0)   
    obj0_anchor_action_arrays = np.stack(obj0_anchor_action_arrays, axis=0)
    obj1_anchor_action_arrays = np.stack(obj1_anchor_action_arrays, axis=0)

    progress_arrays = np.stack(progress_arrays, axis=0)
    progress_binary_arrays = np.stack(progress_binary_arrays, axis=0)
    variation_arrays = np.stack(variation_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)


    # print(obj0_to_obj1_state_arrays.shape)
    # print(obj0_state_in_anchor_arrays.shape)
    # print(obj1_to_obj0_state_arrays.shape)
    # print(obj1_state_in_anchor_arrays.shape)

    # print(obj0_to_obj1_state_next_arrays.shape)
    # print(obj0_state_next_in_anchor_arrays.shape)
    # print(obj1_to_obj0_state_next_arrays.shape)
    # print(obj1_state_next_in_anchor_arrays.shape)
    # # print(obj0_to_obj1_action_arrays.shape)
    # # print(obj1_to_obj0_action_arrays.shape)
    # print(obj0_anchor_action_arrays.shape)
    # print(obj1_anchor_action_arrays.shape)
    # print(progress_arrays.shape)
    # print(progress_binary_arrays.shape)
    # print(variation_arrays.shape)
    # print(episode_ends_arrays.shape)

    # debug only
    episode_start_index = 0
    for idx, episode_end_index in enumerate(episode_ends_arrays):
        obj0_to_obj1_cur_state_array = obj0_to_obj1_state_arrays[episode_start_index: episode_end_index]
        obj0_state_in_anchor_array = obj0_state_in_anchor_arrays[episode_start_index: episode_end_index]

        obj1_to_obj0_cur_state_array = obj1_to_obj0_state_arrays[episode_start_index: episode_end_index]
        obj1_state_in_anchor_array = obj1_state_in_anchor_arrays[episode_start_index: episode_end_index]


        obj0_to_obj1_cur_state_next_array = obj0_to_obj1_state_next_arrays[episode_start_index: episode_end_index]
        obj0_cur_state_next_in_anchor_array = obj0_state_next_in_anchor_arrays[episode_start_index: episode_end_index]

        obj1_to_obj0_cur_state_next_array = obj1_to_obj0_state_next_arrays[episode_start_index: episode_end_index]
        obj1_cur_state_next_in_anchor_array = obj1_state_next_in_anchor_arrays[episode_start_index: episode_end_index]


        # obj0_to_obj1_cur_action_array = obj0_to_obj1_action_arrays[episode_start_index: episode_end_index]
        # obj1_to_obj0_cur_action_array = obj1_to_obj0_action_arrays[episode_start_index: episode_end_index]

        obj0_anchor_cur_action_array = obj0_anchor_action_arrays[episode_start_index: episode_end_index]
        obj1_anchor_cur_action_array = obj1_anchor_action_arrays[episode_start_index: episode_end_index]


        cur_progress_array = progress_arrays[episode_start_index: episode_end_index]
        cur_progress_binary_array = progress_binary_arrays[episode_start_index: episode_end_index]
        cur_variation_array = variation_arrays[episode_start_index: episode_end_index]




        # obj0_to_obj1_action_max = np.max(obj0_to_obj1_cur_action_array[:, :3], axis=0)
        # obj0_to_obj1_action_min = np.min(obj0_to_obj1_cur_action_array[:, :3], axis=0)
        # obj0_to_obj1_action_mean = np.mean(obj0_to_obj1_cur_action_array[:, :3], axis=0)
        # obj0_to_obj1_action_range = obj0_to_obj1_action_max - obj0_to_obj1_action_min

        # obj1_to_obj0_action_max = np.max(obj1_to_obj0_cur_action_array[:, :3], axis=0)
        # obj1_to_obj0_action_min = np.min(obj1_to_obj0_cur_action_array[:, :3], axis=0)
        # obj1_to_obj0_action_mean = np.mean(obj1_to_obj0_cur_action_array[:, :3], axis=0)
        # obj1_to_obj0_action_range = obj1_to_obj0_action_max - obj1_to_obj0_action_min

        obj0_action_max = np.max(obj0_anchor_cur_action_array[:, :3], axis=0)
        obj0_action_min = np.min(obj0_anchor_cur_action_array[:, :3], axis=0)
        obj0_action_mean = np.mean(obj0_anchor_cur_action_array[:, :3], axis=0)
        obj0_action_range = obj0_action_max - obj0_action_min

        obj1_action_max = np.max(obj1_anchor_cur_action_array[:, :3], axis=0)
        obj1_action_min = np.min(obj1_anchor_cur_action_array[:, :3], axis=0)
        obj1_action_mean = np.mean(obj1_anchor_cur_action_array[:, :3], axis=0)
        obj1_action_range = obj1_action_max - obj1_action_min


        obj0_to_obj1_state_max = np.max(obj0_to_obj1_cur_state_array[:, :3], axis=0)
        obj0_to_obj1_state_min = np.min(obj0_to_obj1_cur_state_array[:, :3], axis=0)
        obj0_to_obj1_state_mean = np.mean(obj0_to_obj1_cur_state_array[:, :3], axis=0)
        obj0_to_obj1_state_range = obj0_to_obj1_state_max - obj0_to_obj1_state_min

        obj1_to_obj0_state_max = np.max(obj1_to_obj0_cur_state_array[:, :3], axis=0)
        obj1_to_obj0_state_min = np.min(obj1_to_obj0_cur_state_array[:, :3], axis=0)
        obj1_to_obj0_state_mean = np.mean(obj1_to_obj0_cur_state_array[:, :3], axis=0)
        obj1_to_obj0_state_range = obj1_to_obj0_state_max - obj1_to_obj0_state_min


        obj0_state_max = np.max(obj0_state_in_anchor_array[:, :3], axis=0)
        obj0_state_min = np.min(obj0_state_in_anchor_array[:, :3], axis=0)
        obj0_state_mean = np.mean(obj0_state_in_anchor_array[:, :3], axis=0)
        obj0_state_range = obj0_state_max - obj0_state_min

        obj1_state_max = np.max(obj1_state_in_anchor_array[:, :3], axis=0)
        obj1_state_min = np.min(obj1_state_in_anchor_array[:, :3], axis=0)
        obj1_state_mean = np.mean(obj1_state_in_anchor_array[:, :3], axis=0)
        obj1_state_range = obj1_state_max - obj1_state_min



        progress_max = np.max(cur_progress_array[:, :3], axis=0)
        progress_min = np.min(cur_progress_array[:, :3], axis=0)
        progress_mean = np.mean(cur_progress_array[:, :3], axis=0)
        progress_range = progress_max - progress_min

        np.set_printoptions(precision=3)
        # print(f"Episode {idx}: length {len(obj0_to_obj1_cur_action_array)}, action range {obj0_to_obj1_action_range}, progress range {progress_range}")
        # print(f"Episode {idx}: length {len(obj1_to_obj0_cur_action_array)}, action range {obj1_to_obj0_action_range}, progress range {progress_range}")
        print(f"Episode {idx}: length {len(obj0_anchor_cur_action_array)}, action range {obj0_action_range}, progress range {progress_range}")
        print(f"Episode {idx}: length {len(obj1_anchor_cur_action_array)}, action range {obj1_action_range}, progress range {progress_range}")

        # update start index for next episode
        episode_start_index = episode_end_index

    # save zarr by collecting across all demos
    # create zarr file
    cprint(f'Saved zarr file to {save_dir}', 'green')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)  # remove dir and all contains
    os.makedirs(save_dir, exist_ok=True)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save img, state, action arrays into data, and episode ends arrays into meta

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    # img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
    state_chunk_size01 = (100, obj0_to_obj1_state_arrays.shape[1])
    state_in_anchor_chunk_size0 = (100, obj0_state_in_anchor_arrays.shape[1])
    #state_in_world_chunk_size0 = (100, obj0_state_in_world_arrays.shape[1])

    state_next_chunk_size01 = (100, obj0_to_obj1_state_next_arrays.shape[1])
    state_next_in_anchor_chunk_size0 = (100, obj0_state_next_in_anchor_arrays.shape[1])
    #state_next_in_world_chunk_size0 = (100, obj0_state_next_in_world_arrays.shape[1])

    state_chunk_size10 = (100, obj1_to_obj0_state_arrays.shape[1])
    state_in_anchor_chunk_size1 = (100, obj1_state_in_anchor_arrays.shape[1])
    #state_in_world_chunk_size1 = (100, obj1_state_in_world_arrays.shape[1])

    state_next_chunk_size10 = (100, obj1_to_obj0_state_next_arrays.shape[1])
    state_next_in_anchor_chunk_size1 = (100, obj1_state_next_in_anchor_arrays.shape[1])
    #state_next_in_world_chunk_size1 = (100, obj1_state_next_in_world_arrays.shape[1])



    #world_action_chunk_size0 = (100, obj0_world_action_arrays.shape[1])
    #world_action_chunk_size1 = (100, obj1_world_action_arrays.shape[1])
    anchor_action_chunk_size0 = (100, obj0_anchor_action_arrays.shape[1])
    anchor_action_chunk_size1 = (100, obj1_anchor_action_arrays.shape[1])


    progress_chunk_size = (100, progress_arrays.shape[1])
    progress_binary_chunk_size = (100, progress_binary_arrays.shape[1])
    variation_chunk_size = (100, variation_arrays.shape[1])
    # zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)

    zarr_data.create_dataset('obj0_to_obj1_state', data=obj0_to_obj1_state_arrays, chunks=state_chunk_size01, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('obj0_state_in_anchor', data=obj0_state_in_anchor_arrays, chunks=state_in_anchor_chunk_size0, dtype='float32', overwrite=True, compressor=compressor)
    #zarr_data.create_dataset('obj0_state_in_world', data=obj0_state_in_world_arrays, chunks=state_in_world_chunk_size0, dtype='float32', overwrite=True, compressor=compressor)
   
    zarr_data.create_dataset('obj0_to_obj1_state_next', data=obj0_to_obj1_state_next_arrays, chunks=state_next_chunk_size01, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('obj0_state_next_in_anchor', data=obj0_state_next_in_anchor_arrays, chunks=state_next_in_anchor_chunk_size0, dtype='float32', overwrite=True, compressor=compressor)
    #zarr_data.create_dataset('obj0_state_next_in_world', data=obj0_state_next_in_world_arrays, chunks=state_next_in_world_chunk_size0, dtype='float32', overwrite=True, compressor=compressor)

    zarr_data.create_dataset('obj1_to_obj0_state', data=obj1_to_obj0_state_arrays, chunks=state_chunk_size10, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('obj1_state_in_anchor', data=obj1_state_in_anchor_arrays, chunks=state_in_anchor_chunk_size1, dtype='float32', overwrite=True, compressor=compressor)
    #zarr_data.create_dataset('obj1_state_in_world', data=obj1_state_in_world_arrays, chunks=state_in_world_chunk_size1, dtype='float32', overwrite=True, compressor=compressor)

    zarr_data.create_dataset('obj1_to_obj0_state_next', data=obj1_to_obj0_state_next_arrays, chunks=state_next_chunk_size10, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('obj1_state_next_in_anchor', data=obj1_state_next_in_anchor_arrays, chunks=state_next_in_anchor_chunk_size1, dtype='float32', overwrite=True, compressor=compressor)
    #zarr_data.create_dataset('obj1_state_next_in_world', data=obj1_state_next_in_world_arrays, chunks=state_next_in_world_chunk_size1, dtype='float32', overwrite=True, compressor=compressor)


    #zarr_data.create_dataset('obj0_world_action', data=obj0_world_action_arrays, chunks=world_action_chunk_size0, dtype='float32', overwrite=True, compressor=compressor)
    #zarr_data.create_dataset('obj1_world_action', data=obj1_world_action_arrays, chunks=world_action_chunk_size1, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('obj0_anchor_action', data=obj0_anchor_action_arrays, chunks=anchor_action_chunk_size0, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('obj1_anchor_action', data=obj1_anchor_action_arrays, chunks=anchor_action_chunk_size1, dtype='float32', overwrite=True, compressor=compressor)


    zarr_data.create_dataset('progress', data=progress_arrays, chunks=progress_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('progress_binary', data=progress_binary_arrays, chunks=progress_binary_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('variation', data=variation_arrays, chunks=variation_chunk_size, dtype='int64', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
