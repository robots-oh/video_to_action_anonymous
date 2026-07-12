# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os, shutil
import zarr
import json
import numpy as np
import math
from termcolor import cprint
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

# --- 시각화를 위한 라이브러리 설정 ---
import matplotlib
matplotlib.use('Agg') # 서버 환경(Headless) 지원 (창 안띄우고 파일로 저장)
import matplotlib.pyplot as plt
from PIL import Image
# --------------------------------

from utils.pose_utils import get_rel_pose, euler_from_quaternion, quaternion_from_euler, relative_to_target_to_world, calculate_action, calculate_goal_pose


# ---------------------------------------------------------
# Helper Functions (Check Pose, Cleaning, Smoothing)
# ---------------------------------------------------------

def check_pose_diff(pose1, pose2, trans_threshold, rot_threshold):
    action = calculate_action(pose1, pose2)
    dist = np.linalg.norm(action[:3])
    
    rot = R.from_quat(pose1[3:]) 
    rotvec1 = rot.as_rotvec()
    rot = R.from_quat(pose2[3:]) 
    rotvec2 = rot.as_rotvec()
    angle_diff = rotvec2 - rotvec1
    
    return dist >= trans_threshold or np.max(angle_diff) >= rot_threshold

def clean_keyframe(key_frame_idx_list, obj_anchor_pose, trans_threshold=0.3, rot_threshold=np.radians(30)):
    new_frame_idx_list = []
    if len(key_frame_idx_list) == 0: return new_frame_idx_list
    
    prev_frame_idx = key_frame_idx_list[0]
    new_frame_idx_list.append(key_frame_idx_list[0])

    for idx in range(1, len(key_frame_idx_list)): 
        cur_frame_idx = key_frame_idx_list[idx]
        far = check_pose_diff(
            obj_anchor_pose[prev_frame_idx], obj_anchor_pose[cur_frame_idx],
            trans_threshold=trans_threshold, rot_threshold=rot_threshold,
        )
        
        if far:
            new_frame_idx_list.append(cur_frame_idx) 
            prev_frame_idx = cur_frame_idx 
            
    if len(key_frame_idx_list) > 1 and new_frame_idx_list[-1] != key_frame_idx_list[-1]:
        new_frame_idx_list.append(key_frame_idx_list[-1])

    return new_frame_idx_list

def clean_keyframe_by_dynamics(key_frame_idx_list, poses, thresholds=None, k=5):
    if len(poses) == 0: return key_frame_idx_list
    
    if thresholds is None:
        thresholds = {
            'linear_score_min': 0.02, 
            'angular_score_min': 0.1
        }

    poses_arr = np.array(poses)
    
    for i in range(1, len(poses_arr)):
        if np.dot(poses_arr[i, 3:], poses_arr[i-1, 3:]) < 0:
            poses_arr[i, 3:] *= -1

    smoothed_poses = np.zeros_like(poses_arr)
    if len(poses_arr) > 5:
        for i in range(3):
            smoothed_poses[:, i] = savgol_filter(poses_arr[:, i], window_length=15, polyorder=2)
        for i in range(3, 7):
            smoothed_poses[:, i] = savgol_filter(poses_arr[:, i], window_length=15, polyorder=2)
        quats = smoothed_poses[:, 3:]
        norm = np.linalg.norm(quats, axis=1, keepdims=True)
        norm[norm == 0] = 1e-8
        quats = quats / norm
        smoothed_poses[:, 3:] = quats
    else:
        smoothed_poses = poses_arr

    n_total = len(poses)
    scores = np.zeros((n_total, 2))

    for t in range(k, n_total - k):
        p_prev = smoothed_poses[t-k]
        p_curr = smoothed_poses[t]
        p_next = smoothed_poses[t+k]

        v_in = p_curr[:3] - p_prev[:3]
        v_out = p_next[:3] - p_curr[:3]
        
        if np.linalg.norm(v_in) > 0.005 or np.linalg.norm(v_out) > 0.005:
            scores[t, 0] = np.linalg.norm(v_out - v_in)
        else:
            scores[t, 0] = 0.0

        q_prev = R.from_quat(p_prev[3:])
        q_curr = R.from_quat(p_curr[3:])
        q_next = R.from_quat(p_next[3:])
        
        w_in = (q_curr * q_prev.inv()).as_rotvec()
        w_out = (q_next * q_curr.inv()).as_rotvec()
        
        if np.linalg.norm(w_in) > np.radians(1.0) or np.linalg.norm(w_out) > np.radians(1.0):
            scores[t, 1] = np.linalg.norm(w_out - w_in)
        else:
            scores[t, 1] = 0.0

    new_frame_idx_list = []
    if len(key_frame_idx_list) > 0:
        new_frame_idx_list.append(key_frame_idx_list[0])
    
    candidates = [idx for idx in key_frame_idx_list if k < idx < n_total - k]

    for i in range(len(candidates)):
        t = candidates[i]
        
        l_score = scores[t, 0]
        if l_score > thresholds['linear_score_min']:
            if l_score > scores[t-1, 0] and l_score > scores[t+1, 0]:
                new_frame_idx_list.append(t)
                continue 

        a_score = scores[t, 1]
        if a_score > thresholds['angular_score_min']:
            if a_score > scores[t-1, 1] and a_score > scores[t+1, 1]:
                new_frame_idx_list.append(t)

    if len(key_frame_idx_list) > 0 and key_frame_idx_list[-1] not in new_frame_idx_list:
        new_frame_idx_list.append(key_frame_idx_list[-1])

    new_frame_idx_list.sort()
    
    return new_frame_idx_list

def merge_keyframes_with_priority(static_indices, dynamic_indices, min_interval=15):
    dynamics_set = set(dynamic_indices)
    statics_set = set(static_indices)
    all_candidates = sorted(list(dynamics_set | statics_set))
    
    if not all_candidates:
        return []

    final_indices = []

    for idx in all_candidates:
        if not final_indices:
            final_indices.append(idx)
            continue
        
        last_idx = final_indices[-1]
        dist = idx - last_idx
        
        if dist >= min_interval:
            final_indices.append(idx)
        else:
            is_curr_dynamic = idx in dynamics_set
            is_last_dynamic = last_idx in dynamics_set
            
            if is_curr_dynamic:
                if is_last_dynamic:
                    final_indices.append(idx)
                else:
                    final_indices.pop()
                    final_indices.append(idx)
            else:
                pass
                
    if all_candidates[-1] != final_indices[-1]:
         final_indices.append(all_candidates[-1])

    return final_indices


# ---------------------------------------------------------
# [수정됨] Visualization Function
# ---------------------------------------------------------
def save_keyframe_visualization(episode_path, key_indices):
    """
    episode_path: 데이터가 있는 루트 폴더 (예: .../2026-01-20--15-11-12)
    key_indices: 시각화할 프레임 번호 리스트
    """
    # 1. 박사님 데이터셋 구조에 맞춰 'rgb' 폴더 지정
    images_dir = os.path.join(episode_path, 'rgb')
    save_path = os.path.join(episode_path, 'keyframe_vis.png')

    # 폴더 확인
    if not os.path.exists(images_dir):
        print(f"[Vis Warning] 'rgb' folder not found at {images_dir}. Skipping visualization.")
        return

    num_frames = len(key_indices)
    if num_frames == 0:
        return

    cols = 5
    rows = math.ceil(num_frames / cols)
    
    # Figure 생성
    plt.figure(figsize=(4 * cols, 4 * rows))
    plt.suptitle(f"Keyframes: {key_indices}", fontsize=16)

    for i, frame_idx in enumerate(key_indices):
        ax = plt.subplot(rows, cols, i + 1)
        
        # 2. 박사님 데이터(.jpg)를 우선적으로 찾음
        img_name_jpg = f"{frame_idx}.jpg"
        img_name_png = f"{frame_idx}.png"
        
        img_path = os.path.join(images_dir, img_name_jpg)
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, img_name_png) # png로 혹시나 있으면 fallback

        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(f"Frame: {frame_idx}", fontsize=12, fontweight='bold')
            except Exception as e:
                ax.text(0.5, 0.5, "Load Error", ha='center')
        else:
            ax.text(0.5, 0.5, f"Missing\n{frame_idx}", ha='center', color='red')
            ax.set_title(f"Frame: {frame_idx}", color='red')
        
        ax.axis('off')

    plt.tight_layout()
    try:
        plt.savefig(save_path)
        plt.close()
        print(f"[Vis Success] Saved visualization to: {save_path}")
    except Exception as e:
        print(f"[Vis Error] Failed to save image: {e}")


# ---------------------------------------------------------
# Main Collection Function
# ---------------------------------------------------------

def collect_narr_function(poses_dict_list, episode_paths, intermediate_frame_length=1):

    arrays_sub_dict_list = []
    total_count_sub_list = []

    for i, (poses_dict, episode_path) in enumerate(zip(poses_dict_list, episode_paths)):

        arrays_sub_dict = {
            "obj0_to_obj1_state": [], "obj0_state_in_anchor": [],
            "obj1_to_obj0_state": [], "obj1_state_in_anchor": [],
            "obj0_to_obj1_state_next": [], "obj0_state_next_in_anchor": [],
            "obj1_to_obj0_state_next": [], "obj1_state_next_in_anchor": [],
            "obj0_to_obj1_action": [], "obj1_to_obj0_action": [],
            "obj0_anchor_action": [], "obj1_anchor_action": [],
            "progress": [], "progress_binary": [],
        }
        total_count_sub = 0

        obj0_pose_in_anchor_list = poses_dict["grasp_obj0_pose"]
        obj1_pose_in_anchor_list = poses_dict["grasp_obj1_pose"]
        relative_pose0_list = poses_dict["grasp_obj0_pose_relative_to_grasp_obj1"]
        relative_pose1_list = poses_dict["grasp_obj1_pose_relative_to_grasp_obj0"]

        original_demo_indices = poses_dict.get("original_demo_index")

        obj0_seq_len = len(obj0_pose_in_anchor_list)
        obj1_seq_len = len(obj1_pose_in_anchor_list)
        relative0_seq_len = len(relative_pose0_list)
        relative1_seq_len = len(relative_pose1_list) 
        
        obj0_idx_list = range(0, obj0_seq_len, intermediate_frame_length)
        obj1_idx_list = range(0, obj1_seq_len, intermediate_frame_length)
        rel0_idx_list = range(0, relative0_seq_len, intermediate_frame_length)
        rel1_idx_list = range(0, relative1_seq_len, intermediate_frame_length)

        obj0_key_idx = clean_keyframe_by_dynamics(obj0_idx_list, obj0_pose_in_anchor_list)
        obj1_key_idx = clean_keyframe_by_dynamics(obj1_idx_list, obj1_pose_in_anchor_list)
        rel0_key_idx = clean_keyframe(rel0_idx_list, relative_pose0_list)
        rel1_key_idx = clean_keyframe(rel1_idx_list, relative_pose1_list)

        dynamic_idx_list = sorted(list(set(obj0_key_idx) | set(obj1_key_idx)))
        static_idx_list = sorted(list(set(rel0_key_idx) | set(rel1_key_idx)))

        anchor_key_frame_idx_list = merge_keyframes_with_priority(
            static_idx_list, 
            dynamic_idx_list, 
            min_interval=5
        )

        if episode_path is not None and original_demo_indices is not None:
            try:
                final_original_indices = [int(original_demo_indices[k]) for k in anchor_key_frame_idx_list]

                # 1. JSON 저장
                json_save_path = os.path.join(episode_path, 'keyframelist.json')
                with open(json_save_path, 'w') as f:
                    json.dump(final_original_indices, f, indent=4)
                print(f"[Success] Saved keyframelist.json to: {json_save_path}")

                # 2. [수정됨] 시각화 이미지 저장 호출 (rgb 폴더 자동 탐색)
                save_keyframe_visualization(episode_path, final_original_indices)

            except Exception as e:
                print(f"[Error] Failed to save JSON or Visualization: {e}")
        else:
            print(f"[Warning] Skipping JSON save. Path: {episode_path}, Indices exist: {original_demo_indices is not None}")


        for idx in range(len(anchor_key_frame_idx_list)-1):
            cur_frame_idx = anchor_key_frame_idx_list[idx]
            next_frame_idx = anchor_key_frame_idx_list[idx+1]

            obj0_cur = poses_dict["grasp_obj0_pose"][cur_frame_idx]
            obj1_cur = poses_dict["grasp_obj1_pose"][cur_frame_idx]
            obj0_next = poses_dict["grasp_obj0_pose"][next_frame_idx]
            obj1_next = poses_dict["grasp_obj1_pose"][next_frame_idx]

            obj0_rel_cur = poses_dict["grasp_obj0_pose_relative_to_grasp_obj1"][cur_frame_idx]
            obj0_rel_next = poses_dict["grasp_obj0_pose_relative_to_grasp_obj1"][next_frame_idx]
            obj1_rel_cur = poses_dict["grasp_obj1_pose_relative_to_grasp_obj0"][cur_frame_idx]
            obj1_rel_next = poses_dict["grasp_obj1_pose_relative_to_grasp_obj0"][next_frame_idx]
            
            obj0_rel_act = calculate_action(obj0_rel_cur, obj0_rel_next)
            obj1_rel_act = calculate_action(obj1_rel_cur, obj1_rel_next)
            obj0_anch_act = calculate_action(obj0_cur, obj0_next)
            obj1_anch_act = calculate_action(obj1_cur, obj1_next)
            
            total_count_sub += 1

            arrays_sub_dict["obj0_to_obj1_state"].append(obj0_rel_cur)
            arrays_sub_dict["obj0_state_in_anchor"].append(obj0_cur)
            arrays_sub_dict["obj1_to_obj0_state"].append(obj1_rel_cur)
            arrays_sub_dict["obj1_state_in_anchor"].append(obj1_cur)

            arrays_sub_dict["obj0_to_obj1_state_next"].append(obj0_rel_next)
            arrays_sub_dict["obj0_state_next_in_anchor"].append(obj0_next)
            arrays_sub_dict["obj1_to_obj0_state_next"].append(obj1_rel_next)
            arrays_sub_dict["obj1_state_next_in_anchor"].append(obj1_next)

            arrays_sub_dict["obj0_to_obj1_action"].append(obj0_rel_act)
            arrays_sub_dict["obj1_to_obj0_action"].append(obj1_rel_act)
            arrays_sub_dict["obj0_anchor_action"].append(obj0_anch_act)
            arrays_sub_dict["obj1_anchor_action"].append(obj1_anch_act)

        
        # Last frame handling
        if len(anchor_key_frame_idx_list) > 0:
            last_idx = anchor_key_frame_idx_list[-1]
            total_count_sub += 1
            
            arrays_sub_dict["obj0_to_obj1_state"].append(poses_dict["grasp_obj0_pose_relative_to_grasp_obj1"][last_idx])
            arrays_sub_dict["obj0_state_in_anchor"].append(poses_dict["grasp_obj0_pose"][last_idx])
            arrays_sub_dict["obj1_to_obj0_state"].append(poses_dict["grasp_obj1_pose_relative_to_grasp_obj0"][last_idx])
            arrays_sub_dict["obj1_state_in_anchor"].append(poses_dict["grasp_obj1_pose"][last_idx])
            
            arrays_sub_dict["obj0_to_obj1_state_next"].append(poses_dict["grasp_obj0_pose_relative_to_grasp_obj1"][last_idx])
            arrays_sub_dict["obj0_state_next_in_anchor"].append(poses_dict["grasp_obj0_pose"][last_idx])
            arrays_sub_dict["obj1_to_obj0_state_next"].append(poses_dict["grasp_obj1_pose_relative_to_grasp_obj0"][last_idx])
            arrays_sub_dict["obj1_state_next_in_anchor"].append(poses_dict["grasp_obj1_pose"][last_idx])

            zero_act = calculate_action(poses_dict["grasp_obj0_pose"][last_idx], poses_dict["grasp_obj0_pose"][last_idx])
            
            arrays_sub_dict["obj0_to_obj1_action"].append(zero_act)
            arrays_sub_dict["obj1_to_obj0_action"].append(zero_act)
            arrays_sub_dict["obj0_anchor_action"].append(zero_act)
            arrays_sub_dict["obj1_anchor_action"].append(zero_act)

        if total_count_sub > 0:
            progress = np.linspace(0., 1., num=total_count_sub).reshape(-1, 1)
            arrays_sub_dict["progress"].extend(progress)
            
            progress_binary = np.zeros_like(progress)
            progress_binary[-1] = 1.
            if len(progress_binary) >= 2: progress_binary[-2] = 0.9
            if len(progress_binary) >= 3: progress_binary[-3] = 0.8
            if len(progress_binary) >= 4: progress_binary[-4] = 0.7
            if len(progress_binary) >= 5: progress_binary[-5] = 0.6
            if len(progress_binary) >= 6: progress_binary[-6] = 0.5
            arrays_sub_dict["progress_binary"].extend(progress_binary)

        arrays_sub_dict_list.append(arrays_sub_dict)
        total_count_sub_list.append(total_count_sub)

    return arrays_sub_dict_list, total_count_sub_list


def save_zarr(
        save_dir,
        obj0_to_obj1_state_arrays,
        obj0_state_in_anchor_arrays,
        obj0_to_obj1_state_next_arrays,
        obj0_state_next_in_anchor_arrays,
        obj1_to_obj0_state_arrays,
        obj1_state_in_anchor_arrays,
        obj1_to_obj0_state_next_arrays,
        obj1_state_next_in_anchor_arrays,
        obj0_anchor_action_arrays,
        obj1_anchor_action_arrays,
        progress_arrays,
        progress_binary_arrays,
        variation_arrays,
        episode_ends_arrays,
    ):
    
    if len(obj0_to_obj1_state_arrays) == 0:
        print("[Warning] No data to save in Zarr!")
        return

    obj0_to_obj1_state_arrays = np.stack(obj0_to_obj1_state_arrays, axis=0)
    obj0_state_in_anchor_arrays = np.stack(obj0_state_in_anchor_arrays, axis=0)
    obj1_to_obj0_state_arrays = np.stack(obj1_to_obj0_state_arrays, axis=0)
    obj1_state_in_anchor_arrays = np.stack(obj1_state_in_anchor_arrays, axis=0)

    obj0_to_obj1_state_next_arrays = np.stack(obj0_to_obj1_state_next_arrays, axis=0)
    obj0_state_next_in_anchor_arrays = np.stack(obj0_state_next_in_anchor_arrays, axis=0)
    obj1_to_obj0_state_next_arrays = np.stack(obj1_to_obj0_state_next_arrays, axis=0)   
    obj1_state_next_in_anchor_arrays = np.stack(obj1_state_next_in_anchor_arrays, axis=0)

    obj0_anchor_action_arrays = np.stack(obj0_anchor_action_arrays, axis=0)
    obj1_anchor_action_arrays = np.stack(obj1_anchor_action_arrays, axis=0)

    progress_arrays = np.stack(progress_arrays, axis=0)
    progress_binary_arrays = np.stack(progress_binary_arrays, axis=0)
    variation_arrays = np.stack(variation_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    episode_start_index = 0
    if len(episode_ends_arrays) > 0:
        idx = 0
        episode_end_index = episode_ends_arrays[0]
        obj0_anchor_cur_action_array = obj0_anchor_action_arrays[episode_start_index: episode_end_index]
        obj0_action_max = np.max(obj0_anchor_cur_action_array[:, :3], axis=0)
        obj0_action_min = np.min(obj0_anchor_cur_action_array[:, :3], axis=0)
        obj0_action_range = obj0_action_max - obj0_action_min
        print(f"Episode {idx} Action Range: {obj0_action_range}")

    cprint(f'Saved zarr file to {save_dir}', 'green')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    chunk_size = 100
    
    zarr_data.create_dataset('obj0_to_obj1_state', data=obj0_to_obj1_state_arrays, chunks=(chunk_size, obj0_to_obj1_state_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('obj0_state_in_anchor', data=obj0_state_in_anchor_arrays, chunks=(chunk_size, obj0_state_in_anchor_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('obj0_to_obj1_state_next', data=obj0_to_obj1_state_next_arrays, chunks=(chunk_size, obj0_to_obj1_state_next_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('obj0_state_next_in_anchor', data=obj0_state_next_in_anchor_arrays, chunks=(chunk_size, obj0_state_next_in_anchor_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)

    zarr_data.create_dataset('obj1_to_obj0_state', data=obj1_to_obj0_state_arrays, chunks=(chunk_size, obj1_to_obj0_state_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('obj1_state_in_anchor', data=obj1_state_in_anchor_arrays, chunks=(chunk_size, obj1_state_in_anchor_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('obj1_to_obj0_state_next', data=obj1_to_obj0_state_next_arrays, chunks=(chunk_size, obj1_to_obj0_state_next_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('obj1_state_next_in_anchor', data=obj1_state_next_in_anchor_arrays, chunks=(chunk_size, obj1_state_next_in_anchor_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)

    zarr_data.create_dataset('obj0_anchor_action', data=obj0_anchor_action_arrays, chunks=(chunk_size, obj0_anchor_action_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('obj1_anchor_action', data=obj1_anchor_action_arrays, chunks=(chunk_size, obj1_anchor_action_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)

    zarr_data.create_dataset('progress', data=progress_arrays, chunks=(chunk_size, progress_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('progress_binary', data=progress_binary_arrays, chunks=(chunk_size, progress_binary_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('variation', data=variation_arrays, chunks=(chunk_size, variation_arrays.shape[1]), dtype='int64', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)