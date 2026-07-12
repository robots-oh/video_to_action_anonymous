# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from math import e
import os, shutil
import zarr
import json
import numpy as np
from termcolor import cprint
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from utils.pose_utils import get_rel_pose, euler_from_quaternion, quaternion_from_euler, relative_to_target_to_world, calculate_action, calculate_goal_pose, apply_action_to_pose, get_rel_pose2

# ---------------------------------------------------------
# Helper Functions (Must be defined before collect_narr_function)
# ---------------------------------------------------------

def check_pose_diff(pose1, pose2, trans_threshold, rot_threshold):
    action = calculate_action(pose1, pose2)
    dist = np.linalg.norm(action[:3])
    
    rot = R.from_quat(pose1[3:]) # (x, y, z, w)
    rotvec1 = rot.as_rotvec()
    rot = R.from_quat(pose2[3:]) # (x, y, z, w)
    rotvec2 = rot.as_rotvec()
    angle_diff = rotvec2 - rotvec1
    
    return dist >= trans_threshold or np.max(angle_diff) >= rot_threshold

def clean_keyframe(key_frame_idx_list, obj_anchor_pose, trans_threshold=0.3, rot_threshold=np.radians(30)):
    new_frame_idx_list = []
    if len(key_frame_idx_list) == 0: return new_frame_idx_list
    
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
            
    if len(key_frame_idx_list) > 1 and new_frame_idx_list[-1] != key_frame_idx_list[-1]:
        new_frame_idx_list.append(key_frame_idx_list[-1])

    return new_frame_idx_list

def clean_keyframe_by_dynamics(key_frame_idx_list, poses, thresholds=None, k=5):
    if len(poses) == 0: return key_frame_idx_list
    
    # --- [Settings] ---
    if thresholds is None:
        thresholds = {
            'linear_score_min': 0.02, 
            'angular_score_min': 0.01
        }

    # --- [STEP 1] Smoothing ---
    poses_arr = np.array(poses)
    
    # Quaternion sign flip check
    for i in range(1, len(poses_arr)):
        if np.dot(poses_arr[i, 3:], poses_arr[i-1, 3:]) < 0:
            poses_arr[i, 3:] *= -1

    smoothed_poses = np.zeros_like(poses_arr)
    if len(poses_arr) > 5:
        for i in range(3):
            smoothed_poses[:, i] = savgol_filter(poses_arr[:, i], window_length=5, polyorder=2)
        for i in range(3, 7):
            smoothed_poses[:, i] = savgol_filter(poses_arr[:, i], window_length=3, polyorder=2)
        quats = smoothed_poses[:, 3:]
        norm = np.linalg.norm(quats, axis=1, keepdims=True)
        # Avoid division by zero
        norm[norm == 0] = 1e-8
        quats = quats / norm
        smoothed_poses[:, 3:] = quats
    else:
        smoothed_poses = poses_arr

    # --- [STEP 2] Score Calculation ---
    n_total = len(poses)
    scores = np.zeros((n_total, 2)) # [Linear_Score, Angular_Score]

    for t in range(k, n_total - k):
        p_prev = smoothed_poses[t-k]
        p_curr = smoothed_poses[t]
        p_next = smoothed_poses[t+k]

        # Linear Score
        v_in = p_curr[:3] - p_prev[:3]
        v_out = p_next[:3] - p_curr[:3]
        
        if np.linalg.norm(v_in) > 0.005 or np.linalg.norm(v_out) > 0.005:
            scores[t, 0] = np.linalg.norm(v_out - v_in)
        else:
            scores[t, 0] = 0.0

        # Angular Score
        q_prev = R.from_quat(p_prev[3:])
        q_curr = R.from_quat(p_curr[3:])
        q_next = R.from_quat(p_next[3:])
        
        w_in = (q_curr * q_prev.inv()).as_rotvec()
        w_out = (q_next * q_curr.inv()).as_rotvec()
        
        if np.linalg.norm(w_in) > np.radians(1.0) or np.linalg.norm(w_out) > np.radians(1.0):
            scores[t, 1] = np.linalg.norm(w_out - w_in)
        else:
            scores[t, 1] = 0.0

    # --- [STEP 3] Peak Detection ---
    new_frame_idx_list = []
    if len(key_frame_idx_list) > 0:
        new_frame_idx_list.append(key_frame_idx_list[0])
    
    candidates = [idx for idx in key_frame_idx_list if k < idx < n_total - k]

    for i in range(len(candidates)):
        t = candidates[i]
        
        # Linear Check
        l_score = scores[t, 0]
        if l_score > thresholds['linear_score_min']:
            if l_score > scores[t-1, 0] and l_score > scores[t+1, 0]:
                new_frame_idx_list.append(t)
                continue 

        # Angular Check
        a_score = scores[t, 1]
        if a_score > thresholds['angular_score_min']:
            if a_score > scores[t-1, 1] and a_score > scores[t+1, 1]:
                new_frame_idx_list.append(t)

    # Add last frame
    if len(key_frame_idx_list) > 0 and key_frame_idx_list[-1] not in new_frame_idx_list:
        new_frame_idx_list.append(key_frame_idx_list[-1])

    new_frame_idx_list.sort()
    
    return new_frame_idx_list

# ---------------------------------------------------------
# Main Collection Function
# ---------------------------------------------------------


def merge_keyframes_with_priority(static_indices, dynamic_indices, min_interval=5):
    """
    Static(거리 기반)과 Dynamic(동적 기반) 키프레임을 합치되,
    Dynamic 프레임의 우선순위를 높게 둡니다.
    """
    # 1. 중복 제거 및 정렬
    dynamics_set = set(dynamic_indices)
    statics_set = set(static_indices)
    
    # 전체 후보군 (인덱스만 모음)
    all_candidates = sorted(list(dynamics_set | statics_set))
    
    if not all_candidates:
        return []

    final_indices = []

    for idx in all_candidates:
        # 첫 프레임은 무조건 추가
        if not final_indices:
            final_indices.append(idx)
            continue
        
        last_idx = final_indices[-1]
        dist = idx - last_idx
        
        # [CASE 1] 간격이 충분히 넓을 때 -> 그냥 추가
        if dist >= min_interval:
            final_indices.append(idx)
            
        # [CASE 2] 간격이 너무 좁을 때 (충돌 발생!) -> 우선순위 싸움
        else:
            is_curr_dynamic = idx in dynamics_set
            is_last_dynamic = last_idx in dynamics_set
            
            if is_curr_dynamic:
                if is_last_dynamic:
                    # 둘 다 Dynamics다? -> 사용자가 "무조건" 원했으므로 겹쳐도 그냥 추가
                    final_indices.append(idx)
                else:
                    # 현재는 Dynamics(VIP), 전꺼는 Static(일반) -> 전꺼(Static)를 제거하고 교체!
                    final_indices.pop()
                    final_indices.append(idx)
            else:
                # 현재가 Static(일반) -> 이미 자리가 찼으므로 현재꺼를 버림 (무시)
                pass
                
    # 마지막 프레임 처리 (혹시 짤렸으면 강제 추가)
    if all_candidates[-1] != final_indices[-1]:
         final_indices.append(all_candidates[-1])

    return final_indices

def collect_narr_function(poses_dict_list, episode_paths, intermediate_frame_length=1):
    print(">>> [DEBUG] Executing NEW collect_narr_function from DOLBi_new.py") # This confirms the file is updated

    arrays_sub_dict_list = []
    total_count_sub_list = []

    # Iterate over demos (poses_dict) and their corresponding paths
    # IMPORTANT: Main script must pass [episode_path] as the second argument
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

        # 1. Retrieve Data
        obj0_pose_in_anchor_list = poses_dict["grasp_anchor_obj0_pose"]
        obj1_pose_in_anchor_list = poses_dict["grasp_anchor_obj1_pose"]
        relative_pose0_list = poses_dict["grasp_obj0_pose_relative_to_grasp_obj1"]
        relative_pose1_list = poses_dict["grasp_obj1_pose_relative_to_grasp_obj0"]

        # Retrieve Original Indices (Important!)
        original_demo_indices = poses_dict.get("original_demo_index")
        
        # 2. Determine Keyframe Indices
        obj0_seq_len = len(obj0_pose_in_anchor_list)
        obj1_seq_len = len(obj1_pose_in_anchor_list)
        relative0_seq_len = len(relative_pose0_list)
        relative1_seq_len = len(relative_pose1_list)    

        obj0_idx_list = range(0, obj0_seq_len, intermediate_frame_length)
        obj1_idx_list = range(0, obj1_seq_len, intermediate_frame_length)
        rel0_idx_list = range(0, relative0_seq_len, intermediate_frame_length)
        rel1_idx_list = range(0, relative1_seq_len, intermediate_frame_length)

        # Apply Cleaning Logic
        obj0_key_idx = clean_keyframe_by_dynamics(obj0_idx_list, obj0_pose_in_anchor_list)
        obj1_key_idx = clean_keyframe_by_dynamics(obj1_idx_list, obj1_pose_in_anchor_list)
        rel0_key_idx = clean_keyframe(rel0_idx_list, relative_pose0_list)
        rel1_key_idx = clean_keyframe(rel1_idx_list, relative_pose1_list)

        # Merge Indices
        # anchor_key_frame_idx_list = sorted(list(set(obj0_key_idx) | set(obj1_key_idx) | set(rel0_key_idx) | set(rel1_key_idx)))

        # 1. 소스 분류 (Dynamics vs Static)
        # Static 소스들 (거리/각도 기반)
        dynamic_idx_list = sorted(list(set(obj0_key_idx) | set(obj1_key_idx)))

        # Dynamics 소스들 (속도/힘 기반 - VIP)
        static_idx_list = sorted(list(set(rel0_key_idx) | set(rel1_key_idx)))

        # 2. 우선순위 병합 실행
        anchor_key_frame_idx_list = merge_keyframes_with_priority(
            static_idx_list, 
            dynamic_idx_list, 
            min_interval=5  # 상황에 맞게 조절
        )

        # ==============================================================================
        # [JSON Save Block]
        # ==============================================================================
        if episode_path is not None and original_demo_indices is not None:
            try:
                # Map cleaned internal indices (k) back to original demo frame indices
                final_original_indices = [int(original_demo_indices[k]) for k in anchor_key_frame_idx_list]

                json_save_path = os.path.join(episode_path, 'keyframelist.json')
                with open(json_save_path, 'w') as f:
                    json.dump(final_original_indices, f, indent=4)
                
                print(f"[Success] Saved keyframelist.json to: {json_save_path}")
            except Exception as e:
                print(f"[Error] Failed to save JSON: {e}")
        else:
            print(f"[Warning] Skipping JSON save. Path: {episode_path}, Indices exist: {original_demo_indices is not None}")
        # ==============================================================================

        # 3. Extract Zarr Data (State/Action)
        for idx in range(len(anchor_key_frame_idx_list)-1):
            cur_frame_idx = anchor_key_frame_idx_list[idx]
            next_frame_idx = anchor_key_frame_idx_list[idx+1]

            obj0_cur = poses_dict["grasp_anchor_obj0_pose"][cur_frame_idx]
            obj1_cur = poses_dict["grasp_anchor_obj1_pose"][cur_frame_idx]
            obj0_next = poses_dict["grasp_anchor_obj0_pose"][next_frame_idx]
            obj1_next = poses_dict["grasp_anchor_obj1_pose"][next_frame_idx]

            obj0_rel_cur = poses_dict["grasp_obj0_pose_relative_to_grasp_obj1"][cur_frame_idx]
            obj0_rel_next = poses_dict["grasp_obj0_pose_relative_to_grasp_obj1"][next_frame_idx]
            obj1_rel_cur = poses_dict["grasp_obj1_pose_relative_to_grasp_obj0"][cur_frame_idx]
            obj1_rel_next = poses_dict["grasp_obj1_pose_relative_to_grasp_obj0"][next_frame_idx]
            
            # Action Calculation
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

        # 4. Final Frame & Progress (Simplified for clarity)
        # Add Final Frame Logic Here if needed (omitted for brevity, assume similar to above)
        # Using simple progress for now based on loop count
        
        # Add last frame (duplicate state, zero action) - Logic derived from original
        if len(anchor_key_frame_idx_list) > 0:
            last_idx = anchor_key_frame_idx_list[-1]
            # ... (Append last frame data similar to loop but with 0 action) ...
            # For simplicity, adhering to your structure where last frame is appended after loop
            
            # --- Saving Last Frame Data ---
            total_count_sub += 1
            # Append last state
            arrays_sub_dict["obj0_to_obj1_state"].append(poses_dict["grasp_obj0_pose_relative_to_grasp_obj1"][last_idx])
            arrays_sub_dict["obj0_state_in_anchor"].append(poses_dict["grasp_anchor_obj0_pose"][last_idx])
            arrays_sub_dict["obj1_to_obj0_state"].append(poses_dict["grasp_obj1_pose_relative_to_grasp_obj0"][last_idx])
            arrays_sub_dict["obj1_state_in_anchor"].append(poses_dict["grasp_anchor_obj1_pose"][last_idx])
            
            # Next state = current state (terminal)
            arrays_sub_dict["obj0_to_obj1_state_next"].append(poses_dict["grasp_obj0_pose_relative_to_grasp_obj1"][last_idx])
            arrays_sub_dict["obj0_state_next_in_anchor"].append(poses_dict["grasp_anchor_obj0_pose"][last_idx])
            arrays_sub_dict["obj1_to_obj0_state_next"].append(poses_dict["grasp_obj1_pose_relative_to_grasp_obj0"][last_idx])
            arrays_sub_dict["obj1_state_next_in_anchor"].append(poses_dict["grasp_anchor_obj1_pose"][last_idx])

            # Actions are 0
            zero_action = np.zeros(7) # approx (x,y,z,qx,qy,qz,qw) or similar depending on calculate_action
            # Recalculate zero action properly
            zero_act = calculate_action(poses_dict["grasp_anchor_obj0_pose"][last_idx], poses_dict["grasp_anchor_obj0_pose"][last_idx])
            
            arrays_sub_dict["obj0_to_obj1_action"].append(zero_act)
            arrays_sub_dict["obj1_to_obj0_action"].append(zero_act)
            arrays_sub_dict["obj0_anchor_action"].append(zero_act)
            arrays_sub_dict["obj1_anchor_action"].append(zero_act)

        # Progress
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
    
    # Check if data exists before stacking
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

    # debug only (printing info for first episode)
    episode_start_index = 0
    if len(episode_ends_arrays) > 0:
        idx = 0
        episode_end_index = episode_ends_arrays[0]
        obj0_anchor_cur_action_array = obj0_anchor_action_arrays[episode_start_index: episode_end_index]
        obj0_action_max = np.max(obj0_anchor_cur_action_array[:, :3], axis=0)
        obj0_action_min = np.min(obj0_anchor_cur_action_array[:, :3], axis=0)
        obj0_action_range = obj0_action_max - obj0_action_min
        print(f"Episode {idx} Action Range: {obj0_action_range}")

    # Create Zarr
    cprint(f'Saved zarr file to {save_dir}', 'green')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

    # Chunk sizes (example)
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