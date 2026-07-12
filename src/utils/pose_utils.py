# # Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# #
# # NVIDIA CORPORATION and its licensors retain all intellectual property
# # and proprietary rights in and to this software, related documentation
# # and any modifications thereto.  Any use, reproduction, disclosure or
# # distribution of this software and related documentation without an express
# # license agreement from NVIDIA CORPORATION is strictly prohibited.


# import os
# import sys
# sys.path.insert(0, os.getcwd())

# import numba
# import numpy as np
# import utils.transform_utils as T
# from scipy.spatial.transform import Rotation as R

# from pyquaternion import Quaternion

# def normalize(v, epsilon=1e-8):
#     """
#     벡터를 정규화합니다. 0에 가까운 벡터는 0벡터를 반환합니다.
#     """
#     norm = np.linalg.norm(v)
#     if norm < epsilon:
#         return np.zeros_like(v)
#     return v / norm

# def apply_action_to_pose(anchor_pose: np.ndarray, action: np.ndarray) -> np.ndarray:
#     """
#     'calculate_action'에서 계산된 'action'을 'anchor_pose'에 적용하여
#     새로운 월드 포즈를 계산합니다.

#     - action (T_act)은 world frame 기준의 변환 (T_goal = T_act (+) T_current)
#     - T_new = T_act (+) T_anchor 를 계산합니다.

#     여기서 (+) 연산은 다음과 같이 정의됩니다:
#       - P_new = P_anchor + P_act
#       - Q_new = Q_act * Q_anchor

#     Args:
#         anchor_pose (np.ndarray): 기준이 되는 앵커 포즈 [x, y, z, qx, qy, qz, qw]
#         action (np.ndarray): 적용할 액션 [ax, ay, az, aqx, aqy, aqz, aqw]

#     Returns:
#         np.ndarray: T_new (새로운 월드 포즈) [x, y, z, qx, qy, qz, qw]
#     """
    
#     # --- 1. 앵커 포즈(Anchor Pose) 분리 ---
#     p_anchor = anchor_pose[:3]
#     q_anchor_vec = anchor_pose[3:]  # [qx, qy, qz, qw] 포맷
    
#     # pyquaternion 라이브러리는 (w, x, y, z) 순서로 생성자에 값을 받음
#     # q_anchor_vec[3] = qw, [0]=qx, [1]=qy, [2]=qz
#     q_anchor = Quaternion(q_anchor_vec[3], q_anchor_vec[0], q_anchor_vec[1], q_anchor_vec[2])

#     # --- 2. 액션(Action) 분리 ---
#     p_action = action[:3]
#     q_action_vec = action[3:]  # [aqx, aqy, aqz, aqw] 포맷
    
#     # (w, x, y, z) 순서로 생성자에 값을 받음
#     q_action = Quaternion(q_action_vec[3], q_action_vec[0], q_action_vec[1], q_action_vec[2])

#     # --- 3. 새로운 포즈 계산 ---
    
#     # 3a. 위치 (Position)
#     # P_new = P_anchor + P_action
#     p_new = p_anchor + p_action
    
#     # 3b. 방향 (Rotation)
#     # Q_new = Q_action * Q_anchor
#     # (Q_goal = Q_action * Q_current 이었으므로, 
#     #  Q_new = Q_action * Q_anchor 가 맞습니다.)
#     q_new = q_action * q_anchor

#     # --- 4. 결과 포즈 결합 ---
    
#     # q_new.elements는 [w, x, y, z] 순서로 쿼터니언 값을 반환함
#     q_new_elements = q_new.elements
    
#     # [x, y, z, w] 포맷으로 다시 맞춰주기
#     # q_new_elements[1]=qx, [2]=qy, [3]=qz, [0]=qw
#     q_new_vec_output = [q_new_elements[1], q_new_elements[2], q_new_elements[3], q_new_elements[0]]
    
#     # [x, y, z, qx, qy, qz, qw] 형태로 최종 결합
#     new_world_pose = np.concatenate([p_new, q_new_vec_output])
    
#     return new_world_pose

# def get_average_pose(pose0: np.ndarray, pose1: np.ndarray) -> np.ndarray:
#     """
#     두 포즈(pose0, pose1)의 관계를 기반으로 
#     새로운 '중간 작업 좌표계' 포즈(T_mid)를 계산합니다.
#     (이미지에 설명된 알고리즘 기반)

#     1. 위치: P1과 P2의 산술 평균
#     2. X축 (Primary): P1 -> P2를 향하는 방향
#     3. Z축 (Secondary): 두 포즈의 '상단(up)' 벡터의 평균 방향 (X축과 직교하도록 보정)
#     4. Y축 (Tertiary): X, Z에 모두 수직인 방향 (오른손 좌표계)

#     Args:
#         pose0 (np.ndarray): 첫 번째 포즈 [x, y, z, qx, qy, qz, qw]
#         pose1 (np.ndarray): 두 번째 포즈 [x, y, z, qx, qy, qz, qw]

#     Returns:
#         np.ndarray: 중간 작업 좌표계 포즈 [x, y, z, qx, qy, qz, qw]
#     """
    
#     # --- 0. 데이터 분리 ---
#     pos0 = pose0[:3]
#     quat0 = pose0[3:]
#     pos1 = pose1[:3]
#     quat1 = pose1[3:]

#     try:
#         rot0 = R.from_quat(quat0)
#         rot1 = R.from_quat(quat1)
#     except ValueError as e:
#         raise ValueError(f"입력된 쿼터니언이 유효하지 않습니다: {e}")

#     # --- 1. 중간 위치 계산 ---
#     # 위치는 단순 산술 평균을 냅니다.
#     mid_position = (pos0 + pos1) / 2.0

#     # --- 2. 회전(Rotation) 계산 ---
    
#     # 2a. X축 (Primary Axis)
#     # P1에서 P2를 향하는 벡터
#     x_axis = normalize(pos1 - pos0)
    
#     # Edge Case: 만약 pos0과 pos1이 같은 위치라면 X축 정의가 불가능합니다.
#     # 이 경우, 임의로 월드 X축을 사용합니다.
#     if np.linalg.norm(x_axis) < 1e-8:
#         x_axis = np.array([1.0, 0.0, 0.0])

#     # 2b. Z축 (Secondary Axis) - V_up 계산
#     # 각 로컬 Z축 (0, 0, 1)을 월드 좌표계로 변환
#     z_vec_local = np.array([0.0, 0.0, 1.0])
#     v_z1 = rot0.apply(z_vec_local)
#     v_z2 = rot1.apply(z_vec_local)
    
#     # 두 'up' 벡터의 평균 (V_up)
#     v_up = normalize(v_z1 + v_z2)
    
#     # Edge Case: 두 Z축이 정확히 반대 방향 (e.g., [0,0,1]과 [0,0,-1])이라
#     # 합이 0이 되는 경우입니다. '공통 상단'이 불명확합니다.
#     # 이 경우, 임의로 월드 Z축을 'up' 벡터로 사용합니다.
#     if np.linalg.norm(v_up) < 1e-8:
#         v_up = np.array([0.0, 0.0, 1.0])

#     # 2c. Z축 - 그-램-슈미트 직교화
#     # v_up에서 x_axis 방향 성분을 제거하여 x_axis와 수직인 z_axis를 만듭니다.
#     z_axis = normalize(v_up - np.dot(v_up, x_axis) * x_axis)

#     # Edge Case: v_up과 x_axis가 평행(collinear)한 경우 (e.g., X축이 [0,0,1]).
#     # 이 경우 그-램-슈미트 결과 z_axis가 0이 됩니다.
#     if np.linalg.norm(z_axis) < 1e-8:
#         # X축과 평행하지 않은 다른 임의의 벡터(e.g., 월드 Y축)를 사용해 Y축을 먼저 만듭니다.
#         temp_vec = np.array([0.0, 1.0, 0.0])
#         # 만약 X축이 월드 Y축과도 평행하다면, 월드 X축을 사용합니다.
#         if np.linalg.norm(np.cross(x_axis, temp_vec)) < 1e-8:
#             temp_vec = np.array([1.0, 0.0, 0.0])
        
#         # Y = Z(임시) x X
#         y_axis = normalize(np.cross(temp_vec, x_axis))
#         # Z = X x Y
#         z_axis = normalize(np.cross(x_axis, y_axis))
        
#     # 2d. Y축 (Tertiary Axis)
#     # 오른손 좌표계를 만족하도록 Y축을 계산합니다 (Y = Z x X)
#     y_axis = normalize(np.cross(z_axis, x_axis))

#     # 2e. (선택적) 모든 축 재-직교화
#     # Y축 계산 후, Z축을 다시 계산하여 수치적 오차를 줄이고
#     # 완벽한 직교 프레임을 보장합니다.
#     z_axis_final = normalize(np.cross(x_axis, y_axis))
    
#     # --- 3. 회전 행렬 생성 및 쿼터니언 변환 ---
#     # 각 축을 열(column) 벡터로 하는 회전 행렬 생성
#     # R = [X | Y | Z]
#     rotation_matrix = np.column_stack([x_axis, y_axis, z_axis_final])
    
#     # 회전 행렬을 쿼터니언으로 변환
#     average_rotation = R.from_matrix(rotation_matrix)
#     average_quat = average_rotation.as_quat() # [x, y, z, w]

#     # --- 4. 포즈 결합 ---
#     average_pose = np.concatenate([mid_position, average_quat])
    
#     return average_pose


# def pose_euler_to_pose_quaternion(pose_euler):
#     pose_quaternion = np.zeros(6)
#     pose_quaternion[:3] = pose_euler[:3]
#     pose_quaternion[3:] = quaternion_from_euler(*pose_euler[3:])
#     return pose_quaternion

# def pose_quaternion_to_pose_euler(pose_quaternion):
#     pose_euler = np.zeros(6)
#     pose_euler[:3] = pose_quaternion[:3]
#     pose_euler[3:] = euler_from_quaternion(*pose_quaternion[3:])
#     return pose_euler

# # ref: https://github.com/stepjam/RLBench/blob/master/rlbench/action_modes/arm_action_modes.py#L30
# def calculate_goal_pose(current_pose: np.ndarray, action: np.ndarray):
#     a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = action
#     x, y, z, qx, qy, qz, qw = current_pose

#     Qp = Quaternion(a_qw, a_qx, a_qy, a_qz)
#     Qch = Quaternion(qw, qx, qy, qz)
#     QW = Qp * Qch

#     # new_rot = Quaternion(
#     #     a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
#     qw, qx, qy, qz = list(QW)
#     pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
#     return np.array(pose)

# # ref: https://math.stackexchange.com/questions/2124361/quaternions-multiplication-order-to-rotate-unrotate
# def calculate_action(current_pose: np.ndarray, goal_pose: np.ndarray):
#     x1, y1, z1, qx1, qy1, qz1, qw1 = current_pose
#     x2, y2, z2, qx2, qy2, qz2, qw2 = goal_pose
    
#     #QW == Qp * Qch
#     QW = Quaternion(qw2, qx2, qy2, qz2)
#     Qch = Quaternion(qw1, qx1, qy1, qz1)
#     # Qp == QW * Qch.Inversed
#     Qp = QW * Qch.inverse

#     # QW = R.from_quat([qx2, qy2, qz2, qw2])
#     # Qch = R.from_quat([qx1, qy1, qz1, qw1])
#     # Qp = QW * Qch.inv
#     # Qp = Qp.as_quat()

#     a_qw, a_qx, a_qy, a_qz = list(Qp)
#     a_x, a_y, a_z = x2-x1, y2-y1, z2-z1

#     action = [a_x, a_y, a_z] + [a_qx, a_qy, a_qz, a_qw]
#     return np.array(action)

# def compute_rel_transform(A_pos, A_mat, B_pos, B_mat):
#     T_WA = np.vstack((np.hstack((A_mat, A_pos[:, None])), [0, 0, 0, 1]))
#     T_WB = np.vstack((np.hstack((B_mat, B_pos[:, None])), [0, 0, 0, 1]))

#     T_AB = np.matmul(np.linalg.inv(T_WA), T_WB)

#     return T_AB[:3, 3], T_AB[:3, :3]

# # target_obj_pose, grasp_obj_pose
# def get_rel_pose(pose1, pose2):
#     pos1 = np.array(pose1[:3])
#     quat1 = np.array(pose1[3:])
#     mat1 = T.quat2mat(quat1)
#     pos2 = np.array(pose2[:3])
#     quat2 = np.array(pose2[3:])
#     mat2 = T.quat2mat(quat2)

#     pos, mat = compute_rel_transform(pos1, mat1, pos2, mat2)
#     quat = T.mat2quat(mat)
#     return np.concatenate([pos, quat])



# def compute_rel_transform2(A_pos, A_mat, B_pos, B_mat):
#     T_WA = np.vstack((np.hstack((A_mat, A_pos[:, None])), [0, 0, 0, 1]))
#     T_A0 = np.vstack((np.hstack((B_mat, B_pos[:, None])), [0, 0, 0, 1]))

#     T_W0 = np.matmul(T_WA, T_A0)

#     return T_W0[:3, 3], T_W0[:3, :3]

# # target_obj_pose, grasp_obj_pose
# def get_rel_pose2(pose1, pose2):
#     pos1 = np.array(pose1[:3])
#     quat1 = np.array(pose1[3:])
#     mat1 = T.quat2mat(quat1)
#     pos2 = np.array(pose2[:3])
#     quat2 = np.array(pose2[3:])
#     mat2 = T.quat2mat(quat2)

#     pos, mat = compute_rel_transform2(pos1, mat1, pos2, mat2)
#     quat = T.mat2quat(mat)
#     return np.concatenate([pos, quat])


# # def realtive_to_target_to_world(env, obj_pose_relative_to_cab, cabinet):
# def relative_to_target_to_world(subgoal_relative_to_target, target_obj_pose):

#     # 이거는 결국 취해야하는 action을 obj to gripper 에 곱해줌으로써 
#     pos1 = subgoal_relative_to_target[:3]
#     mat1 = T.quat2mat(subgoal_relative_to_target[3:])
#     pos2 = target_obj_pose[:3]
#     mat2 = T.quat2mat(target_obj_pose[3:])

#     # T_WA = T_WB @ T_BA
#     T_BA = np.vstack((np.hstack((mat1, pos1[:, None])), [0, 0, 0, 1]))
#     T_WB = np.vstack((np.hstack((mat2, pos2[:, None])), [0, 0, 0, 1]))
#     T_WA = np.matmul(T_WB, T_BA)

#     pos = T_WA[:3, 3]
#     mat = T_WA[:3, :3]
#     quat = T.mat2quat(mat)
#     return np.concatenate([pos, quat])

# def euler_from_quaternion(x, y, z, w):
#     # import math
#     # t0 = +2.0 * (w * x + y * z)
#     # t1 = +1.0 - 2.0 * (x * x + y * y)
#     # roll_x = math.atan2(t0, t1)

#     # t2 = +2.0 * (w * y - z * x)
#     # t2 = +1.0 if t2 > +1.0 else t2
#     # t2 = -1.0 if t2 < -1.0 else t2
#     # pitch_y = math.asin(t2)

#     # t3 = +2.0 * (w * z + x * y)
#     # t4 = +1.0 - 2.0 * (y * y + z * z)
#     # yaw_z = math.atan2(t3, t4)

#     # return roll_x, pitch_y, yaw_z

#     rot = R.from_quat([x, y, z, w]) # (x, y, z, w)
#     euler = rot.as_euler('xyz',degrees=True)
#     return euler

# # def matrix_from_quaternion(x, y, z, w):
# #     rot = R.from_quat([x, y, z, w]) # (x, y, z, w)
# #     mat = rot.as_matrix()
# #     return mat

# def quaternion_from_euler(euler):
#     rot = R.from_euler('xyz', euler) 
#     quat = rot.as_quat()    # (x, y, z, w)
#     return quat


# def rodrigues(r, calculate_jacobian=True):
#     """Computes the Rodrigues transform and its derivative

#     :param r: either a 3-vector representing the rotation parameter, or a full rotation matrix
#     :param calculate_jacobian: indicates if the Jacobian of the transform is also required
#     :returns: If `calculate_jacobian` is `True`, the Jacobian is given as the second element of the returned tuple.
#     """

#     r = np.array(r, dtype=np.double)
#     eps = np.finfo(np.double).eps

#     if np.all(r.shape == (3, 1)) or np.all(r.shape == (1, 3)) or np.all(r.shape == (3,)):
#         r = r.flatten()
#         theta = np.linalg.norm(r)
#         if theta < eps:
#             r_out = np.eye(3)
#             if calculate_jacobian:
#                 jac = np.zeros((3, 9))
#                 jac[0, 5] = jac[1, 6] = jac[2, 1] = -1
#                 jac[0, 7] = jac[1, 2] = jac[2, 3] = 1

#         else:
#             c = np.cos(theta)
#             s = np.sin(theta)
#             c1 = 1. - c
#             itheta = 1.0 if theta == 0.0 else 1.0 / theta
#             r *= itheta
#             I = np.eye(3)
#             rrt = np.array([r * r[0], r * r[1], r * r[2]])
#             _r_x_ = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
#             r_out = c * I + c1 * rrt + s * _r_x_
#             if calculate_jacobian:
#                 drrt = np.array([[r[0] + r[0], r[1], r[2], r[1], 0, 0, r[2], 0, 0],
#                                  [0, r[0], 0, r[0], r[1] + r[1], r[2], 0, r[2], 0],
#                                  [0, 0, r[0], 0, 0, r[1], r[0], r[1], r[2] + r[2]]])
#                 d_r_x_ = np.array([[0, 0, 0, 0, 0, -1, 0, 1, 0],
#                                    [0, 0, 1, 0, 0, 0, -1, 0, 0],
#                                    [0, -1, 0, 1, 0, 0, 0, 0, 0]])
#                 I = np.array([I.flatten(), I.flatten(), I.flatten()])
#                 ri = np.array([[r[0]], [r[1]], [r[2]]])
#                 a0 = -s * ri
#                 a1 = (s - 2 * c1 * itheta) * ri
#                 a2 = np.ones((3, 1)) * c1 * itheta
#                 a3 = (c - s * itheta) * ri
#                 a4 = np.ones((3, 1)) * s * itheta
#                 jac = a0 * I + a1 * rrt.flatten() + a2 * drrt + a3 * _r_x_.flatten() + a4 * d_r_x_
#     elif np.all(r.shape == (3, 3)):
#         u, d, v = np.linalg.svd(r)
#         r = np.dot(u, v)
#         rx = r[2, 1] - r[1, 2]
#         ry = r[0, 2] - r[2, 0]
#         rz = r[1, 0] - r[0, 1]
#         s = np.linalg.norm(np.array([rx, ry, rz])) * np.sqrt(0.25)
#         c = np.clip((np.sum(np.diag(r)) - 1) * 0.5, -1, 1)
#         theta = np.arccos(c)
#         if s < 1e-5:
#             if c > 0:
#                 r_out = np.zeros((3, 1))
#             else:
#                 rx, ry, rz = np.clip(np.sqrt((np.diag(r) + 1) * 0.5), 0, np.inf)
#                 if r[0, 1] < 0:
#                     ry = -ry
#                 if r[0, 2] < 0:
#                     rz = -rz
#                 if np.abs(rx) < np.abs(ry) and np.abs(rx) < np.abs(rz) and ((r[1, 2] > 0) != (ry * rz > 0)):
#                     rz = -rz

#                 r_out = np.array([[rx, ry, rz]]).T
#                 theta /= np.linalg.norm(r_out)
#                 r_out *= theta
#             if calculate_jacobian:
#                 jac = np.zeros((9, 3))
#                 if c > 0:
#                     jac[1, 2] = jac[5, 0] = jac[6, 1] = -0.5
#                     jac[2, 1] = jac[3, 2] = jac[7, 0] = 0.5
#         else:
#             vth = 1.0 / (2.0 * s)
#             if calculate_jacobian:
#                 dtheta_dtr = -1. / s
#                 dvth_dtheta = -vth * c / s
#                 d1 = 0.5 * dvth_dtheta * dtheta_dtr
#                 d2 = 0.5 * dtheta_dtr
#                 dvardR = np.array([
#                     [0, 0, 0, 0, 0, 1, 0, -1, 0],
#                     [0, 0, -1, 0, 0, 0, 1, 0, 0],
#                     [0, 1, 0, -1, 0, 0, 0, 0, 0],
#                     [d1, 0, 0, 0, d1, 0, 0, 0, d1],
#                     [d2, 0, 0, 0, d2, 0, 0, 0, d2]])
#                 dvar2dvar = np.array([
#                     [vth, 0, 0, rx, 0],
#                     [0, vth, 0, ry, 0],
#                     [0, 0, vth, rz, 0],
#                     [0, 0, 0, 0, 1]])
#                 domegadvar2 = np.array([
#                     [theta, 0, 0, rx * vth],
#                     [0, theta, 0, ry * vth],
#                     [0, 0, theta, rz * vth]])
#                 jac = np.dot(np.dot(domegadvar2, dvar2dvar), dvardR)
#                 for ii in range(3):
#                     jac[ii] = jac[ii].reshape((3, 3)).T.flatten()
#                 jac = jac.T
#             vth *= theta
#             r_out = np.array([[rx, ry, rz]]).T * vth
#     else:
#         raise Exception("rodrigues: input matrix must be 1x3, 3x1 or 3x3.")
#     if calculate_jacobian:
#         return r_out, jac
#     else:
#         return r_out


# def rodrigues2rotmat(r):
#     # R = np.zeros((3, 3))
#     r_skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
#     theta = np.linalg.norm(r)
#     return np.identity(3) + np.sin(theta) * r_skew + (1 - np.cos(theta)) * r_skew.dot(r_skew)


# if __name__ == "__main__":
#     target_obj_pose = np.array([ 0.32260922, -0.10839751,  0.96184993,  0.57075304,  0.05685034, -0.81511486,  0.08122107])
#     grasp_obj_pose = np.array([ 1.99161470e-01,  3.34810495e-01,  7.91927218e-01, -1.26936930e-05,  2.39512883e-06,  9.92952466e-01, -1.18513443e-01])
#     # obj_pose_relative_to_target = np.array([ 0.17114308, -0.45885825, -0.02656152, -0.01118924, -0.57345784,  0.0159555,  0.81900305])

#     np.set_printoptions(precision=3)
#     obj_pose_relative_to_target = get_rel_pose(target_obj_pose, grasp_obj_pose)
#     grasp_obj_pose_2 = relative_to_target_to_world(obj_pose_relative_to_target, target_obj_pose)
#     print("grasp_obj_pose", grasp_obj_pose)
#     print("grasp_obj_pose_2", grasp_obj_pose_2)

#     obj_pose_relative_to_target = get_rel_pose(target_obj_pose, grasp_obj_pose)
#     grasp_obj_pose_2 = relative_to_target_to_world(obj_pose_relative_to_target, target_obj_pose)
#     print("grasp_obj_pose", grasp_obj_pose)
#     print("grasp_obj_pose_2", grasp_obj_pose_2)

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

import numba
import numpy as np
import utils.transform_utils as T
from scipy.spatial.transform import Rotation as R

from pyquaternion import Quaternion

def normalize(v, epsilon=1e-8):
    """
    벡터를 정규화합니다. 0에 가까운 벡터는 0벡터를 반환합니다.
    """
    norm = np.linalg.norm(v)
    if norm < epsilon:
        return np.zeros_like(v)
    return v / norm

def apply_action_to_pose(anchor_pose: np.ndarray, action: np.ndarray) -> np.ndarray:
    """
    'calculate_action'에서 계산된 'action'을 'anchor_pose'에 적용하여
    새로운 월드 포즈를 계산합니다.

    - action (T_act)은 world frame 기준의 변환 (T_goal = T_act (+) T_current)
    - T_new = T_act (+) T_anchor 를 계산합니다.

    여기서 (+) 연산은 다음과 같이 정의됩니다:
      - P_new = P_anchor + P_act
      - Q_new = Q_act * Q_anchor

    Args:
        anchor_pose (np.ndarray): 기준이 되는 앵커 포즈 [x, y, z, qx, qy, qz, qw]
        action (np.ndarray): 적용할 액션 [ax, ay, az, aqx, aqy, aqz, aqw]

    Returns:
        np.ndarray: T_new (새로운 월드 포즈) [x, y, z, qx, qy, qz, qw]
    """
    
    # --- 1. 앵커 포즈(Anchor Pose) 분리 ---
    p_anchor = anchor_pose[:3]
    q_anchor_vec = anchor_pose[3:]  # [qx, qy, qz, qw] 포맷
    
    # pyquaternion 라이브러리는 (w, x, y, z) 순서로 생성자에 값을 받음
    # q_anchor_vec[3] = qw, [0]=qx, [1]=qy, [2]=qz
    q_anchor = Quaternion(q_anchor_vec[3], q_anchor_vec[0], q_anchor_vec[1], q_anchor_vec[2])

    # --- 2. 액션(Action) 분리 ---
    p_action = action[:3]
    q_action_vec = action[3:]  # [aqx, aqy, aqz, aqw] 포맷
    
    # (w, x, y, z) 순서로 생성자에 값을 받음
    q_action = Quaternion(q_action_vec[3], q_action_vec[0], q_action_vec[1], q_action_vec[2])

    # --- 3. 새로운 포즈 계산 ---
    
    # 3a. 위치 (Position)
    # P_new = P_anchor + P_action
    p_new = p_anchor + p_action
    
    # 3b. 방향 (Rotation)
    # Q_new = Q_action * Q_anchor
    # (Q_goal = Q_action * Q_current 이었으므로, 
    #  Q_new = Q_action * Q_anchor 가 맞습니다.)
    q_new = q_action * q_anchor

    # --- 4. 결과 포즈 결합 ---
    
    # q_new.elements는 [w, x, y, z] 순서로 쿼터니언 값을 반환함
    q_new_elements = q_new.elements
    
    # [x, y, z, w] 포맷으로 다시 맞춰주기
    # q_new_elements[1]=qx, [2]=qy, [3]=qz, [0]=qw
    q_new_vec_output = [q_new_elements[1], q_new_elements[2], q_new_elements[3], q_new_elements[0]]
    
    # [x, y, z, qx, qy, qz, qw] 형태로 최종 결합
    new_world_pose = np.concatenate([p_new, q_new_vec_output])
    
    return new_world_pose



def get_average_pose(pose0: np.ndarray, pose1: np.ndarray) -> np.ndarray:
    """
    두 포즈(pose0, pose1)의 관계를 기반으로 
    새로운 '중간 작업 좌표계' 포즈(T_mid)를 계산합니다.
    (🌟 평면 투영 - Planar Projection 방식 적용 🌟)

    1. 위치: P1과 P2의 산술 평균
    2. Z축 (하늘): 두 물체의 로컬 Z축(정수리)을 평균내어 진짜 중력 반대 방향(v_up)을 찾음
    3. X축 (수평선): P1 -> P2를 향하는 벡터에서 높이(v_up) 성분을 제거하여 완벽한 수평 축 생성
    4. Y축: Z축과 X축에 수직인 방향 (오른손 좌표계)

    Args:
        pose0 (np.ndarray): 첫 번째 포즈 [x, y, z, qx, qy, qz, qw]
        pose1 (np.ndarray): 두 번째 포즈 [x, y, z, qx, qy, qz, qw]

    Returns:
        np.ndarray: 중간 작업 좌표계 포즈 [x, y, z, qx, qy, qz, qw]
    """
    
    # --- 0. 데이터 분리 ---
    pos0 = pose0[:3]
    quat0 = pose0[3:]
    pos1 = pose1[:3]
    quat1 = pose1[3:]

    try:
        rot0 = R.from_quat(quat0)
        rot1 = R.from_quat(quat1)
    except ValueError as e:
        raise ValueError(f"입력된 쿼터니언이 유효하지 않습니다: {e}")

    # --- 1. 중간 위치 계산 ---
    # 위치는 단순 산술 평균 (앵커가 허공에 떠 있어도 무관)
    mid_position = (pos0 + pos1) / 2.0

    # --- 2. 진짜 하늘(v_up) 찾기 ---
    # 각 물체의 로컬 Z축(정수리) 방향을 월드로 변환 후 평균
    z_vec_local = np.array([0.0, 0.0, 1.0])
    v_z1 = rot0.apply(z_vec_local)
    v_z2 = rot1.apply(z_vec_local)

    v_up = normalize(v_z1 + v_z2)

    if np.linalg.norm(v_up) < 1e-8:
        v_up = np.array([0.0, 0.0, 1.0])

    # --- 3. 평면 투영(Planar Projection)을 이용한 X축 계산 ---
    vec = pos1 - pos0
    
    # 🌟 핵심: vec에서 v_up(높이) 방향 성분을 빼버려서 완벽한 수평(그림자) 벡터로 만듭니다.
    vec_flat = vec - np.dot(vec, v_up) * v_up
    
    x_axis = normalize(vec_flat)
    
    # 예외 처리: 두 물체의 X, Y 위치가 완벽히 겹치는 경우
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0])

    # --- 4. Y축 및 Z축 계산 (완벽 직교 보장) ---
    # X는 수평이고 v_up은 수직이므로 둘을 외적하면 완벽한 평면 Y축이 나옵니다.
    y_axis = normalize(np.cross(v_up, x_axis))
    z_axis_final = normalize(np.cross(x_axis, y_axis)) # 사실상 v_up과 100% 동일

    # --- 5. 회전 행렬 생성 및 쿼터니언 변환 ---
    # R = [X | Y | Z]
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis_final])
    
    average_rotation = R.from_matrix(rotation_matrix)
    average_quat = average_rotation.as_quat() # [x, y, z, w]

    # --- 6. 포즈 결합 ---
    average_pose = np.concatenate([mid_position, average_quat])
    
    return average_pose


def pose_euler_to_pose_quaternion(pose_euler):
    pose_quaternion = np.zeros(6)
    pose_quaternion[:3] = pose_euler[:3]
    pose_quaternion[3:] = quaternion_from_euler(*pose_euler[3:])
    return pose_quaternion

def pose_quaternion_to_pose_euler(pose_quaternion):
    pose_euler = np.zeros(6)
    pose_euler[:3] = pose_quaternion[:3]
    pose_euler[3:] = euler_from_quaternion(*pose_quaternion[3:])
    return pose_euler

# ref: https://github.com/stepjam/RLBench/blob/master/rlbench/action_modes/arm_action_modes.py#L30
def calculate_goal_pose(current_pose: np.ndarray, action: np.ndarray):
    a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = action
    x, y, z, qx, qy, qz, qw = current_pose

    Qp = Quaternion(a_qw, a_qx, a_qy, a_qz)
    Qch = Quaternion(qw, qx, qy, qz)
    QW = Qp * Qch

    # new_rot = Quaternion(
    #     a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
    qw, qx, qy, qz = list(QW)
    pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
    return np.array(pose)

# ref: https://math.stackexchange.com/questions/2124361/quaternions-multiplication-order-to-rotate-unrotate
def calculate_action(current_pose: np.ndarray, goal_pose: np.ndarray):
    x1, y1, z1, qx1, qy1, qz1, qw1 = current_pose
    x2, y2, z2, qx2, qy2, qz2, qw2 = goal_pose
    
    #QW == Qp * Qch
    QW = Quaternion(qw2, qx2, qy2, qz2)
    Qch = Quaternion(qw1, qx1, qy1, qz1)
    # Qp == QW * Qch.Inversed
    Qp = QW * Qch.inverse

    # QW = R.from_quat([qx2, qy2, qz2, qw2])
    # Qch = R.from_quat([qx1, qy1, qz1, qw1])
    # Qp = QW * Qch.inv
    # Qp = Qp.as_quat()

    a_qw, a_qx, a_qy, a_qz = list(Qp)
    a_x, a_y, a_z = x2-x1, y2-y1, z2-z1

    action = [a_x, a_y, a_z] + [a_qx, a_qy, a_qz, a_qw]
    return np.array(action)

def compute_rel_transform(A_pos, A_mat, B_pos, B_mat):
    T_WA = np.vstack((np.hstack((A_mat, A_pos[:, None])), [0, 0, 0, 1]))
    T_WB = np.vstack((np.hstack((B_mat, B_pos[:, None])), [0, 0, 0, 1]))

    T_AB = np.matmul(np.linalg.inv(T_WA), T_WB)

    return T_AB[:3, 3], T_AB[:3, :3]

# target_obj_pose, grasp_obj_pose
def get_rel_pose(pose1, pose2):
    pos1 = np.array(pose1[:3])
    quat1 = np.array(pose1[3:])
    mat1 = T.quat2mat(quat1)
    pos2 = np.array(pose2[:3])
    quat2 = np.array(pose2[3:])
    mat2 = T.quat2mat(quat2)

    pos, mat = compute_rel_transform(pos1, mat1, pos2, mat2)
    quat = T.mat2quat(mat)
    return np.concatenate([pos, quat])



def compute_rel_transform2(A_pos, A_mat, B_pos, B_mat):
    T_WA = np.vstack((np.hstack((A_mat, A_pos[:, None])), [0, 0, 0, 1]))
    T_A0 = np.vstack((np.hstack((B_mat, B_pos[:, None])), [0, 0, 0, 1]))

    T_W0 = np.matmul(T_WA, T_A0)

    return T_W0[:3, 3], T_W0[:3, :3]

# target_obj_pose, grasp_obj_pose
def get_rel_pose2(pose1, pose2):
    pos1 = np.array(pose1[:3])
    quat1 = np.array(pose1[3:])
    mat1 = T.quat2mat(quat1)
    pos2 = np.array(pose2[:3])
    quat2 = np.array(pose2[3:])
    mat2 = T.quat2mat(quat2)

    pos, mat = compute_rel_transform2(pos1, mat1, pos2, mat2)
    quat = T.mat2quat(mat)
    return np.concatenate([pos, quat])


# def realtive_to_target_to_world(env, obj_pose_relative_to_cab, cabinet):
def relative_to_target_to_world(subgoal_relative_to_target, target_obj_pose):

    # 이거는 결국 취해야하는 action을 obj to gripper 에 곱해줌으로써 
    pos1 = subgoal_relative_to_target[:3]
    mat1 = T.quat2mat(subgoal_relative_to_target[3:])
    pos2 = target_obj_pose[:3]
    mat2 = T.quat2mat(target_obj_pose[3:])

    # T_WA = T_WB @ T_BA
    T_BA = np.vstack((np.hstack((mat1, pos1[:, None])), [0, 0, 0, 1]))
    T_WB = np.vstack((np.hstack((mat2, pos2[:, None])), [0, 0, 0, 1]))
    T_WA = np.matmul(T_WB, T_BA)

    pos = T_WA[:3, 3]
    mat = T_WA[:3, :3]
    quat = T.mat2quat(mat)
    return np.concatenate([pos, quat])

def euler_from_quaternion(x, y, z, w):
    # import math
    # t0 = +2.0 * (w * x + y * z)
    # t1 = +1.0 - 2.0 * (x * x + y * y)
    # roll_x = math.atan2(t0, t1)

    # t2 = +2.0 * (w * y - z * x)
    # t2 = +1.0 if t2 > +1.0 else t2
    # t2 = -1.0 if t2 < -1.0 else t2
    # pitch_y = math.asin(t2)

    # t3 = +2.0 * (w * z + x * y)
    # t4 = +1.0 - 2.0 * (y * y + z * z)
    # yaw_z = math.atan2(t3, t4)

    # return roll_x, pitch_y, yaw_z

    rot = R.from_quat([x, y, z, w]) # (x, y, z, w)
    euler = rot.as_euler('xyz',degrees=True)
    return euler

# def matrix_from_quaternion(x, y, z, w):
#     rot = R.from_quat([x, y, z, w]) # (x, y, z, w)
#     mat = rot.as_matrix()
#     return mat

def quaternion_from_euler(euler):
    rot = R.from_euler('xyz', euler) 
    quat = rot.as_quat()    # (x, y, z, w)
    return quat


def rodrigues(r, calculate_jacobian=True):
    """Computes the Rodrigues transform and its derivative

    :param r: either a 3-vector representing the rotation parameter, or a full rotation matrix
    :param calculate_jacobian: indicates if the Jacobian of the transform is also required
    :returns: If `calculate_jacobian` is `True`, the Jacobian is given as the second element of the returned tuple.
    """

    r = np.array(r, dtype=np.double)
    eps = np.finfo(np.double).eps

    if np.all(r.shape == (3, 1)) or np.all(r.shape == (1, 3)) or np.all(r.shape == (3,)):
        r = r.flatten()
        theta = np.linalg.norm(r)
        if theta < eps:
            r_out = np.eye(3)
            if calculate_jacobian:
                jac = np.zeros((3, 9))
                jac[0, 5] = jac[1, 6] = jac[2, 1] = -1
                jac[0, 7] = jac[1, 2] = jac[2, 3] = 1

        else:
            c = np.cos(theta)
            s = np.sin(theta)
            c1 = 1. - c
            itheta = 1.0 if theta == 0.0 else 1.0 / theta
            r *= itheta
            I = np.eye(3)
            rrt = np.array([r * r[0], r * r[1], r * r[2]])
            _r_x_ = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            r_out = c * I + c1 * rrt + s * _r_x_
            if calculate_jacobian:
                drrt = np.array([[r[0] + r[0], r[1], r[2], r[1], 0, 0, r[2], 0, 0],
                                 [0, r[0], 0, r[0], r[1] + r[1], r[2], 0, r[2], 0],
                                 [0, 0, r[0], 0, 0, r[1], r[0], r[1], r[2] + r[2]]])
                d_r_x_ = np.array([[0, 0, 0, 0, 0, -1, 0, 1, 0],
                                   [0, 0, 1, 0, 0, 0, -1, 0, 0],
                                   [0, -1, 0, 1, 0, 0, 0, 0, 0]])
                I = np.array([I.flatten(), I.flatten(), I.flatten()])
                ri = np.array([[r[0]], [r[1]], [r[2]]])
                a0 = -s * ri
                a1 = (s - 2 * c1 * itheta) * ri
                a2 = np.ones((3, 1)) * c1 * itheta
                a3 = (c - s * itheta) * ri
                a4 = np.ones((3, 1)) * s * itheta
                jac = a0 * I + a1 * rrt.flatten() + a2 * drrt + a3 * _r_x_.flatten() + a4 * d_r_x_
    elif np.all(r.shape == (3, 3)):
        u, d, v = np.linalg.svd(r)
        r = np.dot(u, v)
        rx = r[2, 1] - r[1, 2]
        ry = r[0, 2] - r[2, 0]
        rz = r[1, 0] - r[0, 1]
        s = np.linalg.norm(np.array([rx, ry, rz])) * np.sqrt(0.25)
        c = np.clip((np.sum(np.diag(r)) - 1) * 0.5, -1, 1)
        theta = np.arccos(c)
        if s < 1e-5:
            if c > 0:
                r_out = np.zeros((3, 1))
            else:
                rx, ry, rz = np.clip(np.sqrt((np.diag(r) + 1) * 0.5), 0, np.inf)
                if r[0, 1] < 0:
                    ry = -ry
                if r[0, 2] < 0:
                    rz = -rz
                if np.abs(rx) < np.abs(ry) and np.abs(rx) < np.abs(rz) and ((r[1, 2] > 0) != (ry * rz > 0)):
                    rz = -rz

                r_out = np.array([[rx, ry, rz]]).T
                theta /= np.linalg.norm(r_out)
                r_out *= theta
            if calculate_jacobian:
                jac = np.zeros((9, 3))
                if c > 0:
                    jac[1, 2] = jac[5, 0] = jac[6, 1] = -0.5
                    jac[2, 1] = jac[3, 2] = jac[7, 0] = 0.5
        else:
            vth = 1.0 / (2.0 * s)
            if calculate_jacobian:
                dtheta_dtr = -1. / s
                dvth_dtheta = -vth * c / s
                d1 = 0.5 * dvth_dtheta * dtheta_dtr
                d2 = 0.5 * dtheta_dtr
                dvardR = np.array([
                    [0, 0, 0, 0, 0, 1, 0, -1, 0],
                    [0, 0, -1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, -1, 0, 0, 0, 0, 0],
                    [d1, 0, 0, 0, d1, 0, 0, 0, d1],
                    [d2, 0, 0, 0, d2, 0, 0, 0, d2]])
                dvar2dvar = np.array([
                    [vth, 0, 0, rx, 0],
                    [0, vth, 0, ry, 0],
                    [0, 0, vth, rz, 0],
                    [0, 0, 0, 0, 1]])
                domegadvar2 = np.array([
                    [theta, 0, 0, rx * vth],
                    [0, theta, 0, ry * vth],
                    [0, 0, theta, rz * vth]])
                jac = np.dot(np.dot(domegadvar2, dvar2dvar), dvardR)
                for ii in range(3):
                    jac[ii] = jac[ii].reshape((3, 3)).T.flatten()
                jac = jac.T
            vth *= theta
            r_out = np.array([[rx, ry, rz]]).T * vth
    else:
        raise Exception("rodrigues: input matrix must be 1x3, 3x1 or 3x3.")
    if calculate_jacobian:
        return r_out, jac
    else:
        return r_out


def rodrigues2rotmat(r):
    # R = np.zeros((3, 3))
    r_skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    theta = np.linalg.norm(r)
    return np.identity(3) + np.sin(theta) * r_skew + (1 - np.cos(theta)) * r_skew.dot(r_skew)


if __name__ == "__main__":
    target_obj_pose = np.array([ 0.32260922, -0.10839751,  0.96184993,  0.57075304,  0.05685034, -0.81511486,  0.08122107])
    grasp_obj_pose = np.array([ 1.99161470e-01,  3.34810495e-01,  7.91927218e-01, -1.26936930e-05,  2.39512883e-06,  9.92952466e-01, -1.18513443e-01])
    # obj_pose_relative_to_target = np.array([ 0.17114308, -0.45885825, -0.02656152, -0.01118924, -0.57345784,  0.0159555,  0.81900305])

    np.set_printoptions(precision=3)
    obj_pose_relative_to_target = get_rel_pose(target_obj_pose, grasp_obj_pose)
    grasp_obj_pose_2 = relative_to_target_to_world(obj_pose_relative_to_target, target_obj_pose)
    print("grasp_obj_pose", grasp_obj_pose)
    print("grasp_obj_pose_2", grasp_obj_pose_2)

    obj_pose_relative_to_target = get_rel_pose(target_obj_pose, grasp_obj_pose)
    grasp_obj_pose_2 = relative_to_target_to_world(obj_pose_relative_to_target, target_obj_pose)
    print("grasp_obj_pose", grasp_obj_pose)
    print("grasp_obj_pose_2", grasp_obj_pose_2)