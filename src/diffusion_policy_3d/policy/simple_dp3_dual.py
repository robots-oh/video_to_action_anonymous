from typing import Dict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
from typing import Optional, Dict, Tuple, Union, List, Type

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d_progress import ConditionalUnet1D_progress
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from utils.vis_utils import dp3_visualize
from diffusion_policy_3d.model.vision.dual_extractor import DualDP3Encoder


def custom_normalize(data, normalizer, key):
    pose = copy.deepcopy(data) # (B, T, 7)

    if key == 'obj0_anchor_pos':
        del pose['obj1_anchor_pos']
        del pose['obj1_image']
    elif key == 'obj1_anchor_pos':
        del pose['obj0_anchor_pos']
        del pose['obj0_image']

    if key is not None:
        use_progress = True if pose[key].shape[-1] > 7 else False
    else:
        use_progress = True if pose.shape[-1] > 7 else False

    if key is not None:
        device = pose[key].device
        # save and normalize quaternion
        if use_progress:
            progress = pose[key][:, :, 7:8]
        pose_ori = pose[key][:, :, 3:7] 
        # print("pose[key]", pose[key].shape)
        # remove quaternion
        pose[key] = pose[key][:, :, :3]
    else:
        device = pose.device
        # save and normalize quaternion
        if use_progress:
            progress = pose[:, :, 7:8] 
        pose_ori = pose[:, :, 3:7] 
        # print("pose", pose.shape)
        # remove quaternion
        pose = pose[:, :, :3]

    # normalize data and quaternion
    pose_ori = torch.nn.functional.normalize(pose_ori, dim=2)
    npose = normalizer.normalize(pose)

    if key is not None:
        pose_ori = pose_ori.to(npose[key].device)
        # add the normalize quaternion back
        if use_progress:
            progress = progress.to(npose[key].device)
            npose[key] = torch.concat([npose[key], pose_ori, progress], dim=-1)
        else:
            npose[key] = torch.concat([npose[key], pose_ori], dim=-1)
        # print("data[key]", data[key].shape)
        # print("npose[key]", npose[key].shape)
    else:
        pose_ori = pose_ori.to(npose.device)
        if use_progress:
            progress = progress.to(npose.device)
            npose = torch.concat([npose, pose_ori, progress], dim=-1)
        else:
            npose = torch.concat([npose, pose_ori], dim=-1)
        # print("data", data.shape)
        # print("npose", npose.shape)
    return npose


def custom_unnormalize(data, normalizer, key):
    pose = copy.deepcopy(data) # (B, T, 7)
    if key is not None:
        use_progress = True if pose[key].shape[-1] > 7 else False
    else:
        use_progress = True if pose.shape[-1] > 7 else False

    if key is not None:
        # save and normalize quaternion
        if use_progress:
            progress = pose[key][:, :, 7:8]
        pose_ori = pose[key][:, :, 3:7] 
        # print("pose[key]", pose[key].shape)
        # remove quaternion
        pose[key] = pose[key][:, :, :3]
    else:
        # save and normalize quaternion
        if use_progress:
            progress = pose[:, :, 7:8] 
        pose_ori = pose[:, :, 3:7] 
        # print("pose", pose.shape)
        # remove quaternion
        pose = pose[:, :, :3]

    # normalize data and quaternion
    pose_ori = torch.nn.functional.normalize(pose_ori, dim=2)
    npose = normalizer.unnormalize(pose)

    if key is not None:
        # add the normalize quaternion back
        if use_progress:
            npose[key] = torch.concat([npose[key], pose_ori, progress], dim=-1)
        else:
            npose[key] = torch.concat([npose[key], pose_ori], dim=-1)
        # print("npose[key]", npose[key].shape)
    else:
        if use_progress:
            npose = torch.concat([npose, pose_ori, progress], dim=-1)
        else:
            npose = torch.concat([npose, pose_ori], dim=-1)
        # print("npose", npose.shape)
    return npose
# def custom_normalize(data, normalizer, key):
#     pose = copy.deepcopy(data) # (B, T, D)

#     # 1. 불필요한 이미지 키 등 제거
#     if key == 'obj0_anchor_pos':
#         if isinstance(pose, dict):
#             pose.pop('obj1_anchor_pos', None)
#             pose.pop('obj1_image', None)
#     elif key == 'obj1_anchor_pos':
#         if isinstance(pose, dict):
#             pose.pop('obj0_anchor_pos', None)
#             pose.pop('obj0_image', None)

#     # 2. 데이터(Tensor) 추출
#     target_tensor = pose[key] if key is not None else pose
    
#     # 3. Progress 분리 로직
#     use_progress = target_tensor.shape[-1] > 7
    
#     if use_progress:
#         pose_7d = target_tensor[..., :7]
#         progress = target_tensor[..., 7:]
#     else:
#         pose_7d = target_tensor
#         progress = None

#     # 4. 정규화 수행 [수정된 부분]
#     # key가 있으면: 전체 Normalizer에서 해당 key의 전용 계산기를 꺼내서 씁니다.
#     # key가 없으면(None): 이미 전용 계산기가 들어온 것이므로 그대로 씁니다.
#     if key is not None:
#         npose_7d = normalizer[key].normalize(pose_7d)
#     else:
#         npose_7d = normalizer.normalize(pose_7d)

#     # 5. 다시 합치기 (Concatenate)
#     if use_progress:
#         npose_final = torch.cat([npose_7d, progress], dim=-1)
#     else:
#         npose_final = npose_7d

#     # 6. 결과 반환
#     if key is not None:
#         pose[key] = npose_final
#         return pose
#     else:
#         return npose_final


# def custom_unnormalize(data, normalizer, key):
#     pose = copy.deepcopy(data)
    
#     # 1. 데이터 추출
#     target_tensor = pose[key] if key is not None else pose
    
#     # 2. Progress 분리
#     use_progress = target_tensor.shape[-1] > 7
    
#     if use_progress:
#         npose_7d = target_tensor[..., :7]
#         progress = target_tensor[..., 7:]
#     else:
#         npose_7d = target_tensor
#         progress = None

#     # 3. 역정규화 (Unnormalize) [수정된 부분]
#     # 여기서도 key 유무에 따라 처리 방식을 나눕니다.
#     # (보통 unnormalize 할 때는 이미 전용 normalizer가 들어오긴 하지만, 안전하게 처리)
#     if key is not None:
#         # 혹시 전체 Normalizer가 들어왔을 경우를 대비
#         if hasattr(normalizer, 'params_dict') and key in normalizer.params_dict:
#              pose_7d = normalizer[key].unnormalize(npose_7d)
#         else:
#              pose_7d = normalizer.unnormalize(npose_7d)
#     else:
#         pose_7d = normalizer.unnormalize(npose_7d)

#     # 4. 쿼터니언 유효성 보정 (Geometric Normalization)
#     pos = pose_7d[..., :3]
#     rot = pose_7d[..., 3:]
#     rot = torch.nn.functional.normalize(rot, dim=-1) 
#     pose_7d = torch.cat([pos, rot], dim=-1)

#     # 5. 합치기
#     if use_progress:
#         pose_final = torch.cat([pose_7d, progress], dim=-1)
#     else:
#         pose_final = pose_7d

#     # 6. 반환
#     if key is not None:
#         pose[key] = pose_final
#         return pose
#     else:
#         return pose_final


from scipy.spatial.transform import Rotation as R
def compute_rel_transform_batch(A_pos, A_mat, B_pos, B_mat):
        batch_size = A_pos.shape[0]
        
        # (N, 4, 4) 크기의 동차 변환 행렬 배치를 생성합니다.
        T_WA = np.zeros((batch_size, 4, 4))
        T_WA[:, :3, :3] = A_mat
        T_WA[:, :3, 3] = A_pos
        T_WA[:, 3, 3] = 1.0
        
        T_WB = np.zeros((batch_size, 4, 4))
        T_WB[:, :3, :3] = B_mat
        T_WB[:, :3, 3] = B_pos
        T_WB[:, 3, 3] = 1.0

        # T_AB = inv(T_WA) @ T_WB
        T_WA_inv = np.linalg.inv(T_WA)
        T_AB = np.matmul(T_WA_inv, T_WB)

        # (N, 3) 위치와 (N, 3, 3) 회전 행렬을 반환합니다.
        return T_AB[:, :3, 3], T_AB[:, :3, :3]

def get_rel_pose_batch_horizon(pose1, pose2):
    """
    (B, H, 7) 크기의 포즈(pos, quat) 배치를 받아
    (B, H, 7) 크기의 상대 포즈 배치를 반환합니다.

    :param pose1: (B, H, 7) array (pos_x, y, z, quat_x, y, z, w)
    :param pose2: (B, H, 7) array (pos_x, y, z, quat_x, y, z, w)
    :return: (B, H, 7) array of relative poses
    """
    # 1. 입력 Shape 저장 및 (B*H, 7)로 Reshape
    batch_size, horizon = pose1.shape[0], pose1.shape[1]
    total_elements = batch_size * horizon
    
    pose1_flat = pose1.reshape(total_elements, 7)
    pose2_flat = pose2.reshape(total_elements, 7)

    # 2. (B*H, 7) -> (B*H, 3) and (B*H, 4)로 분리
    pos1_flat = pose1_flat[:, :3]
    quat1_flat = pose1_flat[:, 3:]
    pos2_flat = pose2_flat[:, :3]
    quat2_flat = pose2_flat[:, 3:]

    # 3. (B*H, 4) 쿼터니언 배치를 (B*H, 3, 3) 회전 행렬 배치로 변환
    mat1_flat = R.from_quat(quat1_flat).as_matrix()
    mat2_flat = R.from_quat(quat2_flat).as_matrix()

    # 4. (B*H, ...) 입력을 사용하여 상대 변환 계산
    pos_rel_flat, mat_rel_flat = compute_rel_transform_batch(
        pos1_flat, mat1_flat, pos2_flat, mat2_flat
    )

    # 5. (B*H, 3, 3) 회전 행렬 배치를 (B*H, 4) 쿼터니언 배치로 변환
    quat_rel_flat = R.from_matrix(mat_rel_flat).as_quat()

    # 6. (B*H, 3)과 (B*H, 4)를 합쳐 (B*H, 7)로 만듦
    pose_rel_flat = np.concatenate([pos_rel_flat, quat_rel_flat], axis=1)
    
    # 7. (B*H, 7) -> (B, H, 7) 원래 Shape으로 복원
    pose_rel = pose_rel_flat.reshape(batch_size, horizon, 7)
    
    return pose_rel


class SimpleDP3Dual(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            use_lang_emb=False,
            use_stage_emb=False,
            use_progress=False,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            predict_type = "relative",
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
        
        self.use_progress = use_progress
        if use_progress:
            assert len(action_shape) == 1 and action_shape[0] == 15
        else:
            raise NotImplementedError("use_progress=False인 경우 구현 안함")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        obs_encoder = DualDP3Encoder(observation_space=obs_dict,
                                #  out_channel=encoder_output_dim,
                                # 기존(encoder_output_dim)은 64였음 마지막에 output_shape의 에서 self.pointcloud_encoder_cfg.out_channels빼는 로직이 있는데 이게 64차원임
                                # 아마 point cloud 제거 과정에서 정리 되지 않고 나음 코드 같음 따라서 0으로 시작해서 output_dim을 더해줌
                                 out_channel=0, #왜 64일까?
                                )
        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps

        if use_progress:
            input_dim = input_dim - 1 # handle progress/gripper separately
            model = ConditionalUnet1D_progress(
                input_dim=input_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                condition_type=condition_type,
                use_down_condition=use_down_condition,
                use_mid_condition=use_mid_condition,
                use_up_condition=use_up_condition,
            )
        else:
            raise NotImplementedError("use_progress=False인 경우 구현 안함")

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.predict_type = predict_type

        print_params(self)
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler


        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)


        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]


            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond)
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], target0=None, target1=None) -> Dict[str, torch.Tensor]:
        # train validation 할때 결국 현재 pos 만 보고 action 어떻게 취할지 예측해보는거 말그대로 predict action
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        nobs_obj0 = custom_normalize(obs_dict['obs_anchor'], self.normalizer, key='obj0_anchor_pos')
        nobs_obj1 = custom_normalize(obs_dict['obs_anchor'], self.normalizer, key='obj1_anchor_pos')

        origin_device = obs_dict['obs_anchor']['obj1_anchor_pos'].device
        obj0_to_obj1_pos = torch.from_numpy(get_rel_pose_batch_horizon(obs_dict['obs_anchor']['obj1_anchor_pos'].detach().cpu().numpy(), obs_dict['obs_anchor']['obj0_anchor_pos'].detach().cpu().numpy())).to(origin_device).float()
        obj1_to_obj0_pos = torch.from_numpy(get_rel_pose_batch_horizon(obs_dict['obs_anchor']['obj0_anchor_pos'].detach().cpu().numpy(), obs_dict['obs_anchor']['obj1_anchor_pos'].detach().cpu().numpy())).to(origin_device).float()
        
        obj0_value = next(iter(nobs_obj0.values()))

        # !! Important
        if obj0_value.shape[1] >1 :
            obj0_value = obj0_value[:, 0].unsqueeze(1)

        B, To = obj0_value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        #Do = self.obs_feature_dim
        To = self.n_obs_steps # 과거 몇개의 관측 데이터를 참고할지

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None

        if self.obs_as_global_cond:
            # 1. 이미지 텐서는 모양을 바꾸면 안 되므로 잠시 분리합니다.
            obj0_img = nobs_obj0.pop('obj0_image', None)
            obj1_img = nobs_obj1.pop('obj1_image', None)

            # 2. 이미지를 제외한 나머지 데이터에만 기존의 reshape 로직을 적용합니다.
            resize_lambda = lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:])
            this_nobs_obj0 = dict_apply(nobs_obj0, resize_lambda)
            this_nobs_obj1 = dict_apply(nobs_obj1, resize_lambda)
            this_obj0_to_obj1_pos  = resize_lambda(obj0_to_obj1_pos)
            this_obj1_to_obj0_pos  = resize_lambda(obj1_to_obj0_pos)


            if self.n_obs_steps > 1:
                nobs_obj0_img = obj0_img.unsqueeze(1).repeat(1,self.n_obs_steps,1,1,1)
                nobs_obj1_img = obj1_img.unsqueeze(1).repeat(1,self.n_obs_steps,1,1,1)

                this_nobs_obj0['obj0_image'] = nobs_obj0_img.reshape(-1, *nobs_obj0_img.shape[2:])
                this_nobs_obj1['obj1_image'] = nobs_obj1_img.reshape(-1, *nobs_obj1_img.shape[2:])

            else:
                this_nobs_obj0['obj0_image'] = obj0_img
                this_nobs_obj1['obj1_image'] = obj1_img


            # 4. 이제 DP3Encoder는 올바른 모양의 데이터를 전달받게 됩니다.
            nobs_features = self.obs_encoder(this_nobs_obj0, this_nobs_obj1, this_obj0_to_obj1_pos, this_obj1_to_obj0_pos)

            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                global_cond = nobs_features.reshape(B, -1)

            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        else:
            raise NotImplementedError("obs_as_global_cond else 구현 안함")


        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        obj0_action = naction_pred[..., :7]
        obj1_action = naction_pred[..., 7:14]
        action_progress = naction_pred[...,14]
        action_progress_expand= action_progress.unsqueeze(-1)

        obj0_action_pred = custom_unnormalize(obj0_action, self.normalizer['obj0_anchor_action'], key=None)
        obj1_action_pred = custom_unnormalize(obj1_action, self.normalizer['obj1_anchor_action'], key=None)

        action_combined = torch.concat([obj0_action_pred, obj1_action_pred, action_progress_expand], dim=-1)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        obj0_action = obj0_action_pred[:,start:end]
        obj1_action = obj1_action_pred[:,start:end]
        full_action = action_combined[:,start:end]

        # get prediction
        result = {
            'action_combined': full_action,
            'obj0_anchor_action': obj0_action,           # [B, 3 ,7]
            'obj1_anchor_action': obj1_action,           # [B, 3 ,7]
            'obj0_anchor_action_pred': obj0_action_pred, # [B, 4 ,7]
            'obj1_anchor_action_pred': obj1_action_pred, # [B, 4 ,7]
        }

        if target0 is not None and target1 is not None:
            obj0_action_pred = obj0_action_pred.to(target0.device)
            mse_obj0 = torch.nn.functional.mse_loss(obj0_action_pred, target0)
            obj1_action_pred = obj1_action_pred.to(target1.device)
            mse_obj1 = torch.nn.functional.mse_loss(obj1_action_pred, target1)
            
            result["obj0_loss"] = mse_obj0.item()
            result["obj1_loss"] = mse_obj1.item()
            
            average_loss = (mse_obj0 + mse_obj1) / 2.0
            result["loss"] = average_loss.item()

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    

    def compute_loss(self, batch):
        nobs_obj0 = custom_normalize(batch['obs_anchor'], self.normalizer, key='obj0_anchor_pos')
        nobs_obj1 = custom_normalize(batch['obs_anchor'], self.normalizer, key='obj1_anchor_pos')
        
        obj0_nactions = custom_normalize(batch['obj0_anchor_action'], self.normalizer['obj0_anchor_action'], key=None)
        obj1_nactions = custom_normalize(batch['obj1_anchor_action'], self.normalizer['obj1_anchor_action'], key=None)

        obj0_to_obj1_pos = torch.from_numpy(get_rel_pose_batch_horizon(batch['obs_anchor']['obj1_anchor_pos'].detach().cpu().numpy(), batch['obs_anchor']['obj0_anchor_pos'].detach().cpu().numpy())).to(obj0_nactions.device).float()
        obj1_to_obj0_pos = torch.from_numpy(get_rel_pose_batch_horizon(batch['obs_anchor']['obj0_anchor_pos'].detach().cpu().numpy(), batch['obs_anchor']['obj1_anchor_pos'].detach().cpu().numpy())).to(obj0_nactions.device).float()
        
        assert obj0_nactions.shape == obj1_nactions.shape == obj0_to_obj1_pos.shape == obj1_to_obj0_pos.shape
        progress = batch['progress'].to(obj0_nactions.device)
        batch_size = obj0_nactions.shape[0]
        
        local_cond = None
        global_cond = None
        obj0_trajectory = obj0_nactions
        obj1_trajectory = obj1_nactions
       
        if self.obs_as_global_cond:
            obj0_img = nobs_obj0.pop('obj0_image', None)
            obj1_img = nobs_obj1.pop('obj1_image', None)

            resize_lambda = lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:])
            this_nobs_obj0 = dict_apply(nobs_obj0, resize_lambda)
            this_nobs_obj1 = dict_apply(nobs_obj1, resize_lambda)
            this_obj0_to_obj1_pos = resize_lambda(obj0_to_obj1_pos)
            this_obj1_to_obj0_pos = resize_lambda(obj1_to_obj0_pos)

            if self.n_obs_steps > 1:
                nobs_obj0_img = obj0_img.unsqueeze(1).repeat(1,self.n_obs_steps,1,1,1)
                nobs_obj1_img = obj1_img.unsqueeze(1).repeat(1,self.n_obs_steps,1,1,1)

                this_nobs_obj0['obj0_image'] = nobs_obj0_img.reshape(-1, *nobs_obj0_img.shape[2:])
                this_nobs_obj1['obj1_image'] = nobs_obj1_img.reshape(-1, *nobs_obj1_img.shape[2:])

            else:
                this_nobs_obj0['obj0_image'] = obj0_img
                this_nobs_obj1['obj1_image'] = obj1_img


            # 4. 이제 DP3Encoder는 올바른 모양의 데이터를 전달받게 됩니다.
            nobs_features = self.obs_encoder(this_nobs_obj0, this_nobs_obj1, this_obj0_to_obj1_pos, this_obj1_to_obj0_pos)

            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                global_cond = nobs_features.reshape(batch_size, -1)

        else:
            raise NotImplementedError("obs_as_global_cond else 구현 안함")

        #region mask generation version 1
        trajectory = torch.cat([obj0_trajectory, obj1_trajectory, progress], dim=-1)
        noise = torch.randn_like(trajectory)
        condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        #endregion

        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        cond_data = trajectory
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                            local_cond=local_cond, 
                            global_cond=global_cond)


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")


        visualize = False
        predict_type = 'relative'
        if visualize:
            dp3_visualize(
                # self.normalizer['agent_pos'].unnormalize(nobs['agent_pos']), 
                # self.normalizer['action'].unnormalize(pred), 
                # self.normalizer['action'].unnormalize(target)
                custom_unnormalize(nobs['agent_pos'], self.normalizer['agent_pos'], key=None),
                # pred=custom_unnormalize(pred.clone().detach(), self.normalizer['action'], key=None),
                target=custom_unnormalize(target, self.normalizer['action'], key=None),
                predict_type=predict_type,
            )

        if self.use_progress:
            # continous loss for progress
            loss = F.mse_loss(pred, target, reduction='none')
            # loss[..., -1:] *= 0.1

            # # binary loss for gripper
            # obj0_loss_trans = F.mse_loss(pred[..., :3], target[..., :3], reduction='none')
            # obj1_loss_trans = F.mse_loss(pred[..., 7:10], target[..., 7:10], reduction='none')

            # obj0_loss_ori = F.mse_loss(pred[..., 3:7], target[..., 3:7], reduction='none')
            # obj1_loss_ori = F.mse_loss(pred[..., 10:14], target[..., 10:14], reduction='none')

            # loss_gripper = F.binary_cross_entropy(pred[..., -1:], target[..., -1:], reduction='none')

            # loss = torch.concat([
            #             obj1_loss_trans, 
            #             obj1_loss_ori, 
            #             obj0_loss_trans, 
            #             obj0_loss_ori, 
            #             loss_gripper
            #             ], dim=-1)

            # loss[..., -1:] *= 0.1
        else:
            loss = F.mse_loss(pred, target, reduction='none')

        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()


        loss_dict = {
                'bc_loss': loss.item(),
            }

        return loss, loss_dict