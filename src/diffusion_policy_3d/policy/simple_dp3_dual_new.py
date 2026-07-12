from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.common.model_util import print_params

# Import Model Architecture
from diffusion_policy_3d.model.vision.dual_extractor_transformer import DualTransformerEncoder
from diffusion_policy_3d.model.diffusion.dolbi_denoise_actor import DualTransformerHead

# ==============================================================================
# Helper Functions
# ==============================================================================

def compute_rel_transform_batch(A_pos, A_mat, B_pos, B_mat):
    batch_size = A_pos.shape[0]
    T_WA = np.zeros((batch_size, 4, 4))
    T_WA[:, :3, :3] = A_mat
    T_WA[:, :3, 3] = A_pos
    T_WA[:, 3, 3] = 1.0
    
    T_WB = np.zeros((batch_size, 4, 4))
    T_WB[:, :3, :3] = B_mat
    T_WB[:, :3, 3] = B_pos
    T_WB[:, 3, 3] = 1.0

    T_WA_inv = np.linalg.inv(T_WA)
    T_AB = np.matmul(T_WA_inv, T_WB)
    return T_AB[:, :3, 3], T_AB[:, :3, :3]

def get_rel_pose_batch_horizon(pose1, pose2):
    batch_size, horizon = pose1.shape[0], pose1.shape[1]
    total_elements = batch_size * horizon
    pose1_flat = pose1.reshape(total_elements, 7)
    pose2_flat = pose2.reshape(total_elements, 7)

    pos1_flat = pose1_flat[:, :3]
    quat1_flat = pose1_flat[:, 3:]
    pos2_flat = pose2_flat[:, :3]
    quat2_flat = pose2_flat[:, 3:]

    mat1_flat = R.from_quat(quat1_flat).as_matrix()
    mat2_flat = R.from_quat(quat2_flat).as_matrix()

    pos_rel_flat, mat_rel_flat = compute_rel_transform_batch(pos1_flat, mat1_flat, pos2_flat, mat2_flat)
    quat_rel_flat = R.from_matrix(mat_rel_flat).as_quat()

    pose_rel_flat = np.concatenate([pos_rel_flat, quat_rel_flat], axis=1)
    return pose_rel_flat.reshape(batch_size, horizon, 7)

def custom_normalize(data, normalizer, key):
    pose = copy.deepcopy(data)
    use_progress = True if (key is not None and pose[key].shape[-1] > 7) or (key is None and pose.shape[-1] > 7) else False

    if key is not None:
        if key == 'obj0_anchor_pos':
            pose.pop('obj1_anchor_pos', None)
            pose.pop('obj1_image', None)
        elif key == 'obj1_anchor_pos':
            pose.pop('obj0_anchor_pos', None)
            pose.pop('obj0_image', None)
        
        target_tensor = pose[key]
    else:
        target_tensor = pose

    # Rotation(3:7)과 Progress(7:) 추출
    if use_progress:
        progress = target_tensor[..., 7:8]
    
    # 쿼터니언은 따로 정규화 (방향 벡터이므로 길이 1로 맞춤)
    pose_ori = target_tensor[..., 3:7]
    pose_ori = F.normalize(pose_ori, dim=-1)

    # 2. [수정됨] Normalizer에는 7차원(또는 8차원) 데이터를 그대로 넣음
    # Normalizer가 가지고 있는 통계(Scale) 차원과 입력 차원을 맞춰주기 위함입니다.
    # Pos(3)만 넣으면 차원 불일치 에러 발생함.
    npose = normalizer.normalize(pose)

    # 3. Normalizer의 결과에서 Rotation 부분만 우리가 만든 Unit-Norm 쿼터니언으로 교체
    if key is not None:
        # npose[key]는 현재 Linear Normalize된 상태 (Pos + Scaled_Rot + Scaled_Progress)
        # 여기서 Pos는 유지하고, Rot와 Progress를 교체/복구함
        npose_val = npose[key]
        npose_pos = npose_val[..., :3] # 정규화된 Position
        
        # 합치기: [Norm_Pos, Unit_Quat, Raw_Progress]
        # (Progress는 보통 0~1이므로 별도 정규화 없이 쓰거나, 필요 시 LinearNorm 결과 그대로 씀. 
        #  여기서는 기존 코드 로직에 따라 분리했던 progress를 다시 붙이는 방식을 따름)
        concat_list = [npose_pos, pose_ori]
        if use_progress:
            # Progress도 Normalizer를 통과한 값을 쓸지, 원본을 쓸지 결정해야 함.
            # 기존 코드는 원본을 다시 붙이는 로직이었으므로 원본 사용.
            # 만약 Normalizer가 Progress 통계도 가지고 있다면 npose_val[..., 7:8]을 써야 함.
            # 안전하게 기존 로직(normalize 제외)을 따름:
            concat_list.append(progress)
            
        npose[key] = torch.cat(concat_list, dim=-1)
    else:
        npose_pos = npose[..., :3]
        concat_list = [npose_pos, pose_ori]
        if use_progress:
            concat_list.append(progress)
        npose = torch.cat(concat_list, dim=-1)

    return npose


def custom_unnormalize(data, normalizer, key):
    pose = copy.deepcopy(data)
    use_progress = True if (key is not None and pose[key].shape[-1] > 7) or (key is None and pose.shape[-1] > 7) else False

    if key is not None:
        target_tensor = pose[key]
    else:
        target_tensor = pose

    if use_progress:
        progress = target_tensor[..., 7:8]
    
    pose_ori = target_tensor[..., 3:7]
    pose_ori = F.normalize(pose_ori, dim=-1)

    # [수정됨] Unnormalize도 동일하게 7차원 전체를 넣어서 수행
    # 단, 우리가 덮어씌운 Rotation 값 때문에 Unnormalize 결과의 Rotation은 엉망일 수 있음.
    # 하지만 Position 복원이 목적이므로 상관없음. Rotation은 따로 저장해둔 pose_ori 사용.
    npose = normalizer.unnormalize(pose)

    if key is not None:
        npose_val = npose[key]
        npose_pos = npose_val[..., :3] # 복원된 Position
        
        concat_list = [npose_pos, pose_ori]
        if use_progress:
            concat_list.append(progress)
        npose[key] = torch.cat(concat_list, dim=-1)
    else:
        npose_pos = npose[..., :3]
        concat_list = [npose_pos, pose_ori]
        if use_progress:
            concat_list.append(progress)
        npose = torch.cat(concat_list, dim=-1)
        
    return npose

# ==============================================================================
# Policy Class
# ==============================================================================

class TransformerDP3DualPolicy(BasePolicy):
    def __init__(self, 
             shape_meta: dict,
             noise_scheduler: DDPMScheduler,
             horizon, 
             n_action_steps, 
             n_obs_steps,
             num_inference_steps=None,
             embedding_dim=256,
             num_attn_heads=8,
             num_layers=4,
             dropout=0.1,
             use_progress=True,
             **kwargs):
        super().__init__()
        
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.use_progress = use_progress
        self.noise_scheduler = noise_scheduler
        
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        # Action Dim Check (Obj0(7) + Obj1(7) + Progress(1) = 15)
        action_shape = shape_meta['action']['shape']
        self.action_dim = action_shape[0]
        if self.use_progress:
            assert self.action_dim == 15, "Action dim must be 15 when using progress"

        # 1. Encoder (Transformer-Friendly)
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])
        
        self.obs_encoder = DualTransformerEncoder(
            observation_space=obs_dict,
            embedding_dim=embedding_dim,
        )
        
        # 2. Backbone (TransformerHead)
        self.model = DualTransformerHead(
            action_dim=self.action_dim,
            embedding_dim=embedding_dim,
            num_heads=num_attn_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.normalizer = LinearNormalizer()
        self.kwargs = kwargs
        
        print_params(self)

    # ==========================================================================
    # Helper: Prepare Inputs for Encoder
    # ==========================================================================
    def _prepare_encoder_inputs(self, nobs_obj0, nobs_obj1, obj0_to_obj1_pos, obj1_to_obj0_pos):
        # Extract images
        obj0_img = nobs_obj0.pop('obj0_image', None)
        obj1_img = nobs_obj1.pop('obj1_image', None)

        # Reshape function: (B, T, ...) -> (B*T, ...)
        resize_lambda = lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        
        this_nobs_obj0 = dict_apply(nobs_obj0, resize_lambda)
        this_nobs_obj1 = dict_apply(nobs_obj1, resize_lambda)
        this_obj0_to_obj1_pos = resize_lambda(obj0_to_obj1_pos)
        this_obj1_to_obj0_pos = resize_lambda(obj1_to_obj0_pos)

        # Handle Image Repeat/Slice
        if self.n_obs_steps > 1:
            nobs_obj0_img = obj0_img.unsqueeze(1).repeat(1, self.n_obs_steps, 1, 1, 1)
            nobs_obj1_img = obj1_img.unsqueeze(1).repeat(1, self.n_obs_steps, 1, 1, 1)
            this_nobs_obj0['obj0_image'] = nobs_obj0_img.reshape(-1, *nobs_obj0_img.shape[2:])
            this_nobs_obj1['obj1_image'] = nobs_obj1_img.reshape(-1, *nobs_obj1_img.shape[2:])
        else:
            this_nobs_obj0['obj0_image'] = obj0_img
            this_nobs_obj1['obj1_image'] = obj1_img
            
        return this_nobs_obj0, this_nobs_obj1, this_obj0_to_obj1_pos, this_obj1_to_obj0_pos

    # ==========================================================================
    # Inference: Predict Action
    # ==========================================================================
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], target0=None, target1=None) -> Dict[str, torch.Tensor]:
        # 1. Normalize
        nobs_obj0 = custom_normalize(obs_dict['obs_anchor'], self.normalizer, key='obj0_anchor_pos')
        nobs_obj1 = custom_normalize(obs_dict['obs_anchor'], self.normalizer, key='obj1_anchor_pos')
        
        # 2. Relative Pose
        origin_device = obs_dict['obs_anchor']['obj1_anchor_pos'].device
        obj0_to_obj1_pos = torch.from_numpy(get_rel_pose_batch_horizon(
            obs_dict['obs_anchor']['obj1_anchor_pos'].detach().cpu().numpy(), 
            obs_dict['obs_anchor']['obj0_anchor_pos'].detach().cpu().numpy()
        )).to(origin_device).float()
        
        obj1_to_obj0_pos = torch.from_numpy(get_rel_pose_batch_horizon(
            obs_dict['obs_anchor']['obj0_anchor_pos'].detach().cpu().numpy(), 
            obs_dict['obs_anchor']['obj1_anchor_pos'].detach().cpu().numpy()
        )).to(origin_device).float()

        # 3. Encode
        bsz = nobs_obj0['obj0_anchor_pos'].shape[0]
        inp_obj0, inp_obj1, inp_rel01, inp_rel10 = self._prepare_encoder_inputs(
            nobs_obj0, nobs_obj1, obj0_to_obj1_pos, obj1_to_obj0_pos
        )
        
        encoded_tokens = self.obs_encoder(inp_obj0, inp_obj1, inp_rel01, inp_rel10)
        context_tokens = encoded_tokens.reshape(bsz, -1, encoded_tokens.shape[-1])

        # 4. Diffusion Loop
        trajectory = torch.randn((bsz, self.horizon, self.action_dim), device=self.device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
        for t in self.noise_scheduler.timesteps:
            timesteps = torch.tensor([t], dtype=torch.long, device=self.device).expand(bsz)
            
            # Predict Noise
            model_output = self.model(
                noisy_action=trajectory,
                timestep=timesteps,
                context_tokens=context_tokens
            )
            
            # Step (Denoise)
            trajectory = self.noise_scheduler.step(model_output, t, trajectory).prev_sample

        # 5. Output Formatting & Unnormalization
        obj0_action = trajectory[..., :7]
        obj1_action = trajectory[..., 7:14]
        action_progress = trajectory[..., 14:15]

        obj0_action_pred = custom_unnormalize(obj0_action, self.normalizer['obj0_anchor_action'], key=None)
        obj1_action_pred = custom_unnormalize(obj1_action, self.normalizer['obj1_anchor_action'], key=None)
        
        # [수정됨] Progress 강제 범위 제한: Sigmoid 대신 Clamp 사용
        # Diffusion Model은 선형적인 값을 복원하므로, 0~1 범위를 벗어난 값을 잘라내는 것이 안전합니다.
        action_progress_pred = torch.clamp(action_progress, 0, 1)

        action_combined = torch.cat([obj0_action_pred, obj1_action_pred, action_progress_pred], dim=-1)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        
        result = {
            'action_combined': action_combined[:, start:end],
            'obj0_anchor_action': obj0_action_pred[:, start:end],
            'obj1_anchor_action': obj1_action_pred[:, start:end],
            'obj0_anchor_action_pred': obj0_action_pred,
            'obj1_anchor_action_pred': obj1_action_pred
        }
        
        if target0 is not None and target1 is not None:
            l0 = F.mse_loss(obj0_action_pred.to(target0.device), target0)
            l1 = F.mse_loss(obj1_action_pred.to(target1.device), target1)
            result["obj0_loss"] = l0.item()
            result["obj1_loss"] = l1.item()
            result["loss"] = (l0 + l1).item() / 2.0

        return result

    # ==========================================================================
    # Training: Compute Loss
    # ==========================================================================
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # 1. Normalize & Prepare Data
        nobs_obj0 = custom_normalize(batch['obs_anchor'], self.normalizer, key='obj0_anchor_pos')
        nobs_obj1 = custom_normalize(batch['obs_anchor'], self.normalizer, key='obj1_anchor_pos')
        
        obj0_nactions = custom_normalize(batch['obj0_anchor_action'], self.normalizer['obj0_anchor_action'], key=None)
        obj1_nactions = custom_normalize(batch['obj1_anchor_action'], self.normalizer['obj1_anchor_action'], key=None)
        progress = batch['progress'].to(obj0_nactions.device)

        obj0_to_obj1_pos = torch.from_numpy(get_rel_pose_batch_horizon(
            batch['obs_anchor']['obj1_anchor_pos'].detach().cpu().numpy(), 
            batch['obs_anchor']['obj0_anchor_pos'].detach().cpu().numpy()
        )).to(obj0_nactions.device).float()
        
        obj1_to_obj0_pos = torch.from_numpy(get_rel_pose_batch_horizon(
            batch['obs_anchor']['obj0_anchor_pos'].detach().cpu().numpy(), 
            batch['obs_anchor']['obj1_anchor_pos'].detach().cpu().numpy()
        )).to(obj0_nactions.device).float()

        # 2. Encode
        bsz = obj0_nactions.shape[0]
        inp_obj0, inp_obj1, inp_rel01, inp_rel10 = self._prepare_encoder_inputs(
            nobs_obj0, nobs_obj1, obj0_to_obj1_pos, obj1_to_obj0_pos
        )
        
        encoded_tokens = self.obs_encoder(inp_obj0, inp_obj1, inp_rel01, inp_rel10)
        context_tokens = encoded_tokens.reshape(bsz, -1, encoded_tokens.shape[-1])

        # 3. Ground Truth Trajectory
        trajectory = torch.cat([obj0_nactions, obj1_nactions, progress], dim=-1)

        # 4. Noise
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # 5. Forward
        pred = self.model(
            noisy_action=noisy_trajectory,
            timestep=timesteps,
            context_tokens=context_tokens
        )

        # 6. Loss
        target = noise if self.noise_scheduler.config.prediction_type == 'epsilon' else trajectory
        loss = F.mse_loss(pred, target)
        
        return loss, {'loss': loss.item()}