import os
from typing import Dict
from diffusion_policy_3d.model.clip.clip import build_model, load_clip, tokenize
import torch
import numpy as np
import copy
from termcolor import cprint
import pickle

import cv2

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

import utils.transform_utils as T
from utils.pose_utils import calculate_action, rodrigues


class RLBenchBaseDataset(BaseDataset):
    def __init__(self,
            root_dir, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            # add task specific embedding
            has_lang_emb=False,
            # add task specific augmentation
            random_aug=True,
            symmetric_axis=None,
            symmetric_theta_start=-1,
            symmetric_theta_end=-1,
            symmetric_theta_step=-1,

            ):
        super().__init__()
        self.root_dir = root_dir
        if 'pt' in self.root_dir:
            zarr_path = os.path.join(self.root_dir, 'zarr_pt')
        else:
            zarr_path = os.path.join(self.root_dir, 'zarr')
        cprint(f"[Dataset] zarr_path: {zarr_path}", "yellow")
        self.replay_buffer = self.get_replay_buffer(zarr_path)
        self.replay_buffer_val = self.get_replay_buffer(zarr_path.replace('train', 'test'))

        # (debug only) Dummy replay buffer
        # self.replay_buffer = ReplayBuffer.create_dummy()

        train_mask, val_mask = self.get_train_val_mask()
        if train_mask is not None:
            train_mask = downsample_mask(
                mask=train_mask, 
                max_n=max_train_episodes, 
                seed=seed)
        

        # applying symetric augmentation
        self.symmetric_aug_choice = []
        if symmetric_axis is not None:
            symmetric_axis = np.array(symmetric_axis).reshape(-1, 3)
            symmetric_theta_start = np.array(symmetric_axis).reshape(-1, 1)
            symmetric_theta_end = np.array(symmetric_theta_end).reshape(-1, 1)
            symmetric_theta_step = np.array(symmetric_theta_step).reshape(-1, 1)
            for idx, axis in enumerate(symmetric_axis):
                cur_theta_list = np.radians(np.arange(start=symmetric_theta_start[idx], stop=symmetric_theta_end[idx]+1, step=symmetric_theta_step[idx]))
                for theta in cur_theta_list:
                    self.symmetric_aug_choice.append(np.concatenate([axis, theta.reshape(-1)]))

        self.sampler = self.get_sampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            n_aug=len(self.symmetric_aug_choice) if len(self.symmetric_aug_choice) > 0 else -1,
        )
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # data augmentation
        self.random_aug = random_aug
        self.symmetric_aug = symmetric_axis is not None

        cprint(f"[Dataset] random_aug: {random_aug}", "yellow")
        cprint(f"[Dataset] symmetric_aug: {self.symmetric_aug}", "yellow")

        # language embeding
        self.has_lang_emb = has_lang_emb
        self._lang_token_embs = self.get_language_embedding()
        cprint(f"[Dataset] has_lang_emb: {has_lang_emb}", "yellow")
        

    def get_replay_buffer(self, zarr_path):
        return ReplayBuffer.copy_from_path(
                    zarr_path, keys=['state', 'state_in_world', 'state_next', 'action', 'progress', 'progress_binary', 'variation', 'task_stage'])#, 'point_cloud', 'img'])

    def get_sampler(self, replay_buffer, sequence_length, pad_before, pad_after, episode_mask, n_aug=-1):
        return SequenceSampler(
                    replay_buffer=replay_buffer, 
                    sequence_length=sequence_length,
                    pad_before=pad_before, 
                    pad_after=pad_after,
                    episode_mask=episode_mask, 
                    n_aug=n_aug,
                )

    def get_language_embedding(self):
        # load clip model
        model, _ = load_clip("RN50", jit=False, device='cuda')
        clip_model = build_model(model.state_dict())
        clip_model.to('cuda')
        del model

        # load all descriptions                
        description_path = os.path.join(self.root_dir, "all_variation_descriptions.pkl")
        print(f"Loading demo from the path {description_path}")
        with open(description_path, "rb") as fin:
            descriptions = pickle.load(fin) # dict of VAR: DESC

        cur_task_id = 0 # cur_task_id is always zero as there is only one task

        # pre-generate all language embeddings
        _lang_token_embs = {}
        n_variation = len(descriptions)
        for var_id in range(n_variation):
            cur_description = descriptions[var_id]

            tokens = tokenize(cur_description).numpy()
            token_tensor = torch.from_numpy(tokens).to('cuda')
            sentence_emb, token_embs = clip_model.encode_text_with_embeddings(token_tensor)

            # print(cur_description)      # 5 sentence
            # print(token_embs.shape)     # [5, 77, 512]
            # print(sentence_emb.shape)   # [5, 1024]

            # obs_dict["lang_goal_emb"] = sentence_emb[0].float().detach().cpu().numpy()
            # lang_token_embs = token_embs[0].float().detach().cpu().numpy()
            lang_token_embs = sentence_emb[0].float().detach().cpu().numpy()[None, :]

            _lang_token_embs[f"{cur_task_id}_{var_id}"] = lang_token_embs
        return _lang_token_embs

    def get_train_val_mask(self):
        train_mask = np.full(self.replay_buffer.n_episodes, True)
        val_mask = np.full(self.replay_buffer_val.n_episodes, True)
        return train_mask, val_mask

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer_val, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.train_mask = self.val_mask
        val_set.random_aug = False
        val_set.symmetric_aug = False
        val_set.symmetric_aug_choice = []
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'obj0_world_pos' : self.replay_buffer['obj0_state_in_world'][...,:][:,:3],
            'obj1_world_pos' : self.replay_buffer['obj1_state_in_world'][...,:][:,:3],
            'obj0_world_action' : self.replay_buffer['obj0_world_action'][:,:3],
            'obj1_world_action' : self.replay_buffer['obj1_world_action'][:,:3],
            'obj0_relpos' : self.replay_buffer['obj0_to_obj1_state'][...,:][:,:3],
            'obj1_relpos' : self.replay_buffer['obj1_to_obj0_state'][...,:][:,:3],
            'obj0_relaction': self.replay_buffer['obj0_to_obj1_action'][:,:3],
            'obj1_relaction': self.replay_buffer['obj1_to_obj0_action'][:,:3],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _add_symmetric_noise_to_pos(self, poses, axis, theta):
        new_poses = []
        assert theta <= np.pi # !! theta should be radians
        pose_noise = np.eye(4)

        # pose_noise[:3,:3] = cv2.Rodrigues(axis*theta)[0]
        pose_noise[:3,:3] = rodrigues(axis*theta)[0] # equavalent to cv2.Rodrigues(axis*theta)[0]

        for p in poses:
            pose = T.pose2mat((p[:3], p[3:]))
            pose_perturbed = pose @ pose_noise

            new_pose = np.concatenate(T.mat2pose(pose_perturbed), axis=0)
            new_poses.append(new_pose)
        return np.array(new_poses)

    def _add_random_noise_to_pos(self, poses, max_theta=np.radians(5), max_tran=1e-3):
        def random_direction():
            vec = np.random.randn(3).reshape(3)
            vec /= np.linalg.norm(vec)
            return vec
        
        max_trans = [max_tran for _ in range(3)]
        # !! theta in radians
        assert max_theta <= np.pi

        new_poses = []
        for p in poses:
            pose = T.pose2mat((p[:3], p[3:]))
          
            axis = random_direction()
            theta = np.random.uniform(0, max_theta)
            pose_noise = np.eye(4)
            # pose_noise[:3,:3] = cv2.Rodrigues(axis*theta)[0]
            pose_noise[:3,:3] = rodrigues(axis*theta)[0] # equavalent to cv2.Rodrigues(axis*theta)[0]
            pose_perturbed = pose @ pose_noise
            trans_noise = np.zeros((3))
            for ii in range(3):
                trans_noise[ii] = np.random.uniform(-max_trans[ii], max_trans[ii])
            pose_perturbed[:3,3] += trans_noise

            new_pose = np.concatenate(T.mat2pose(pose_perturbed), axis=0)
            new_poses.append(new_pose)
        return np.array(new_poses)

    def _sample_to_data(self, sample, aug_idx):
        if self.random_aug or self.symmetric_aug:
            # all in target's frame
            agent_pos = sample['state'][:,].astype(np.float32)
            action = sample['action'].astype(np.float32)
            agent_pos_next = sample['state_next'][:,].astype(np.float32)

            if self.symmetric_aug:
                # get parameter
                aug_param = self.symmetric_aug_choice[aug_idx]
                # apply symmetric augmentation
                noisy_agent_pos = self._add_symmetric_noise_to_pos(agent_pos, axis=aug_param[:3], theta=aug_param[3]).astype(np.float32)
                noisy_agent_pos_next = self._add_symmetric_noise_to_pos(agent_pos_next, axis=aug_param[:3], theta=aug_param[3]).astype(np.float32)
            else:
                # no symmetric augmentation
                noisy_agent_pos = agent_pos
                noisy_agent_pos_next = agent_pos_next
            
            if self.random_aug:
                # apply random noise augmentation
                noisy_agent_pos2 = self._add_random_noise_to_pos(noisy_agent_pos).astype(np.float32)

            AUG_GOAL = True # TODO: read from cfg
            if AUG_GOAL:
                # apply symmetric augmentation to next pose
                action_from_noisy_to_next = np.array([calculate_action(p1, p2) for p1, p2 in zip(noisy_agent_pos2, noisy_agent_pos_next)]).astype(np.float32)
                # debug only
                # agent_pos_next2 = np.array([calculate_goal_pose(p1, a) for p1, a in zip(noisy_agent_pos2, action_from_noisy_to_next)]).astype(np.float32)
                # print(agent_pos_next, agent_pos_next2) # should be the same
            else:
                # DO NOT apply symmetric augmentation to next pose
                action_from_noisy_to_next = np.array([calculate_action(p1, p2) for p1, p2 in zip(noisy_agent_pos2, agent_pos_next)]).astype(np.float32)

            agent_pos = noisy_agent_pos2
            action = action_from_noisy_to_next

            # print("agent_pos after", agent_pos[:, 3:])
            # print("action after", action[:, 3:])
            # print("agent_pos after norm", agent_pos[:, 3:] / np.linalg.norm(agent_pos[:, 3:], axis=-1))
            # print("action after norm", action[:, 3:] / np.linalg.norm(action[:, 3:], axis=-1))
        else:
            agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
            # point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)

            # delta pose
            action = sample['action'].astype(np.float32) # T, D_action
            # absolute pose
            # action = sample['state_next'].astype(np.float32) # T, D_action

        # progress prediction
        # progress = sample['progress'].astype(np.float32)
        progress = sample['progress_binary'].astype(np.float32)
        progress[progress <= 0.8] *= 0.
        action = np.concatenate([action, progress], axis=1).astype(np.float32)
        
        # use_euler = True
        # if use_euler:
        #     agent_pos = sample['state'][:,].astype(np.float32)
        #     agent_pos_euler = np.array([np.concatenate([p[:3], euler_from_quaternion(*p[3:])]) for p in agent_pos])
        #     agent_pos_next = sample['state_next'][:,].astype(np.float32)
        #     agent_pos_next_euler = np.array([np.concatenate([p[:3], euler_from_quaternion(*p[3:])]) for p in agent_pos_next])
        #     action_euler = agent_pos_next_euler - agent_pos_euler

        #     print("agent_pos_euler", agent_pos_euler[:, 3:])
        #     print("action_euler", action_euler[:, 3:])

        data = {
            'obs': {
                # 'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos, # T, D_pos
            },
            'action': action,
        }

        cur_task_id = 0 # cur_task_id is always zero as there is only one task
        if self.has_lang_emb:
            var_id = sample['variation'].astype(np.int64)[0][0] # (t, 1)
            # obs_dict["lang_goal_emb"] = sentence_emb[0].float().detach().cpu().numpy()
            lang_token_embs = self._lang_token_embs[f"{cur_task_id}_{var_id}"]
        else:
            lang_token_embs = self._lang_token_embs[f"{0}_{0}"]
            # lang_token_embs = np.zeros((1, 1024)).astype(np.float32)
        data['obs']['lang_token_embs'] = lang_token_embs


        stage_embs = np.zeros((1, 3)).astype(np.float32)
        data['obs']['stage_embs'] = stage_embs
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample, aug_idx = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample, aug_idx)
        torch_data = dict_apply(data, torch.from_numpy)

        # debug: visualize input and groundtruth
        # from utils.vis_utils import dp3_visualize
        # dp3_visualize(agent_pos=data['obs']['agent_pos'], target=data['action'], visualize=True)

        # print(torch_data['obs']['agent_pos'][0])
        # print(torch_data['obs']['agent_pos'][3])
        # print(torch_data['action'][0])
        # print(torch_data['action'][3])
        return torch_data


class RLBenchDualBaseDataset(BaseDataset):
    def __init__(self,
            root_dir, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            # add task specific embedding
            has_lang_emb=False,
            # add task specific augmentation
            random_aug=True,
            symmetric_axis=None,
            symmetric_theta_start=-1,
            symmetric_theta_end=-1,
            symmetric_theta_step=-1,

            ):
        super().__init__()
        self.root_dir = root_dir
        if 'pt' in self.root_dir:
            zarr_path = os.path.join(self.root_dir, 'zarr_pt')
        else:
            zarr_path = os.path.join(self.root_dir, 'zarr')
        cprint(f"[Dataset] zarr_path: {zarr_path}", "yellow")
        self.replay_buffer = self.get_replay_buffer(zarr_path)

        try:
            self.replay_buffer_val = self.get_replay_buffer(zarr_path.replace('train', 'val')) # should change 변경해야함!!!!!!!!!!!!!!!!
        except:
            self.replay_buffer_val = self.get_replay_buffer(zarr_path.replace('train', 'train')) 

        # (debug only) Dummy replay buffer
        # self.replay_buffer = ReplayBuffer.create_dummy()

        train_mask, val_mask = self.get_train_val_mask()
        if train_mask is not None:
            train_mask = downsample_mask(
                mask=train_mask, 
                max_n=max_train_episodes, 
                seed=seed)
        

        # applying symetric augmentation
        self.symmetric_aug_choice = []
        if symmetric_axis is not None:
            symmetric_axis = np.array(symmetric_axis).reshape(-1, 3)
            symmetric_theta_start = np.array(symmetric_axis).reshape(-1, 1)
            symmetric_theta_end = np.array(symmetric_theta_end).reshape(-1, 1)
            symmetric_theta_step = np.array(symmetric_theta_step).reshape(-1, 1)
            for idx, axis in enumerate(symmetric_axis):
                cur_theta_list = np.radians(np.arange(start=symmetric_theta_start[idx], stop=symmetric_theta_end[idx]+1, step=symmetric_theta_step[idx]))
                for theta in cur_theta_list:
                    self.symmetric_aug_choice.append(np.concatenate([axis, theta.reshape(-1)]))

        self.sampler = self.get_sampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            n_aug=len(self.symmetric_aug_choice) if len(self.symmetric_aug_choice) > 0 else -1,
        )
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # data augmentation
        self.random_aug = random_aug
        self.symmetric_aug = symmetric_axis is not None

        cprint(f"[Dataset] random_aug: {random_aug}", "yellow")
        cprint(f"[Dataset] symmetric_aug: {self.symmetric_aug}", "yellow")

        # language embeding
        self.has_lang_emb = has_lang_emb
        self._lang_token_embs = self.get_language_embedding()
        cprint(f"[Dataset] has_lang_emb: {has_lang_emb}", "yellow")

        self.obj0_image, self.obj1_image = self.get_grasp_image()

    def get_grasp_image(self):
        obj0_image_path = os.path.join(self.root_dir, 'episodes', 'episode0', 'grasp_obj0_rgb', '0.png')
        obj1_image_path = os.path.join(self.root_dir, 'episodes', 'episode0', 'grasp_obj1_rgb', '0.png')
        
        assert os.path.isfile(obj0_image_path), f'{obj0_image_path} There is no image of obj0'
        assert os.path.isfile(obj1_image_path), f'{obj1_image_path} There is no image of obj1'
        
        # 1. 이미지 로드 및 리사이즈
        obj0_image = cv2.resize(cv2.imread(obj0_image_path), (224, 224), interpolation=cv2.INTER_AREA)
        obj1_image = cv2.resize(cv2.imread(obj1_image_path), (224, 224), interpolation=cv2.INTER_AREA)

        # 2. BGR -> RGB 색상 공간 변환
        obj0_image_rgb = cv2.cvtColor(obj0_image, cv2.COLOR_BGR2RGB)
        obj1_image_rgb = cv2.cvtColor(obj1_image, cv2.COLOR_BGR2RGB)

        # 3. (H, W, C) -> (C, H, W) 차원 순서 변경
        # (224, 224, 3) -> (3, 224, 224)
        obj0_image_chw = obj0_image_rgb.transpose(2, 0, 1)
        obj1_image_chw = obj1_image_rgb.transpose(2, 0, 1)

        obj0_image_chw_normalized = obj0_image_chw.astype(np.float32) / 255.0
        obj1_image_chw_normalized = obj1_image_chw.astype(np.float32) / 255.0

        return obj0_image_chw_normalized, obj1_image_chw_normalized

    def get_replay_buffer(self, zarr_path):
        return ReplayBuffer.copy_from_path(
                    zarr_path, keys=['obj0_anchor_action', 'obj0_state_in_anchor', 'obj0_state_next_in_anchor',
                                     'obj1_anchor_action', 'obj1_state_in_anchor', 'obj1_state_next_in_anchor',
                                    #  'obj0_to_obj1_state', 'obj1_to_obj0_state',
                                     'progress','progress_binary','variation'])

    def get_sampler(self, replay_buffer, sequence_length, pad_before, pad_after, episode_mask, n_aug=-1):
        return SequenceSampler(
                    replay_buffer=replay_buffer, 
                    sequence_length=sequence_length,
                    pad_before=pad_before, 
                    pad_after=pad_after,
                    episode_mask=episode_mask, 
                    n_aug=n_aug,
                )

    def get_language_embedding(self):
        # load clip model
        model, _ = load_clip("RN50", jit=False, device='cuda')
        clip_model = build_model(model.state_dict())
        clip_model.to('cuda')
        del model

        # load all descriptions                
        description_path = os.path.join(self.root_dir, "all_variation_descriptions.pkl")
        print(f"Loading demo from the path {description_path}")
        with open(description_path, "rb") as fin:
            descriptions = pickle.load(fin) # dict of VAR: DESC

        cur_task_id = 0 # cur_task_id is always zero as there is only one task

        # pre-generate all language embeddings
        _lang_token_embs = {}
        n_variation = len(descriptions)
        for var_id in range(n_variation):
            cur_description = descriptions[var_id]

            tokens = tokenize(cur_description).numpy()
            token_tensor = torch.from_numpy(tokens).to('cuda')
            sentence_emb, token_embs = clip_model.encode_text_with_embeddings(token_tensor)

            # print(cur_description)      # 5 sentence
            # print(token_embs.shape)     # [5, 77, 512]
            # print(sentence_emb.shape)   # [5, 1024]

            # obs_dict["lang_goal_emb"] = sentence_emb[0].float().detach().cpu().numpy()
            # lang_token_embs = token_embs[0].float().detach().cpu().numpy()
            lang_token_embs = sentence_emb[0].float().detach().cpu().numpy()[None, :]

            _lang_token_embs[f"{cur_task_id}_{var_id}"] = lang_token_embs
        return _lang_token_embs

    def get_train_val_mask(self):
        train_mask = np.full(self.replay_buffer.n_episodes, True)
        val_mask = np.full(self.replay_buffer_val.n_episodes, True)
        return train_mask, val_mask

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer_val, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.train_mask = self.val_mask
        val_set.random_aug = False
        val_set.symmetric_aug = False
        val_set.symmetric_aug_choice = []
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        raise NotImplementedError("rlbench_base_dataset의 get_normalizer 구현 안함(rlbench_dataset_list와 동일하게 만들면 됨)")

    def __len__(self) -> int:
        return len(self.sampler)

    def _add_symmetric_noise_to_pos(self, poses, axis, theta):
        new_poses = []
        assert theta <= np.pi # !! theta should be radians
        pose_noise = np.eye(4)

        # pose_noise[:3,:3] = cv2.Rodrigues(axis*theta)[0]
        pose_noise[:3,:3] = rodrigues(axis*theta)[0] # equavalent to cv2.Rodrigues(axis*theta)[0]

        for p in poses:
            pose = T.pose2mat((p[:3], p[3:]))
            pose_perturbed = pose @ pose_noise

            new_pose = np.concatenate(T.mat2pose(pose_perturbed), axis=0)
            new_poses.append(new_pose)
        return np.array(new_poses)

    def _add_random_noise_to_pos(self, poses, max_theta=np.radians(5), max_tran=1e-3):
        def random_direction():
            vec = np.random.randn(3).reshape(3)
            vec /= np.linalg.norm(vec)
            return vec
        
        max_trans = [max_tran for _ in range(3)]
        # !! theta in radians
        assert max_theta <= np.pi

        new_poses = []
        for p in poses:
            pose = T.pose2mat((p[:3], p[3:]))
          
            axis = random_direction()
            theta = np.random.uniform(0, max_theta)
            pose_noise = np.eye(4)
            pose_noise[:3,:3] = rodrigues(axis*theta)[0] # equavalent to cv2.Rodrigues(axis*theta)[0]
            pose_perturbed = pose @ pose_noise
            trans_noise = np.zeros((3))
            for ii in range(3):
                trans_noise[ii] = np.random.uniform(-max_trans[ii], max_trans[ii])
            pose_perturbed[:3,3] += trans_noise

            new_pose = np.concatenate(T.mat2pose(pose_perturbed), axis=0)
            new_poses.append(new_pose)
        return np.array(new_poses)

    def _sample_to_data(self, sample, aug_idx):
        if self.random_aug or self.symmetric_aug:
            obj0_anchor_pos = sample['obj0_state_in_anchor'][:,].astype(np.float32)
            obj0_anchor_action = sample['obj0_anchor_action'].astype(np.float32)
            obj0_anchor_pos_next = sample['obj0_state_next_in_anchor'][:,].astype(np.float32)

            obj1_anchor_pos = sample['obj1_state_in_anchor'][:,].astype(np.float32)
            obj1_anchor_action = sample['obj1_anchor_action'].astype(np.float32)
            obj1_anchor_pos_next = sample['obj1_state_next_in_anchor'][:,].astype(np.float32)


            if self.symmetric_aug:
                # aug_param = self.symmetric_aug_choice[aug_idx]
                # # apply symmetric augmentation
                # noisy_agent_pos = self._add_symmetric_noise_to_pos(agent_pos, axis=aug_param[:3], theta=aug_param[3]).astype(np.float32)
                # noisy_agent_pos_next = self._add_symmetric_noise_to_pos(agent_pos_next, axis=aug_param[:3], theta=aug_param[3]).astype(np.float32)
                raise NotImplementedError("vtob 에서 symmetric_aug 구현 안함")

            else:
                # no symmetric augmentation
                # noisy_agent_pos = agent_pos
                # noisy_agent_pos_next = agent_pos_next
    
                noisy_obj0_anchor_pos = obj0_anchor_pos
                noisy_obj0_anchor_pos_next = obj0_anchor_pos_next
                noisy_obj1_anchor_pos = obj1_anchor_pos
                noisy_obj1_anchor_pos_next = obj1_anchor_pos_next


            if self.random_aug:
                # apply random noise augmentation
                # noisy_agent_pos2 = self._add_random_noise_to_pos(noisy_agent_pos).astype(np.float32)
                noisy_obj0_anchor_pos_2 = self._add_random_noise_to_pos(noisy_obj0_anchor_pos).astype(np.float32)
                noisy_obj1_anchor_pos_2 = self._add_random_noise_to_pos(noisy_obj1_anchor_pos).astype(np.float32)


            AUG_GOAL = True 
            if AUG_GOAL:
                # action_from_noisy_to_next = np.array([calculate_action(p1, p2) for p1, p2 in zip(noisy_agent_pos2, noisy_agent_pos_next)]).astype(np.float32)
                obj0_anchor_action_from_noisy_to_next = np.array([calculate_action(p1, p2) for p1, p2 in zip(noisy_obj0_anchor_pos_2, noisy_obj0_anchor_pos_next)]).astype(np.float32)
                obj1_anchor_action_from_noisy_to_next = np.array([calculate_action(p1, p2) for p1, p2 in zip(noisy_obj1_anchor_pos_2, noisy_obj1_anchor_pos_next)]).astype(np.float32)
            else:
                raise NotImplementedError("vtob 에서 AUG_GOAL 구현 안함")

            # agent_pos = noisy_agent_pos2
            obj0_anchor_pos = noisy_obj0_anchor_pos_2
            obj0_anchor_action = obj0_anchor_action_from_noisy_to_next
            obj1_anchor_pos = noisy_obj1_anchor_pos_2
            obj1_anchor_action = obj1_anchor_action_from_noisy_to_next
        else:
            raise NotImplementedError("vtob 에서 self.random_aug=False인 경우 구현 안함")

        progress_binary = sample['progress_binary'].astype(np.float32)
        progress = sample['progress'].astype(np.float32)
        lang_token_embs = self._lang_token_embs[f"{0}_{0}"]

        data = {
            'obs_anchor':{
                'obj0_anchor_pos': obj0_anchor_pos,
                'obj1_anchor_pos': obj1_anchor_pos,
                'lang_token_embs': lang_token_embs,
                'obj0_image': self.obj0_image,
                'obj1_image': self.obj1_image,
            },
            'obj0_anchor_action': obj0_anchor_action,
            'obj1_anchor_action': obj1_anchor_action,
            'progress_binary': progress_binary,
            'progress': progress,
        }

        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample, aug_idx = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample, aug_idx)
        torch_data = dict_apply(data, torch.from_numpy)

        return torch_data
    


class RealDualBaseDataset(BaseDataset):
    def __init__(self,
            root_dir, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            has_lang_emb=False,
            random_aug=True,
            symmetric_axis=None,
            symmetric_theta_start=-1,
            symmetric_theta_end=-1,
            symmetric_theta_step=-1,
            ):
        super().__init__()
        
        # [핵심 수정] 변수들을 로직 실행 전에 미리 저장합니다.
        self.val_ratio = val_ratio
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.root_dir = root_dir
        self.random_aug = random_aug
        self.symmetric_aug = symmetric_axis is not None
        if 'pt' in self.root_dir:
            zarr_path = os.path.join(self.root_dir, 'zarr_pt')
        else:
            zarr_path = os.path.join(self.root_dir, 'zarr')
        cprint(f"[Dataset] zarr_path: {zarr_path}", "yellow")
        self.replay_buffer = self.get_replay_buffer(zarr_path)
        self.replay_buffer_val = None
        # (debug only) Dummy replay buffer
        # self.replay_buffer = ReplayBuffer.create_dummy()

        train_mask, val_mask = self.get_train_val_mask()
        if train_mask is not None:
            train_mask = downsample_mask(
                mask=train_mask, 
                max_n=max_train_episodes, 
                seed=seed)
        

        # applying symetric augmentation
        self.symmetric_aug_choice = []
        if symmetric_axis is not None:
            symmetric_axis = np.array(symmetric_axis).reshape(-1, 3)
            symmetric_theta_start = np.array(symmetric_axis).reshape(-1, 1)
            symmetric_theta_end = np.array(symmetric_theta_end).reshape(-1, 1)
            symmetric_theta_step = np.array(symmetric_theta_step).reshape(-1, 1)
            for idx, axis in enumerate(symmetric_axis):
                cur_theta_list = np.radians(np.arange(start=symmetric_theta_start[idx], stop=symmetric_theta_end[idx]+1, step=symmetric_theta_step[idx]))
                for theta in cur_theta_list:
                    self.symmetric_aug_choice.append(np.concatenate([axis, theta.reshape(-1)]))

        self.sampler = self.get_sampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            n_aug=len(self.symmetric_aug_choice) if len(self.symmetric_aug_choice) > 0 else -1,
        )
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # data augmentation
        self.random_aug = random_aug
        self.symmetric_aug = symmetric_axis is not None

        cprint(f"[Dataset] random_aug: {random_aug}", "yellow")
        cprint(f"[Dataset] symmetric_aug: {self.symmetric_aug}", "yellow")

        # language embeding
        self.has_lang_emb = has_lang_emb
        self._lang_token_embs = self.get_language_embedding()
        cprint(f"[Dataset] has_lang_emb: {has_lang_emb}", "yellow")

        self.obj0_image, self.obj1_image = self.get_grasp_image()

    def get_grasp_image(self):
        obj0_image_path = os.path.join(self.root_dir, 'r3d', 'episode0', 'crop_512_object0', '0.png')
        obj1_image_path = os.path.join(self.root_dir, 'r3d', 'episode0', 'crop_512_object1', '0.png')
        
        assert os.path.isfile(obj0_image_path), f'{obj0_image_path} There is no image of obj0'
        assert os.path.isfile(obj1_image_path), f'{obj1_image_path} There is no image of obj1'
        
        # 1. 이미지 로드 및 리사이즈
        obj0_image = cv2.resize(cv2.imread(obj0_image_path), (224, 224), interpolation=cv2.INTER_AREA)
        obj1_image = cv2.resize(cv2.imread(obj1_image_path), (224, 224), interpolation=cv2.INTER_AREA)

        # 2. BGR -> RGB 색상 공간 변환
        obj0_image_rgb = cv2.cvtColor(obj0_image, cv2.COLOR_BGR2RGB)
        obj1_image_rgb = cv2.cvtColor(obj1_image, cv2.COLOR_BGR2RGB)

        # 3. (H, W, C) -> (C, H, W) 차원 순서 변경
        # (224, 224, 3) -> (3, 224, 224)
        obj0_image_chw = obj0_image_rgb.transpose(2, 0, 1)
        obj1_image_chw = obj1_image_rgb.transpose(2, 0, 1)

        obj0_image_chw_normalized = obj0_image_chw.astype(np.float32) / 255.0
        obj1_image_chw_normalized = obj1_image_chw.astype(np.float32) / 255.0

        return obj0_image_chw_normalized, obj1_image_chw_normalized

    def get_replay_buffer(self, zarr_path):
        return ReplayBuffer.copy_from_path(
                    zarr_path, keys=['obj0_anchor_action', 'obj0_state_in_anchor', 'obj0_state_next_in_anchor',
                                     'obj1_anchor_action', 'obj1_state_in_anchor', 'obj1_state_next_in_anchor',
                                    #  'obj0_to_obj1_state', 'obj1_to_obj0_state',
                                     'progress','progress_binary','variation'])

    def get_sampler(self, replay_buffer, sequence_length, pad_before, pad_after, episode_mask, n_aug=-1):
        return SequenceSampler(
                    replay_buffer=replay_buffer, 
                    sequence_length=sequence_length,
                    pad_before=pad_before, 
                    pad_after=pad_after,
                    episode_mask=episode_mask, 
                    n_aug=n_aug,
                )

    def get_language_embedding(self):
        # load clip model
        model, _ = load_clip("RN50", jit=False, device='cuda')
        clip_model = build_model(model.state_dict())
        clip_model.to('cuda')
        del model

        # load all descriptions                
        description_path = os.path.join(self.root_dir,'r3d', "all_variation_descriptions.pkl")
        print(f"Loading demo from the path {description_path}")
        with open(description_path, "rb") as fin:
            descriptions = pickle.load(fin) # dict of VAR: DESC

        cur_task_id = 0 # cur_task_id is always zero as there is only one task

        # pre-generate all language embeddings
        _lang_token_embs = {}
        n_variation = len(descriptions)
        for var_id in range(n_variation):
            cur_description = descriptions[var_id]

            tokens = tokenize(cur_description).numpy()
            token_tensor = torch.from_numpy(tokens).to('cuda')
            sentence_emb, token_embs = clip_model.encode_text_with_embeddings(token_tensor)

            # print(cur_description)      # 5 sentence
            # print(token_embs.shape)     # [5, 77, 512]
            # print(sentence_emb.shape)   # [5, 1024]

            # obs_dict["lang_goal_emb"] = sentence_emb[0].float().detach().cpu().numpy()
            # lang_token_embs = token_embs[0].float().detach().cpu().numpy()
            lang_token_embs = sentence_emb[0].float().detach().cpu().numpy()[None, :]

            _lang_token_embs[f"{cur_task_id}_{var_id}"] = lang_token_embs
        return _lang_token_embs

    def get_train_val_mask(self):
        # 전체 데이터 개수 확인
        n_episodes = self.replay_buffer.n_episodes
        
        # val_ratio(예: 0.1) 만큼 개수 계산
        val_size = int(n_episodes * self.val_ratio)
        train_size = n_episodes - val_size
        
        # 마스크 초기화 (전부 False)
        train_mask = np.zeros(n_episodes, dtype=bool)
        
        # 앞부분을 Train으로 사용 (원하시면 뒷부분을 Train으로 해도 됩니다)
        train_mask[:train_size] = True
        
        # 나머지는 Validation
        val_mask = ~train_mask
        
        return train_mask, val_mask

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,      # <--- [수정] 원본 버퍼를 그대로 씁니다.
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask             # 마스크가 알아서 걸러줍니다.
            )
        val_set.train_mask = self.val_mask
        val_set.random_aug = False
        val_set.symmetric_aug = False
        val_set.symmetric_aug_choice = []
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        raise NotImplementedError("rlbench_base_dataset의 get_normalizer 구현 안함(rlbench_dataset_list와 동일하게 만들면 됨)")

    def __len__(self) -> int:
        return len(self.sampler)

    def _add_symmetric_noise_to_pos(self, poses, axis, theta):
        new_poses = []
        assert theta <= np.pi # !! theta should be radians
        pose_noise = np.eye(4)

        # pose_noise[:3,:3] = cv2.Rodrigues(axis*theta)[0]
        pose_noise[:3,:3] = rodrigues(axis*theta)[0] # equavalent to cv2.Rodrigues(axis*theta)[0]

        for p in poses:
            pose = T.pose2mat((p[:3], p[3:]))
            pose_perturbed = pose @ pose_noise

            new_pose = np.concatenate(T.mat2pose(pose_perturbed), axis=0)
            new_poses.append(new_pose)
        return np.array(new_poses)

    def _add_random_noise_to_pos(self, poses, max_theta=np.radians(5), max_tran=1e-3):
        def random_direction():
            vec = np.random.randn(3).reshape(3)
            vec /= np.linalg.norm(vec)
            return vec
        
        max_trans = [max_tran for _ in range(3)]
        # !! theta in radians
        assert max_theta <= np.pi

        new_poses = []
        for p in poses:
            pose = T.pose2mat((p[:3], p[3:]))
          
            axis = random_direction()
            theta = np.random.uniform(0, max_theta)
            pose_noise = np.eye(4)
            pose_noise[:3,:3] = rodrigues(axis*theta)[0] # equavalent to cv2.Rodrigues(axis*theta)[0]
            pose_perturbed = pose @ pose_noise
            trans_noise = np.zeros((3))
            for ii in range(3):
                trans_noise[ii] = np.random.uniform(-max_trans[ii], max_trans[ii])
            pose_perturbed[:3,3] += trans_noise

            new_pose = np.concatenate(T.mat2pose(pose_perturbed), axis=0)
            new_poses.append(new_pose)
        return np.array(new_poses)

    def _sample_to_data(self, sample, aug_idx):
        if self.random_aug or self.symmetric_aug:
            obj0_anchor_pos = sample['obj0_state_in_anchor'][:,].astype(np.float32)
            obj0_anchor_action = sample['obj0_anchor_action'].astype(np.float32)
            obj0_anchor_pos_next = sample['obj0_state_next_in_anchor'][:,].astype(np.float32)

            obj1_anchor_pos = sample['obj1_state_in_anchor'][:,].astype(np.float32)
            obj1_anchor_action = sample['obj1_anchor_action'].astype(np.float32)
            obj1_anchor_pos_next = sample['obj1_state_next_in_anchor'][:,].astype(np.float32)


            if self.symmetric_aug:
                # aug_param = self.symmetric_aug_choice[aug_idx]
                # # apply symmetric augmentation
                # noisy_agent_pos = self._add_symmetric_noise_to_pos(agent_pos, axis=aug_param[:3], theta=aug_param[3]).astype(np.float32)
                # noisy_agent_pos_next = self._add_symmetric_noise_to_pos(agent_pos_next, axis=aug_param[:3], theta=aug_param[3]).astype(np.float32)
                raise NotImplementedError("vtob 에서 symmetric_aug 구현 안함")

            else:
                # no symmetric augmentation
                # noisy_agent_pos = agent_pos
                # noisy_agent_pos_next = agent_pos_next
    
                noisy_obj0_anchor_pos = obj0_anchor_pos
                noisy_obj0_anchor_pos_next = obj0_anchor_pos_next
                noisy_obj1_anchor_pos = obj1_anchor_pos
                noisy_obj1_anchor_pos_next = obj1_anchor_pos_next


            if self.random_aug:
                # apply random noise augmentation
                # noisy_agent_pos2 = self._add_random_noise_to_pos(noisy_agent_pos).astype(np.float32)
                noisy_obj0_anchor_pos_2 = self._add_random_noise_to_pos(noisy_obj0_anchor_pos).astype(np.float32)
                noisy_obj1_anchor_pos_2 = self._add_random_noise_to_pos(noisy_obj1_anchor_pos).astype(np.float32)


            AUG_GOAL = True 
            if AUG_GOAL:
                # action_from_noisy_to_next = np.array([calculate_action(p1, p2) for p1, p2 in zip(noisy_agent_pos2, noisy_agent_pos_next)]).astype(np.float32)
                obj0_anchor_action_from_noisy_to_next = np.array([calculate_action(p1, p2) for p1, p2 in zip(noisy_obj0_anchor_pos_2, noisy_obj0_anchor_pos_next)]).astype(np.float32)
                obj1_anchor_action_from_noisy_to_next = np.array([calculate_action(p1, p2) for p1, p2 in zip(noisy_obj1_anchor_pos_2, noisy_obj1_anchor_pos_next)]).astype(np.float32)
            else:
                raise NotImplementedError("vtob 에서 AUG_GOAL 구현 안함")

            # agent_pos = noisy_agent_pos2
            obj0_anchor_pos = noisy_obj0_anchor_pos_2
            obj0_anchor_action = obj0_anchor_action_from_noisy_to_next
            obj1_anchor_pos = noisy_obj1_anchor_pos_2
            obj1_anchor_action = obj1_anchor_action_from_noisy_to_next
        else:
            raise NotImplementedError("vtob 에서 self.random_aug=False인 경우 구현 안함")

        progress_binary = sample['progress_binary'].astype(np.float32)
        progress = sample['progress'].astype(np.float32)
        lang_token_embs = self._lang_token_embs[f"{0}_{0}"]

        data = {
            'obs_anchor':{
                'obj0_anchor_pos': obj0_anchor_pos,
                'obj1_anchor_pos': obj1_anchor_pos,
                'lang_token_embs': lang_token_embs,
                'obj0_image': self.obj0_image,
                'obj1_image': self.obj1_image,
            },
            'obj0_anchor_action': obj0_anchor_action,
            'obj1_anchor_action': obj1_anchor_action,
            'progress_binary': progress_binary,
            'progress': progress,
        }

        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample, aug_idx = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample, aug_idx)
        torch_data = dict_apply(data, torch.from_numpy)

        return torch_data


if __name__ == '__main__':
    # configure dataset
    dataset: BaseDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
    train_dataloader = DataLoader(dataset, **cfg.dataloader)
    normalizer = dataset.get_normalizer()

    # configure validation dataset
    val_dataset = dataset.get_validation_dataset()
    val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
