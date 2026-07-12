from diffusion_policy_3d.dataset.rlbench_base_dataset import RLBenchBaseDataset, RLBenchDualBaseDataset, RealDualBaseDataset
from diffusion_policy_3d.common.pytorch_util import dict_apply
import os
import glob
from PIL import Image
import torch
import torchvision.transforms as T
from typing import Dict


class RLBenchDataset(RLBenchBaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    ######################모든 이미지 경로 추가###############
        self.image_paths = list()
        self.transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        
        ])

        episode_ends = self.replay_buffer.meta['episode_ends']
        episode_dirs = sorted(
            glob.glob(os.path.join(self.root_dir, 'episodes', 'episode*')),
            key=lambda x: int(os.path.basename(x).replace('episode', ''))
        )

        episode_idx = 0
        for sample_idx in range(len(self.replay_buffer['state'])):
            if sample_idx >= episode_ends[episode_idx]:
                episode_idx += 1
            
            if episode_idx < len(episode_dirs):
                img_path = os.path.join(episode_dirs[episode_idx], 'grasp_obj_rgb', '0.png')
                self.image_paths.append(img_path)
            else:
                # 만약 에피소드 디렉토리 개수가 부족하면, 마지막 경로를 반복 사용하거나 에러 처리를 할 수 있습니다.
                # 여기서는 마지막 경로를 반복해서 추가합니다.
                if len(self.image_paths) > 0:
                    self.image_paths.append(self.image_paths[-1]) 
                else: # 첫 에피소드부터 없는 경우
                    self.image_paths.append(None)
        
        assert len(self.image_paths) == len(self.replay_buffer['state'])
    ######################이미지 #############################################

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 1. DataLoader가 준 '뻥튀기된 인덱스(idx)'를 '원본 시퀀스 인덱스'로 변환합니다.
        #    sampler가 원본 시퀀스의 개수(n_sequences)를 알고 있습니다.
        #    idx를 n_sequences로 나눈 나머지가 원본 시퀀스의 인덱스가 됩니다.
        original_sequence_idx = idx % len(self.sampler.indices)
        
        # 2. 이 원본 시퀀스 인덱스를 사용해, sampler가 가진 실제 데이터 시작 위치를 찾습니다.
        start_idx = self.sampler.indices[original_sequence_idx]

        # 3. 변환된 start_idx를 사용해 올바른 이미지 경로를 가져옵니다.
        img_path = self.image_paths[int(start_idx[0])]
        
        # 4. 이제 sampler에게 원래의 '뻥튀기된 인덱스(idx)'를 전달해
        #    올바른 데이터 시퀀스와 증강(augmentation) 정보를 가져옵니다.
        sample, aug_idx = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample, aug_idx)
        torch_data = dict_apply(data, torch.from_numpy)

        # 5. 이미지를 로딩하고 딕셔너리에 추가합니다.
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)
        except (FileNotFoundError, TypeError):
            print(f"Warning: Image not found or path is invalid ({img_path}), returning zeros.")
            image_tensor = torch.zeros((3, 224, 224))

        torch_data['obs']['grasp_image'] = image_tensor
        
        return torch_data
    

class RLBenchDualDataset(RLBenchDualBaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class RealDualDataset(RealDualBaseDataset):
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)