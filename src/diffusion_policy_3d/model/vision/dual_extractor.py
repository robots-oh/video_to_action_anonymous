import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from typing import Dict, List, Type
from termcolor import cprint

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules

class DualDP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 out_channel=0,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 ):
        super().__init__()
        # 실험 2때문 나중에 상대 좌표로 바꿔야함
        self.ojb0_state_key = 'obj0_anchor_pos'
        self.ojb1_state_key = 'obj1_anchor_pos'
        self.n_output_channels = out_channel
        
        self.state_shape = observation_space['agent_pos'] # obj0, obj1 모두 같은 동일 차원이니까 같은 키워드로 불러옴, rlbench_multi.yaml에 정의됨
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")

        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        obj0_output_dim = state_mlp_size[-1] * 8
        obj1_output_dim = obj0_output_dim
        obj0_to_obj1_pos_output_dim = obj0_output_dim
        obj1_to_obj0_pos_output_dim = obj0_output_dim

        self.n_output_channels += obj0_output_dim 
        self.n_output_channels += obj1_output_dim 
        self.n_output_channels += obj0_to_obj1_pos_output_dim 
        self.n_output_channels += obj1_to_obj0_pos_output_dim 
        
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], obj1_output_dim, net_arch, state_mlp_activation_fn))
        self.relative_mlp = nn.Sequential(*create_mlp(self.state_shape[0]+1, obj0_to_obj1_pos_output_dim, net_arch, state_mlp_activation_fn))

        #--------------------------Image Encoding-------------------
        self.obj0_image_key = 'obj0_image' # Dataset에서 반환하는 이미지의 키 이름
        self.obj1_image_key = 'obj1_image' # Dataset에서 반환하는 이미지의 키 이름
        
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 마지막 분류 레이어를 제거 (출력 차원: 512)
        resnet.fc = nn.Identity()
        self.image_encoder = resnet

        # ResNet 가중치는 훈련하지 않도록 설정 (Frozen)
        self.image_encoder.eval()
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        # 최종 출력 채널 수에 이미지 특징 차원(512)을 더해줍니다.
        self.n_output_channels += 512 # obj0
        self.n_output_channels += 512 # obj1
        #--------------------------Image Encoding-------------------

        #--------------------------Language Encoding-------------------
        # CLIP language feature dimensions
        lang_feat_dim, lang_emb_dim, lang_max_seq_len = 1024, 512, 77
        lang_output_dim = 512
        # lang_output_dim = 256
        self.lang_preprocess = nn.Linear(lang_feat_dim, lang_output_dim)
        self.n_output_channels += lang_output_dim
        self.lang_key = "lang_token_embs"
        #--------------------------Language Encoding-------------------

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    # def forward(self, observations: Dict) -> torch.Tensor:
    def forward(self, obj0_observations: Dict, obj1_observations: Dict, obj0_to_obj1_pos: torch.Tensor, obj1_to_obj0_pos: torch.Tensor) -> torch.Tensor:
        obj0_state = obj0_observations[self.ojb0_state_key]
        obj1_state = obj1_observations[self.ojb1_state_key]

        obj0_state_feat = self.state_mlp(obj0_state)  # shape: (B*T, 64)
        obj1_state_feat = self.state_mlp(obj1_state)  # shape: (B*T, 64)

        dist = torch.norm(obj0_to_obj1_pos, p=2, dim=-1, keepdim=True)
        rel_input_01 = torch.cat([obj0_to_obj1_pos, dist], dim=-1)
        rel_input_10 = torch.cat([obj1_to_obj0_pos, dist], dim=-1)

        obj0_to_obj1_pos_feat = self.relative_mlp(rel_input_01)
        obj1_to_obj0_pos_feat = self.relative_mlp(rel_input_10)

        global_features = []

        obj0_image_tensor = obj0_observations[self.obj0_image_key]
        obj1_image_tensor = obj1_observations[self.obj1_image_key]

        obj0_image_feat = self.image_encoder(obj0_image_tensor)
        obj1_image_feat = self.image_encoder(obj1_image_tensor)
        global_features.append(obj0_image_feat)
        global_features.append(obj1_image_feat)

        lang_token_embs = obj0_observations[self.lang_key] # obj0_observations과 obj1_observations의 "lang_token_embs"는 동일한 값이 들어가 있다.
        lang_feat = self.lang_preprocess(lang_token_embs) # shape: (B, 512)
        global_features.append(lang_feat)

        if global_features:
            emb_feat = torch.cat(global_features, dim=-1)

            n_repeat = len(obj0_state_feat) // len(emb_feat) # 무슨 의미가 있지? len들은 batch size 로서 동일하게 나옴
            if n_repeat > 1:
                emb_feat = emb_feat.repeat(n_repeat, 1)

            final_feat = torch.cat((lang_feat, obj0_image_feat, obj0_state_feat, obj0_to_obj1_pos_feat, obj1_image_feat, obj1_state_feat, obj1_to_obj0_pos_feat), dim=-1)
            #final_feat = torch.cat((lang_feat, obj0_image_feat, obj0_state_feat, obj1_image_feat, obj1_state_feat), dim=-1)

        else:
            raise NotImplementedError("global_features없는 경우 구현 안함")
            
        return final_feat

    def output_shape(self):
        return self.n_output_channels