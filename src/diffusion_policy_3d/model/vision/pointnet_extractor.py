import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

import torchvision.models as models

from typing import Optional, Dict, Tuple, Union, List, Type
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




class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()

    


class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 use_lang_emb=False,
                 use_stage_emb=False,
                 use_image_cond=True,
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.pointcloud_encoder_cfg = pointcloud_encoder_cfg
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))


#-------------------------------------------------------------이미지 인코딩 부분-------------------
        self.use_image_cond = use_image_cond
        self.image_key = 'grasp_image' # Dataset에서 반환하는 이미지의 키 이름
        if self.use_image_cond:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # 마지막 분류 레이어를 제거 (출력 차원: 512)
            resnet.fc = nn.Identity()
            self.image_encoder = resnet

            # ResNet 가중치는 훈련하지 않도록 설정 (Frozen)
            self.image_encoder.eval()
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            
            # 최종 출력 채널 수에 이미지 특징 차원(512)을 더해줍니다.
            self.n_output_channels += 512
#-------------------------------------------------------------이미지 인코딩 부분-------------------


        self.use_lang_emb = use_lang_emb
        if self.use_lang_emb:
            # CLIP language feature dimensions
            lang_feat_dim, lang_emb_dim, lang_max_seq_len = 1024, 512, 77
            im_channels = 256 #64
            # self.lang_preprocess = nn.Linear(lang_emb_dim, im_channels * 2)
            self.lang_preprocess = nn.Linear(lang_feat_dim, im_channels * 2)
            self.n_output_channels += im_channels * 2
            self.lang_key = "lang_token_embs"

        self.use_stage_emb = use_stage_emb
        if self.use_stage_emb:
            self.n_output_channels += 3
            self.stage_key = "stage_embs"

        cprint(f"[DP3Encoder] use_lang_emb: {self.use_lang_emb}", "yellow")
        cprint(f"[DP3Encoder] use_stage_emb: {self.use_stage_emb}", "yellow")
        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    # def forward(self, observations: Dict) -> torch.Tensor:
    #     # points = observations[self.point_cloud_key]
    #     # assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
    #     # if self.use_imagined_robot:
    #     #     img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
    #     #     points = torch.concat([points, img_points], dim=1)
        
    #     # # points = torch.transpose(points, 1, 2)   # B * 3 * N
    #     # # points: B * 3 * (N + sum(Ni))
    #     # pn_feat = self.extractor(points)    # B * out_channel
            
    #     # state = observations[self.state_key]
    #     # state_feat = self.state_mlp(state)  # B * 64
    #     # final_feat = torch.cat([pn_feat, state_feat], dim=-1)

    #     state = observations[self.state_key]
    #     state_feat = self.state_mlp(state)  # B * 64

    #     if self.use_lang_emb or self.use_stage_emb:
    #         assert self.use_lang_emb and self.use_stage_emb # assume they were used at the same time

    #         # get lang feat
    #         lang_token_embs = observations[self.lang_key]
    #         lang_feat = self.lang_preprocess(lang_token_embs) # B * 128

    #         # get stage feat
    #         stage_embs = observations[self.stage_key]
    #         stage_feat = stage_embs

    #         emb_feat =  torch.cat((lang_feat, stage_feat), dim=-1)

    #         # repeat for all states (when horizon > 1)
    #         n_repeat = len(state) // len(emb_feat)
    #         emb_feat = emb_feat.repeat(n_repeat, 1)

    #         final_feat = torch.cat((emb_feat, state_feat), dim=-1) # B * (192+3)
    #     else:
    #         final_feat = state_feat
    #     return final_feat


    def forward(self, observations: Dict) -> torch.Tensor:
        # 1. state_feat는 항상 기본으로 추출합니다. (프레임별 정보)
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # shape: (B*T, 64)

        # 2. 전역적인(global) 특징들을 담을 리스트를 준비합니다.
        global_features = []

        # 3. 이미지 특징을 추출해서 리스트에 추가합니다. (새로 추가된 부분)
        if self.use_image_cond:
            image_tensor = observations[self.image_key]
            image_feat = self.image_encoder(image_tensor) # shape: (B, 512)
            global_features.append(image_feat)

        # 4. 언어 특징을 추출해서 리스트에 추가합니다. (기존 로직)
        if self.use_lang_emb:
            lang_token_embs = observations[self.lang_key]
            lang_feat = self.lang_preprocess(lang_token_embs) # shape: (B, 512)
            global_features.append(lang_feat)

        # 5. 스테이지 특징을 리스트에 추가합니다. (기존 로직)
        if self.use_stage_emb:
            stage_embs = observations[self.stage_key]
            stage_feat = stage_embs # shape: (B, 3)
            global_features.append(stage_feat)

        # 6. 전역 특징들이 하나라도 있다면, state_feat와 합쳐줍니다.
        if global_features:
            # 모든 전역 특징들을 먼저 하나로 합칩니다.
            emb_feat = torch.cat(global_features, dim=-1)

            # state_feat의 길이에 맞게 emb_feat를 복제(repeat)합니다.
            n_repeat = len(state_feat) // len(emb_feat)
            if n_repeat > 1:
                emb_feat = emb_feat.repeat(n_repeat, 1)

            # 최종적으로 state_feat와 복제된 emb_feat를 합칩니다.
            final_feat = torch.cat((emb_feat, state_feat), dim=-1)
        else:
            # 전역 특징이 없으면 state_feat만 사용합니다.
            final_feat = state_feat
            
        return final_feat



    def output_shape(self):
        return self.n_output_channels - self.pointcloud_encoder_cfg.out_channels # remove point features