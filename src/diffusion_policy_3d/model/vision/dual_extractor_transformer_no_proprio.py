import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List
from termcolor import cprint

class DualTransformerEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 embedding_dim=256,
                 state_mlp_size=(64, 64), 
                 state_mlp_activation_fn=nn.ReLU,
                 use_rel_feat: bool = False,  # [수정 1] Ablation을 위한 플래그 추가
                 ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_rel_feat = use_rel_feat  # 저장
        
        # --- State Config ---
        self.ojb0_state_key = 'obj0_anchor_pos'
        self.ojb1_state_key = 'obj1_anchor_pos'
        self.state_shape = observation_space['agent_pos'] 
        
        # --- Image Config ---
        self.obj0_image_key = 'obj0_image'
        self.obj1_image_key = 'obj1_image'
        self.image_dim = 512 # ResNet34 output

        # --- Language Config ---
        self.lang_key = "lang_token_embs"
        self.lang_input_dim = 1024 # CLIP dimension

        # 1. State Projection
        self.state_proj = nn.Sequential(
            nn.Linear(self.state_shape[0], state_mlp_size[0]),
            state_mlp_activation_fn(),
            nn.Linear(state_mlp_size[0], embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # 2. Relative Position Projection
        # [수정 2] 플래그가 True일 때만 프로젝션 레이어 생성
        if self.use_rel_feat:
            self.rel_proj = nn.Sequential(
                nn.Linear(16, state_mlp_size[0]),
                state_mlp_activation_fn(),
                nn.Linear(state_mlp_size[0], embedding_dim),
                nn.LayerNorm(embedding_dim)
            )

        # 3. Image Projection
        self.image_encoder = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.image_encoder.fc = nn.Identity()
        self.image_encoder.eval() 
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_dim, embedding_dim), 
            nn.LayerNorm(embedding_dim)
        )

        # 4. Language Projection
        self.lang_proj = nn.Sequential(
            nn.Linear(self.lang_input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Token Type Embeddings
        # [수정 3] rel_feat 유무에 따라 토큰 개수를 6개 또는 5개로 유동적 할당
        self.num_tokens = 6 if self.use_rel_feat else 5
        self.token_type_embed = nn.Embedding(self.num_tokens, embedding_dim)

        cprint(f"[DualTransformerEncoder] Initialized. Target Embedding Dim: {embedding_dim} | use_rel_feat: {use_rel_feat}", "cyan")

    def forward(self, obj0_observations: Dict, obj1_observations: Dict, obj0_to_obj1_pos: torch.Tensor, obj1_to_obj0_pos: torch.Tensor) -> torch.Tensor:
        """
        Returns: (B_total, Num_Tokens, Embedding_Dim)
        """
        B_total = obj0_to_obj1_pos.shape[0]
        device = obj0_to_obj1_pos.device

        # --- 1. Language ---
        lang_emb = obj0_observations[self.lang_key]
        if lang_emb.shape[0] != B_total:
            ratio = B_total // lang_emb.shape[0]
            lang_emb = lang_emb.repeat_interleave(ratio, dim=0)
        lang_feat = self.lang_proj(lang_emb).unsqueeze(1)

        # --- 2. Images ---
        obj0_img = self.image_encoder(obj0_observations[self.obj0_image_key])
        obj1_img = self.image_encoder(obj1_observations[self.obj1_image_key])
        
        if obj0_img.shape[-1] != self.image_dim:
             raise RuntimeError(f"Image Encoder output dim is {obj0_img.shape[-1]}, but expected {self.image_dim}.")

        obj0_img_feat = self.image_proj(obj0_img).unsqueeze(1)
        obj1_img_feat = self.image_proj(obj1_img).unsqueeze(1)

        # --- 3. States ---
        obj0_state = obj0_observations[self.ojb0_state_key]
        obj1_state = obj1_observations[self.ojb1_state_key]
        if obj0_state.dim() == 3: 
            obj0_state = obj0_state[:, -1, :]
            obj1_state = obj1_state[:, -1, :]
            
        obj0_state_feat = self.state_proj(obj0_state).unsqueeze(1)
        obj1_state_feat = self.state_proj(obj1_state).unsqueeze(1)

        # --- 4. Relative (선택적 계산) ---
        # [수정 4] 토큰들을 리스트로 관리하여 조건부로 추가
        tokens_list = [
            lang_feat,       
            obj0_img_feat,   
            obj0_state_feat, 
            obj1_img_feat,   
            obj1_state_feat
        ]

        if self.use_rel_feat:
            dist01 = torch.norm(obj0_to_obj1_pos[..., :3], p=2, dim=-1, keepdim=True)
            dist10 = torch.norm(obj1_to_obj0_pos[..., :3], p=2, dim=-1, keepdim=True)
            
            rel_input = torch.cat([obj0_to_obj1_pos, dist01, obj1_to_obj0_pos, dist10], dim=-1)
            if rel_input.dim() == 3: rel_input = rel_input[:, -1, :]
            
            rel_feat = self.rel_proj(rel_input).unsqueeze(1)
            tokens_list.append(rel_feat)

        # [DEBUG] 차원 최종 확인
        if lang_feat.shape[0] != obj0_img_feat.shape[0]:
            print(f"!!! BATCH SIZE MISMATCH !!!")
            raise RuntimeError("Batch sizes do not match! Check repeat_interleave logic.")

        # --- 5. Combine Sequence ---
        seq_tokens = torch.cat(tokens_list, dim=1)

        # 위치 임베딩 추가 (동적으로 설정된 self.num_tokens 사용)
        semantic_pos = torch.arange(self.num_tokens, device=device).unsqueeze(0)
        seq_tokens = seq_tokens + self.token_type_embed(semantic_pos)

        return seq_tokens

    def output_shape(self):
        return self.embedding_dim