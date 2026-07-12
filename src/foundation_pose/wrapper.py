import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import trimesh
from foundation_pose.estimater import ScorePredictor, PoseRefinePredictor, FoundationPose
from foundation_pose.Utils import draw_posed_3d_box, draw_xyz_axis, trimesh_add_pure_colored_texture


class FoundationPoseWrapper:
    def __init__(self, mesh_dir, debug_dir=None) -> None:
        # load object mesh
        self.debug_dir = "./debug" #debug_dir
        self.mesh_dir = mesh_dir
        self.mesh = None

        self.grasp_obj_name = None
        self.cur_grasp_obj_name = None
    
    def update_grasp_obj_name(self, obj_name):
        self.grasp_obj_name = obj_name


    def load_mesh(self):
        assert self.grasp_obj_name is not None
        mesh_path = os.path.join(self.mesh_dir, self.grasp_obj_name + ".obj")
        print(f"Loading mesh from: {mesh_path}")

        # 1. 기본 로드 (process=False를 하면 원본 데이터 보존에 유리할 수 있음)
        mesh = trimesh.load(mesh_path, force='mesh', process=False)

        # 2. 텍스처 로드 확인 및 예외 처리
        has_texture = False
        
        # 시각적 속성이 있고, 재질이 있으며, 그 재질에 이미지가 포함된 경우
        if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
             if hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None:
                has_texture = True
                print("Texture loaded successfully.")
        
        # 3. 텍스처가 없는 경우에만 Vertex Color 사용 (기존 로직 유지)
        if not has_texture:
            print("No texture image found. Falling back to vertex colors.")
            # 텍스처가 없으면 명시적으로 재질을 건너뛰고 로드하여 Vertex Color를 사용
            mesh = trimesh.load(mesh_path, force='mesh', skip_materials=True)

        # 4. 중심점 맞추기 (이전과 동일)
        mesh.vertices = mesh.vertices - np.mean(mesh.vertices, axis=0)

        self.mesh = mesh
        self.cur_grasp_obj_name = self.grasp_obj_name

    def create_estimator(self, debug_level=-1):
        # load mesh if mesh have not been loaded or grasp_obj_name changed
        if (self.mesh is None) or not (self.cur_grasp_obj_name == self.grasp_obj_name):
            self.load_mesh()

        debug_level = 0 if (self.debug_dir is None) or (debug_level < 0) else debug_level

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        return FoundationPose(
            model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals, mesh=self.mesh, 
            scorer=scorer, refiner=refiner, 
            debug_dir=self.debug_dir, debug=debug_level,
        )
    

class FoundationPoseWrapperReal:
    def __init__(self) -> None:
        # load object mesh
        self.debug_dir = "./debug" #debug_dir
        self.mesh = None

    def create_estimator(self, debug_level=-1):
        assert self.mesh is not None

        debug_level = 0 if (self.debug_dir is None) or (debug_level < 0) else debug_level

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        return FoundationPose(
            model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals, mesh=self.mesh, 
            scorer=scorer, refiner=refiner, 
            debug_dir=self.debug_dir, debug=debug_level,
        )