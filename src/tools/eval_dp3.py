if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent) # root directory : object_centric_diffusion
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
from omegaconf import OmegaConf
import pathlib
from train_dp3_vtob import TrainDP3Workspace

OmegaConf.register_new_resolver("eval", eval, replace=True)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).resolve().parent.parent.joinpath('config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.eval()

if __name__ == "__main__":
    main()
