from collections import defaultdict
from typing import List, Tuple
import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from rlbench.backend.conditions import DetectedSeveralCondition
from rlbench.backend.task import BimanualTask
from rlbench.backend.spawn_boundary import SpawnBoundary

class PourWaterToCup(BimanualTask):

    def init_task(self) -> None:
        
        self.cup = Shape('cup')
        self.pot = Shape('pot')
        self.register_graspable_objects([self.cup, self.pot])

        self.water = [Shape(f'waterparticle{i}') for i in range(6)]
        success_sensor = ProximitySensor('success')
        self.register_success_conditions(
            [DetectedSeveralCondition(self.water, success_sensor, 6)])

        self.waypoint_mapping = defaultdict(lambda: 'left')
        self.waypoint_mapping.update({'waypoint1': 'right', 'waypoint3': 'right', 'waypoint5': 'right', 'waypoint6': 'right'})

        self.left_boundary = Shape('left_cup_boundary')
        self.right_boundary = Shape('right_cup_boundary')


    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        b_left = SpawnBoundary([self.left_boundary])
        b_right = SpawnBoundary([self.right_boundary])

        b_left.sample(self.pot, min_distance=0.1, max_rotation=(0, 0, 0), min_rotation=(0, 0, 0))
        b_right.sample(self.cup, min_distance=0.1, max_rotation=(0, 0, 0), min_rotation=(0, 0, 0))
        return ['Pour water to cup']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 12], [0, 0, np.pi / 12]