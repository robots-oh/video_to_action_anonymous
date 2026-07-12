# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

task_object_dict = {
    "vase_and_flower": {
        "grasp_object0_name": "flower",
        "grasp_object1_name": "vase",
        "visual_object0_name": "flower_visual",
        "visual_object1_name": "vase_visual"
    },

    "pour_water_to_cup": {
        "grasp_object0_name": "pot",
        "grasp_object1_name": "cup",
        "visual_object0_name": "pot_visual",
        "visual_object1_name": "cup_visual",
    },

    "pour_water_to_cup_real": {
        "grasp_object0_name": "cup",
        "grasp_object1_name": "pitcher",
        "visual_object0_name": "cup_visual",
        "visual_object1_name": "pitcher_visual",
    },
    "put_object_in_crate": {
        "grasp_object0_name": "toy",
        "grasp_object1_name": "crate",
        "visual_object0_name": "toy_visual",
        "visual_object1_name": "crate_visual"
    },

    "wipe_dish_with_sponge": {
        "grasp_object0_name": "sponge",
        "grasp_object1_name": "plate",
        "visual_object0_name": "sponge_visual",
        "visual_object1_name": "plate_visual"
    },
    "wipe_dish_with_sponge_real": {
        "grasp_object0_name": "sponge",
        "grasp_object1_name": "dish",
        "visual_object0_name": "sponge_visual",
        "visual_object1_name": "dish_visual"
    },
    "put_truck_toy_in_basket_real": {
        "grasp_object0_name": "basket",
        "grasp_object1_name": "trucktoy",
        "visual_object0_name": "basket_visual",
        "visual_object1_name": "trucktoy_visual"
    },

    "close_the_pot_lid": {
        "grasp_object0_name": "lid",
        "grasp_object1_name": "black_pot",
        "visual_object0_name": "lid_visual",
        "visual_object1_name": "black_pot_visual"
    },

    "cheers_coke_easy": {
        "grasp_object0_name": "coke0",
        "grasp_object1_name": "coke1",
        "visual_object0_name": "coke0_visual",
        "visual_object1_name": "coke1_visual"
    },

    "cheers_coke_hard": {
        "grasp_object0_name": "coke0",
        "grasp_object1_name": "coke1",
        "visual_object0_name": "coke0_visual",
        "visual_object1_name": "coke1_visual"
    },

    "ring_the_bell_with_mallet" :{
        "grasp_object0_name": "mallet",
        "grasp_object1_name": "bell",
        "visual_object0_name": "mallet_visual",
        "visual_object1_name": "bell_visual"
    },

    "place_the_figure": {
        "grasp_object0_name": "figure",
        "grasp_object1_name": "showcase",
        "visual_object0_name": "figure_visual",
        "visual_object1_name": "showcase_visual"

    },

    "plating_the_grilled_meat": {
        "grasp_object0_name": "meat",
        "grasp_object1_name": "pan",
        "visual_object0_name": "meat_visual",
        "visual_object1_name": "pan_visual"
    },

    "hang_cup_top": {
        "grasp_object0_name": "cup",
        "grasp_object1_name": "stand",
        "visual_object0_name": "cup_visual",
        "visual_object1_name": "stand_visual"
    },

    "scan_the_bottle": {
        "grasp_object0_name": "bottle",
        "grasp_object1_name": "scanner",
        "visual_object0_name": "bottle_visual",
        "visual_object1_name": "scanner_visual"
    },

    "screw_the_bottle_cap": {
        "grasp_object0_name": "bottlecap",
        "grasp_object1_name": "bottle",
        "visual_object0_name": "bottlecap_visual",
        "visual_object1_name": "bottle_visual"
    },
    "vase_and_flower_mirror": {
        "grasp_object0_name": "vase",
        "grasp_object1_name": "flower",
        "visual_object0_name": "vase_visual",
        "visual_object1_name": "flower_visual"
    },

    "pour_water_to_cup_mirror": {
        "grasp_object0_name": "cup",
        "grasp_object1_name": "pot",
        "visual_object0_name": "cup_visual",
        "visual_object1_name": "pot_visual"
    },

    "pour_water_to_cup_real_mirror": {
        "grasp_object0_name": "pitcher",
        "grasp_object1_name": "cup",
        "visual_object0_name": "pitcher_visual",
        "visual_object1_name": "cup_visual"
    },
    
    "put_object_in_crate_mirror": {
        "grasp_object0_name": "crate",
        "grasp_object1_name": "toy",
        "visual_object0_name": "crate_visual",
        "visual_object1_name": "toy_visual"
    },

    "wipe_dish_with_sponge_mirror": {
        "grasp_object0_name": "plate",
        "grasp_object1_name": "sponge",
        "visual_object0_name": "plate_visual",
        "visual_object1_name": "sponge_visual"
    },
    
    "wipe_dish_with_sponge_real_mirror": {
        "grasp_object0_name": "dish",
        "grasp_object1_name": "sponge",
        "visual_object0_name": "dish_visual",
        "visual_object1_name": "sponge_visual"
    },

    "close_the_pot_lid_mirror": {
        "grasp_object0_name": "black_pot",
        "grasp_object1_name": "lid",
        "visual_object0_name": "black_pot_visual",
        "visual_object1_name": "lid_visual"
    },

    "cheers_coke_easy_mirror": {
        "grasp_object0_name": "coke1",
        "grasp_object1_name": "coke0",
        "visual_object0_name": "coke1_visual",
        "visual_object1_name": "coke0_visual"
    },

    "cheers_coke_hard_mirror": {
        "grasp_object0_name": "coke1",
        "grasp_object1_name": "coke0",
        "visual_object0_name": "coke1_visual",
        "visual_object1_name": "coke0_visual"
    },

    "ring_the_bell_with_mallet_mirror" :{
        "grasp_object0_name": "bell",
        "grasp_object1_name": "mallet",
        "visual_object0_name": "bell_visual",
        "visual_object1_name": "mallet_visual"
    },

    "place_the_figure_mirror": {
        "grasp_object0_name": "showcase",
        "grasp_object1_name": "figure",
        "visual_object0_name": "showcase_visual",
        "visual_object1_name": "figure_visual"
    },

    "plating_the_grilled_meat_mirror": {
        "grasp_object0_name": "pan",
        "grasp_object1_name": "meat",
        "visual_object0_name": "pan_visual",
        "visual_object1_name": "meat_visual"
    },

    "hang_cup_top_mirror": {
        "grasp_object0_name": "stand",
        "grasp_object1_name": "cup",
        "visual_object0_name": "stand_visual",
        "visual_object1_name": "cup_visual"
    },

    "scan_the_bottle_mirror": {
        "grasp_object0_name": "scanner",
        "grasp_object1_name": "bottle",
        "visual_object0_name": "scanner_visual",
        "visual_object1_name": "bottle_visual"
    },

    "screw_the_bottle_cap_mirror": {
        "grasp_object0_name": "bottle",
        "grasp_object1_name": "bottlecap",
        "visual_object0_name": "bottle_visual",
        "visual_object1_name": "bottlecap_visual"
    }


}