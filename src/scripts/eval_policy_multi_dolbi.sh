for rlbench_task_name in \
    pour_water_to_cup
    # hang_cup_top\
    # cheers_coke_easy\
    # put_object_in_crate\
    # scan_the_bottle\
    # close_the_pot_lid\
    # wipe_dish_with_sponge\
    # place_the_figure\
    # vase_and_flower\
    # screw_the_bottle_cap\
    
do
    DEBUG=True
    seed=1001

    alg_name=simple_dp3
    task_name=${rlbench_task_name}
    config_name=${alg_name}
    seed=${seed}
    exp_name="rlbench_multi"
    run_dir="data/outputs/${exp_name}_seed${seed}"

    gpu_id=0
    use_fp=False # if false use ground-truth object pose instead
    # use_fp=True # if false use ground-truth object pose instead
    eval_epoch=3000


    export HYDRA_FULL_ERROR=1
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    python /workspace/video_to_action/src/tools/eval_dp3.py --config-name=${config_name}.yaml \
                                task=rlbench/${task_name} \
                                hydra.run.dir=${run_dir} \
                                training.debug=$DEBUG \
                                training.seed=${seed} \
                                training.device="cuda:0" \
                                exp_name=${exp_name} \
                                logging.mode="disabled" \
                                checkpoint.save_ckpt=${save_ckpt} \
                                task.env_runner.use_fp=${use_fp} \
                                evaluation.eval_epoch=${eval_epoch}
done
