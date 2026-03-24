import time

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.ray.pipeline_rl_controller import PipelineRLCoordinator, get_pipeline_rl_step_lead
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.misc import should_run_periodic_action


def _should_force_final_save(args, rollout_id: int, stop_status: dict | None = None) -> bool:
    should_stop_after_training_batch = bool((stop_status or {}).get("should_stop_after_training_batch", False))
    return should_stop_after_training_batch or rollout_id == args.num_rollout - 1


def _maybe_create_pipeline_rl_controller(args):
    if getattr(args, "pipeline_rl_k", None) is None:
        return None
    return PipelineRLCoordinator.remote()


def _wait_for_pipeline_rl_slot(args, pipeline_rl_controller, *, target_trainer_step: int) -> None:
    if pipeline_rl_controller is None or getattr(args, "pipeline_rl_k", None) is None:
        return

    wait_start = time.time()
    last_log_time = wait_start

    while True:
        allowed, status = ray.get(
            pipeline_rl_controller.is_update_allowed.remote(
                target_trainer_step=target_trainer_step,
                max_step_lead=args.pipeline_rl_k,
            )
        )
        if allowed:
            return

        now = time.time()
        if now - last_log_time >= 5.0:
            projected_step_lead = get_pipeline_rl_step_lead(
                trainer_step=target_trainer_step,
                generation_step=status["generator_step"],
            )
            print(
                "PipelineRL-k waiting: "
                f"target_trainer_step={target_trainer_step}, "
                f"trainer_step={status['trainer_step']}, "
                f"generator_step={status['generator_step']}, "
                f"current_step_lead={status['current_step_lead']}, "
                f"projected_step_lead={projected_step_lead}/{args.pipeline_rl_k}, "
                f"waited={now - wait_start:.1f}s",
                flush=True,
            )
            last_log_time = now
        time.sleep(0.05)


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)
    pipeline_rl_controller = _maybe_create_pipeline_rl_controller(args)
    if pipeline_rl_controller is not None:
        args.pipeline_rl_controller = pipeline_rl_controller

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()
    pipeline_trainer_step = 1
    if pipeline_rl_controller is not None:
        ray.get(pipeline_rl_controller.set_trainer_step.remote(pipeline_trainer_step))

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    def save(rollout_id, *, force_sync: bool = False):
        if args.save is None:
            return
        should_force_sync = force_sync or rollout_id == args.num_rollout - 1
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
            actor_model.save_model(rollout_id, force_sync=should_force_sync)
        if args.use_critic:
            critic_model.save_model(rollout_id, force_sync=should_force_sync)
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)
            stop_status = ray.get(rollout_manager.get_last_generate_status.remote())
        else:
            rollout_data_curr_ref = None
            stop_status = {}

        # Start the next rollout early.
        if (not stop_status.get("should_stop_after_training_batch")) and rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)
        else:
            rollout_data_next_future = None

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

        saved_this_rollout = False
        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id, force_sync=_should_force_final_save(args, rollout_id, stop_status))
            saved_this_rollout = True

        if stop_status.get("should_stop_after_training_batch"):
            if not saved_this_rollout:
                save(rollout_id, force_sync=True)
            break

        if (rollout_id + 1) % args.update_weights_interval == 0:
            target_trainer_step = pipeline_trainer_step + 1
            _wait_for_pipeline_rl_slot(args, pipeline_rl_controller, target_trainer_step=target_trainer_step)
            actor_model.update_weights()
            pipeline_trainer_step = target_trainer_step
            if pipeline_rl_controller is not None:
                ray.get(pipeline_rl_controller.set_trainer_step.remote(pipeline_trainer_step))

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
