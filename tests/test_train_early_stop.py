import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class FakeRolloutManager:
    def __init__(self, stop_status_by_rollout):
        self.stop_status_by_rollout = stop_status_by_rollout
        self.last_status = {}
        self.generated_rollouts = []
        self.saved_rollouts = []
        self.eval_rollouts = []
        self.disposed = False
        self.offload_calls = 0
        self.onload_weights_calls = 0
        self.onload_kv_calls = 0
        self.generate = _RemoteMethod(self._generate)
        self.get_last_generate_status = _RemoteMethod(self._get_last_generate_status)
        self.save = _RemoteMethod(self._save)
        self.dispose = _RemoteMethod(self._dispose)
        self.eval = _RemoteMethod(self._eval)
        self.offload = _RemoteMethod(self._offload)
        self.onload_weights = _RemoteMethod(self._onload_weights)
        self.onload_kv = _RemoteMethod(self._onload_kv)
        self.check_weights = _RemoteMethod(lambda **kwargs: None)

    def _generate(self, rollout_id):
        self.generated_rollouts.append(rollout_id)
        self.last_status = dict(self.stop_status_by_rollout.get(rollout_id, {}))
        return f"rollout-data-{rollout_id}"

    def _get_last_generate_status(self):
        return dict(self.last_status)

    def _save(self, rollout_id):
        self.saved_rollouts.append(rollout_id)

    def _dispose(self):
        self.disposed = True

    def _eval(self, rollout_id):
        self.eval_rollouts.append(rollout_id)

    def _offload(self):
        self.offload_calls += 1

    def _onload_weights(self):
        self.onload_weights_calls += 1

    def _onload_kv(self):
        self.onload_kv_calls += 1


class FakeTrainModel:
    def __init__(self):
        self.train_rollouts = []
        self.save_calls = []
        self.update_weights_calls = 0
        self.offload_calls = 0
        self.clear_memory_calls = 0

    def async_train(self, rollout_id, rollout_data_ref):
        self.train_rollouts.append((rollout_id, rollout_data_ref))
        return None

    def save_model(self, rollout_id, force_sync=False):
        self.save_calls.append((rollout_id, force_sync))

    def update_weights(self):
        self.update_weights_calls += 1

    def offload(self):
        self.offload_calls += 1

    def clear_memory(self):
        self.clear_memory_calls += 1


def _load_training_script(monkeypatch, script_name: str, rollout_manager, actor_model, critic_model=None):
    fake_ray = types.ModuleType("ray")
    fake_ray.get = lambda value: value

    placement_group_module = types.ModuleType("slime.ray.placement_group")
    placement_group_module.create_placement_groups = lambda args: {"rollout": "fake-rollout-pg"}
    placement_group_module.create_rollout_manager = lambda args, pg: (rollout_manager, None)
    placement_group_module.create_training_models = lambda args, pgs, rm: (actor_model, critic_model)

    arguments_module = types.ModuleType("slime.utils.arguments")
    arguments_module.parse_args = lambda: None

    logging_utils_module = types.ModuleType("slime.utils.logging_utils")
    logging_utils_module.configure_logger = lambda: None
    logging_utils_module.init_tracking = lambda args: None

    misc_module = types.ModuleType("slime.utils.misc")

    def should_run_periodic_action(rollout_id, interval, num_rollout_per_epoch=None, num_rollout=None):
        if interval is None:
            return False
        if num_rollout is not None and rollout_id == num_rollout - 1:
            return True
        step = rollout_id + 1
        return (step % interval == 0) or (
            num_rollout_per_epoch is not None and step % num_rollout_per_epoch == 0
        )

    misc_module.should_run_periodic_action = should_run_periodic_action

    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "slime.ray.placement_group", placement_group_module)
    monkeypatch.setitem(sys.modules, "slime.utils.arguments", arguments_module)
    monkeypatch.setitem(sys.modules, "slime.utils.logging_utils", logging_utils_module)
    monkeypatch.setitem(sys.modules, "slime.utils.misc", misc_module)

    script_path = Path(__file__).resolve().parents[1] / f"{script_name}.py"
    module_name = f"test_{script_name}_module"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_args(**overrides):
    base = dict(
        start_rollout_id=0,
        num_rollout=4,
        save="/tmp/fake-save",
        save_interval=None,
        rollout_global_dataset=True,
        offload_rollout=False,
        offload_train=False,
        use_critic=False,
        num_critic_only_steps=0,
        check_weight_update_equal=False,
        eval_interval=None,
        skip_eval_before_train=False,
        colocate=False,
        update_weights_interval=100,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_sync_train_stops_after_full_batch_and_forces_save(monkeypatch):
    rollout_manager = FakeRolloutManager(
        {
            0: {"should_stop_after_training_batch": False},
            1: {"should_stop_after_training_batch": True, "reason": "done"},
        }
    )
    actor_model = FakeTrainModel()
    module = _load_training_script(monkeypatch, "train", rollout_manager, actor_model)

    module.train(_make_args())

    assert rollout_manager.generated_rollouts == [0, 1]
    assert actor_model.train_rollouts == [(0, "rollout-data-0"), (1, "rollout-data-1")]
    assert actor_model.update_weights_calls == 2
    assert actor_model.save_calls == [(1, True)]
    assert rollout_manager.saved_rollouts == [1]
    assert rollout_manager.disposed is True


def test_async_train_does_not_queue_extra_rollout_after_final_status(monkeypatch):
    rollout_manager = FakeRolloutManager(
        {
            0: {"should_stop_after_training_batch": False},
            1: {"should_stop_after_training_batch": True, "reason": "done"},
        }
    )
    actor_model = FakeTrainModel()
    module = _load_training_script(monkeypatch, "train_async", rollout_manager, actor_model)

    module.train(_make_args())

    assert rollout_manager.generated_rollouts == [0, 1]
    assert actor_model.train_rollouts == [(0, "rollout-data-0"), (1, "rollout-data-1")]
    assert actor_model.update_weights_calls == 1
    assert actor_model.save_calls == [(1, True)]
    assert rollout_manager.saved_rollouts == [1]
    assert rollout_manager.disposed is True
