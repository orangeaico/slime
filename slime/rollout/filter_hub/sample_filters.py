from collections.abc import Sequence

from slime.rollout.scalerl import flatten_samples
from slime.utils.types import Sample

__all__ = ["mark_truncated_samples_inactive"]


def mark_truncated_samples_inactive(args, samples: Sequence[Sample] | Sequence[Sequence[Sample]], **kwargs) -> None:
    del args, kwargs

    for sample in flatten_samples(samples):
        if sample.status == Sample.Status.TRUNCATED:
            sample.remove_sample = True
