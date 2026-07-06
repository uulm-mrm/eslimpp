import subjective_logic as sl
import numpy as np


def _tag(ops_list: list[sl.Opinion], segments: list[dict[str, any]], label: str) -> list[sl.Opinion]:
    try:
        start = segments[-1]["end"]
    except:
        start = 0
    segments.append({"start": start, "end": start + len(ops_list), "label": label})
    return ops_list


def _generate_sl_opinions(target_op: sl.Opinion, variance: float, rng, num_samples: int = 1) -> list[sl.Opinion]:
    dirichlet_target = target_op.as_dirichlet()
    evidence_target = dirichlet_target.evidences

    noise = rng.normal(loc=0.0, scale=np.sqrt(variance), size=(num_samples, 2))

    return [
        sl.Opinion2d(sl.DirichletDistribution2d().from_evidences(np.maximum(evidence_target + n, 0.0))) for n in noise
    ]


def _transition(
    current_op: tuple[float, float, float],
    target_op: tuple[float, float, float],
    variance: float,
    rng,
    num_samples=25,
) -> list[sl.Opinion]:

    targets_b = np.linspace(current_op.belief(), target_op.belief(), num_samples)
    targets_d = np.linspace(current_op.disbelief(), target_op.disbelief(), num_samples)

    return [
        _generate_sl_opinions(sl.Opinion(b, d), variance=variance, num_samples=1, rng=rng)[0]
        for b, d in zip(targets_b, targets_d)
    ]


def generate_scenario(sequence: list[str, int, float], good_op: sl.Opinion, bad_op: sl.Opinion, rng) -> tuple[any, any]:

    ops, segments = [], []

    current_state = None
    for label, length, variance in sequence:
        if label == "good":
            ops.extend(
                _tag(
                    _generate_sl_opinions(target_op=good_op, variance=variance, num_samples=length, rng=rng),
                    segments,
                    label=label,
                )
            )
        elif label == "bad":
            ops.extend(
                _tag(
                    _generate_sl_opinions(target_op=bad_op, variance=variance, num_samples=length, rng=rng),
                    segments,
                    label=label,
                )
            )
            pass
        elif label == "transition":
            if current_state is None:
                raise ValueError("Cannot have transition at the start")
            current = good_op if current_state == "good" else bad_op
            target = bad_op if current_state == "good" else good_op
            ops.extend(
                _tag(
                    _transition(current_op=current, target_op=target, variance=variance, num_samples=length, rng=rng),
                    segments,
                    label="transition",
                )
            )
        else:
            raise ValueError(f"Unknown Label {label}")

        current_state = label

    return ops, segments
