"""
Script for evaluating LSTM on simulated opinions.

Fusion type and parameter can be chosen manually by uncommenting the desired type.
Scenarios shown in the publication are 'transition_jump' and 'single_jump' using
the AvgDoC ST reset strategy for both.

Fig (4): variance=3.0, WEIGHTED, HANDLE_ST_CONFLICT=True, AVG_DC_CONFLICT_HANDLING=True, scenario="transition_jump"
Fig (5): variance=3.0, WEIGHTED, HANDLE_ST_CONFLICT=False, scenario="single_jump"
Fig (6): variance=3.0, WEIGHTED, HANDLE_ST_CONFLICT=True, AVG_DC_CONFLICT_HANDLING=True, scenario="transition_jump"
"""

import csv
import math
import matplotlib.pyplot as plt
import numba
import numpy as np
import subjective_logic as sl
import tqdm
import os
import os.path as osp

from util import generate_scenario


class EWMA:

    def __init__(self, alpha) -> None:
        self.alpha = alpha
        self.value = None

    def __call__(self, value):
        if isinstance(value, float):
            if self.value is None:
                self.value = value
            else:
                self.value = self.alpha * value + (1 - self.alpha) * self.value
        elif isinstance(value, sl.Opinion2d):
            if self.value is None:
                self.value = value
            else:
                belief = self.alpha * value.belief() + (1 - self.alpha) * self.value.belief()
                disbelief = self.alpha * value.disbelief() + (1 - self.alpha) * self.value.disbelief()
                self.value = sl.Opinion2d(belief, disbelief)
        else:
            raise NotImplementedError("Currently only float is supported")
        return self.value


@numba.njit(cache=True)
def dirichlet_multinomial_ll(alphas, evidence) -> float:
    alpha_0 = sum(alphas)
    n = sum(evidence)

    term_1 = math.lgamma(alpha_0) + math.lgamma(n + 1) - math.lgamma(n + alpha_0)
    term_2 = 0.0
    for a, e in zip(alphas, evidence):
        term_2 += math.lgamma(a + e) - math.lgamma(a) - math.lgamma(e + 1.0)
    log_likelihood = term_1 + term_2
    return log_likelihood


def monte_carlo_ll(long_op: sl.Opinion, short_op: sl.Opinion, num_simulations: int = 1000) -> None:

    dirichlet_lt, dirichlet_st = long_op.as_dirichlet(), short_op.as_dirichlet()
    # assuming non-informative prior W
    alphas_lt = np.array(dirichlet_lt.evidences + 1)
    evidence_st = np.array(dirichlet_st.evidences)
    log_likelihood = dirichlet_multinomial_ll(alphas_lt, evidence_st)

    simulated_probs = rng.dirichlet(alphas_lt, size=num_simulations)
    mc_log_likelihoods = np.zeros(num_simulations)
    for idx, ps in enumerate(simulated_probs):
        synthetic_counts = rng.multinomial(sum(evidence_st), ps)
        mc_log_likelihoods[idx] = dirichlet_multinomial_ll(alphas_lt, synthetic_counts)

    # count more extreme outcomes
    tail_events = np.sum(mc_log_likelihoods <= log_likelihood)
    pval = 1 - tail_events / num_simulations
    mc_sim.append(pval)


def post_hoc_conflict_detection(memory) -> None:

    long_size = memory.get_long_size()
    if long_size == 0 and not memory.is_last_conflicted():
        mc_sim.append(math.nan)
        dcs.append(math.nan)
        return

    if memory.is_last_conflicted():
        short_op, long_op = memory.get_conflicted_pair()
    else:
        short_op, long_op = memory.get_short_opinion(), memory.get_long_opinion()

    monte_carlo_ll(long_op, short_op)
    dcs.append(short_op.degree_of_conflict(long_op))


def sliding_window(arr: np.ndarray, window_size) -> np.ndarray:
    moving_sum = np.convolve(arr, np.ones(window_size), mode="full")[: len(arr)]
    # sliding window only contains single value at start
    divisors = np.minimum(np.arange(1, len(arr) + 1), window_size)
    return moving_sum / divisors


def plot(segments: list[dict[str, any]]) -> None:
    def plot_ppu(axis, data: np.array, **kwargs) -> None:
        axis.plot(data[:, 0], **kwargs)
        try:
            kwargs["label"] = f"{kwargs["label"]}-unc"
        except:
            pass
        axis.plot(data[:, 1], linestyle="--", **kwargs)

    nrows = 2
    nrows += 1 if len(mc_sim) > 0 else 0

    height_ratios = [1] + [20] * (nrows - 1)
    fig, axes = plt.subplots(nrows, 1, sharex=True, gridspec_kw={"height_ratios": height_ratios})

    cmap = {"good": "tab:green", "bad": "tab:red", "transition": "tab:orange"}

    for seg in segments:
        axes[0].axvspan(seg["start"], seg["end"], color=cmap[seg["label"]], alpha=0.9)
    axes[0].get_yaxis().set_visible(False)
    for spine in axes[0].spines.values():
        spine.set_visible(False)
    axes[0].tick_params(bottom=False)

    plot_ppu(axes[1], inp, color="tab:blue", label="input")
    plot_ppu(axes[1], buffer, color="tab:orange", label="buffer")
    plot_ppu(axes[1], ewma_ppu, color="tab:green", label="ewma")
    axes[1].plot(sliding_window_pp, color="tab:purple", label="window")
    axes[1].vlines(np.argwhere(resets), ymin=0.0, ymax=1.0, linestyles=":", color="tab:red")

    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()

    if len(mc_sim) > 0:
        axes[2].plot(mc_sim, label="MC Sim", color="tab:blue")
        axes[2].hlines(0.95, 0, len(mc_sim), linestyles="--", color="tab:blue")

    if len(dcs) > 0:
        axes[2].plot(dcs, label="Degree of Conflict", color="tab:orange")
        axes[2].hlines(THRESHOLD, 0, len(dcs), linestyles="--", color="tab:orange")
        axes[2].vlines(np.argwhere(resets), ymin=0.0, ymax=1.0, linestyles=":", color="tab:red")
        lines_dc, labels_dc = axes[2].get_legend_handles_labels()

    if len(mc_sim) > 0:
        axes[2].legend()

    plt.show()


def write_csv(base_path="/tmp") -> None:
    os.makedirs(base_path, exist_ok=True)
    headers = ["inp-pp", "inp-u", "buffer-pp", "buffer-u", "resets", "dc", "pval", "ewma-pp", "ewma-u", "window-pp"]
    np_arrs_to_write = [inp, buffer, resets.astype(int), dcs, mc_sim, ewma_ppu]
    cols = []

    for arr in np_arrs_to_write:
        if type(arr) == list or len(arr.shape) == 1:
            cols.append(arr.tolist() if type(arr) != list else arr)
        else:
            for idx in range(arr.shape[1]):
                cols.append(arr[:, idx])

    ftype = str(FUSION_TYPE).split(".")[-1].lower()
    reset_strategy = "_RS" if HANDLE_ST_CONFLICT else ""
    fname = f"ltst_{ftype}{reset_strategy}.csv"
    path = osp.join(base_path, fname)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in zip(*cols):
            writer.writerow(row)
    print(f"Successfully written results to {path}")


def run_simulation(ops: list[sl.Opinion], ltst: sl.LongShortTermMemory) -> None:
    for idx, op in tqdm.tqdm(enumerate(ops), total=len(ops)):
        ltst.add(op)
        op_buffer = ltst.get_opinion()

        inp[idx, :] = (op.getBinomialProjection(), op.uncertainty())
        buffer[idx, :] = (op_buffer.getBinomialProjection(), op_buffer.uncertainty())
        resets[idx] = ltst.is_last_conflicted()
        post_hoc_conflict_detection(ltst)

        ewma_op = ewma(op)
        ewma_ppu[idx, :] = (ewma_op.getBinomialProjection(), ewma_op.uncertainty())

    sliding_window_pp[:] = sliding_window(inp[:, 0], window_size=int(2 / (1 - DISCOUNT) - 1))


SEED = 42
rng = np.random.default_rng(seed=SEED)

GOOD_OP = sl.Opinion(0.8, 0.0)
BAD_OP = sl.Opinion(0.0, 0.8)

variance = 3.0
scenarios = {}
scenarios["single_jump"] = [("bad", 25, variance), ("good", 25, variance)]
scenarios["double_jump"] = [("good", 100, variance), ("bad", 100, variance), ("good", 100, variance)]
scenarios["double_transition"] = [
    ("good", 100, variance),
    ("transition", 50, variance),
    ("bad", 100, variance),
    ("transition", 50, variance),
    ("good", 100, variance),
]
scenarios["transition_jump"] = [
    ("good", 100, variance),
    ("transition", 50, variance),
    ("bad", 100, variance),
    ("good", 100, variance),
]
scenarios["jump_transition"] = [
    ("good", 100, variance),
    ("bad", 100, variance),
    ("transition", 50, variance),
    ("good", 100, variance),
]


WRITE_CSV = True
EXPORT_ALL = False
SHORT_WINDOW_SIZE = 10
THRESHOLD = 0.2
DISCOUNT = 0.9

# FUSION_TYPE = sl.FusionType.AVERAGE
# FUSION_TYPE = sl.FusionType.CUMULATIVE
FUSION_TYPE = sl.FusionType.WEIGHTED

HANDLE_ST_CONFLICT = True
AVG_DC_CONFLICT_HANDLING = True

ops, segments = generate_scenario(scenarios["transition_jump"], GOOD_OP, BAD_OP, rng)

inp = np.zeros((len(ops), 2))
buffer = np.zeros((len(ops), 2))
resets = np.zeros(len(ops))
mc_sim, dcs = [], []
ewma_ppu = np.zeros((len(ops), 2))
sliding_window_pp = np.zeros(len(ops))

ltst = sl.LongShortTermMemory2d(
    SHORT_WINDOW_SIZE, THRESHOLD, DISCOUNT, FUSION_TYPE, HANDLE_ST_CONFLICT, AVG_DC_CONFLICT_HANDLING
)
ewma = EWMA(1 - DISCOUNT)

run_simulation(ops, ltst)
plot(segments)
if WRITE_CSV:
    write_csv()

if EXPORT_ALL:
    for scenario, params in scenarios.items():
        rng = np.random.default_rng(seed=SEED)
        ops, segments = generate_scenario(params, GOOD_OP, BAD_OP, rng)
        for ftype in [sl.FusionType.AVERAGE, sl.FusionType.CUMULATIVE, sl.FusionType.WEIGHTED]:
            for st_conflict in range(2):
                HANDLE_ST_CONFLICT = bool(st_conflict)
                FUSION_TYPE = ftype
                inp = np.zeros((len(ops), 2))
                buffer = np.zeros((len(ops), 2))
                resets = np.zeros(len(ops))
                mc_sim, dcs = [], []

                ltst = sl.LongShortTermMemory2d(
                    SHORT_WINDOW_SIZE, THRESHOLD, DISCOUNT, FUSION_TYPE, HANDLE_ST_CONFLICT, AVG_DC_CONFLICT_HANDLING
                )
                run_simulation(ops, ltst)
                write_csv(base_path=osp.join("/tmp", scenario))
