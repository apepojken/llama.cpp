#!/usr/bin/env python3
"""Phase 0 analysis of the indexer's block selection (DESIGN.md §9), from the raw ids the server
dumps with LLAMA_QSA_TRACE (<file>.ids: records of int32 [layer, k, ids[k]], one per decoded token).
Reports, per layer and across layers, what a hot tier would have to hold and fetch:
  never_pct      blocks that existed at the start and were never selected (the evictable share)
  miss@W         mean share of a step's selected blocks NOT selected in the previous W steps
                 (= fetches per step if the hot tier is "everything used in the last W steps")
  hot@W          mean size of that hot set, in blocks (RAM it would pin: x 51 KiB per block)
  union          blocks selected at least once over the whole run (the working set's ceiling)
"""
import struct, sys, collections
import numpy as np

path = sys.argv[1]
raw = np.fromfile(path, dtype=np.int32)
steps = collections.defaultdict(list)          # layer -> list of np arrays (sorted unique ids)
i = 0
while i + 2 <= len(raw):
    il, k = int(raw[i]), int(raw[i+1]); i += 2
    ids = raw[i:i+k]; i += k
    if k < 256 or k > 1024:                    # warm-up / non-selection tensors
        continue
    steps[il].append(np.unique(ids[ids >= 0]))

WINDOWS = (1, 4, 16, 64, 256)
print(f"{'layer':>5} {'steps':>5} {'n_blk0':>6} {'union':>6} {'never%':>6} " +
      " ".join(f"miss@{w:<3} hot@{w:<4}" for w in WINDOWS))
cross = []                                      # per step: union over layers of selected ids
n_steps_min = min(len(v) for v in steps.values())
for il in sorted(steps):
    S = steps[il]
    n0 = int(S[0].max()) + 1                    # blocks that existed at the first step
    seen = np.zeros(max(int(s.max()) for s in S) + 1, dtype=np.int64)
    for s in S:
        seen[s] += 1
    never = 100.0 * np.sum(seen[:n0] == 0) / n0
    union = int(np.sum(seen > 0))
    cols = []
    for W in WINDOWS:
        last = {}                                # block id -> last step it was selected
        miss, hot = [], []
        for t, s in enumerate(S):
            if t > 0:
                m = sum(1 for b in s if (b not in last) or (t - last[b] > W))
                miss.append(m / len(s))
                hot.append(sum(1 for b, ts in last.items() if t - ts <= W))
            for b in s:
                last[int(b)] = t
        cols.append(f"{np.mean(miss):7.3f} {np.mean(hot):8.0f}")
    print(f"{il:>5} {len(S):>5} {n0:>6} {union:>6} {never:6.1f} " + " ".join(cols))

# the per-step working set across all layers (what one decode step touches in total)
for t in range(n_steps_min):
    u = set()
    for il in steps:
        u.update(steps[il][t].tolist())
    cross.append(len(u))
print(f"\ncross-layer per-step working set: mean {np.mean(cross):.0f} blocks "
      f"(x 51 KiB = {np.mean(cross)*51/1024:.0f} MiB), max {max(cross)}; "
      f"per layer {int(np.mean([np.mean([len(s) for s in steps[il]]) for il in steps]))} blocks")
