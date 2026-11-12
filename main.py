import bisect
from collections import defaultdict
import datetime
import json
import math
import os
import random
import sys


def parse_iso(ts):
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts / 1000.0 if ts > 1e11 else ts)
    if isinstance(ts, str):
        ts = ts.strip()
        if not ts:
            return 0.0
        try:
            f = float(ts)
            return float(f / 1000.0 if f > 1e11 else f)
        except ValueError:
            pass
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        try:
            dt = datetime.datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt.timestamp()
        except Exception:
            return 0.0
    return 0.0


def main():
    random.seed(42)

    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

    task_file = os.path.join(input_dir, "task.json")
    history_file = os.path.join(input_dir, "history.ndjson")
    out_file = os.path.join(output_dir, "forecast.json")

    if not os.path.exists(task_file):
        sys.exit(0)

    with open(task_file, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    reads = task_data.get("reads", [])
    edges = [round(2.00 + i * 0.05, 2) for i in range(81)]

    marks = []
    n_nonempty = 0
    all_history_mints = set()
    mint_history = defaultdict(list)
    mint_info = {}

    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                mark = json.loads(line)
                marks.append(mark)
                if mark.get("rows"):
                    n_nonempty += 1

    marks.sort(key=lambda m: parse_iso(m.get("at")))

    if not marks:
        sys.exit(1)

    for m_idx, mark in enumerate(marks):
        t_mark = parse_iso(mark.get("at"))
        rows = mark.get("rows") or []
        for row in rows:
            if not isinstance(row, dict):
                continue
            mint = row.get("mint")
            if not mint:
                continue
            all_history_mints.add(mint)
            cap_lamports = float(row.get("market_cap_lamports") or 0)
            cap_sol = cap_lamports / 1e9
            cap_log10 = math.log10(max(1.0, cap_sol))
            rank = int(row.get("rank") or 999)
            venue = str(row.get("venue") or "")
            complete = bool(row.get("complete") or False)
            created_at_val = row.get("created_at") or mark.get("at")
            created_at_ts = parse_iso(created_at_val)

            mint_history[mint].append((m_idx, t_mark, rank, cap_log10, venue, complete))

            mint_info[mint] = {
                "last_seen_mark": m_idx,
                "last_rank": rank,
                "last_cap": cap_log10,
                "last_venue": venue,
                "last_complete": complete,
                "created_at_ts": created_at_ts,
            }

    t_last = parse_iso(marks[-1].get("at"))
    num_marks = len(marks)
    horizons_min = [(parse_iso(r) - t_last) / 60.0 for r in reads]
    if not horizons_min:
        horizons_min = [1440.0]

    candidates = []
    for mint, info in mint_info.items():
        stale_marks = (num_marks - 1) - info["last_seen_mark"]
        hist = mint_history[mint]
        if len(hist) >= 2:
            dt_min = (hist[-1][1] - hist[-2][1]) / 60.0
            dt_units = max(0.5, dt_min / 10.0)
            momentum = (hist[-1][3] - hist[-2][3]) / dt_units
        else:
            momentum = 0.05 if info["last_complete"] else 0.02
        momentum = max(-0.40, min(0.50, momentum))

        is_cand = (
            (stale_marks == 0 and info["last_rank"] <= 100)
            or (stale_marks == 1 and info["last_rank"] <= 20 and info["last_complete"])
            or (info["last_rank"] <= 10 and info["last_complete"])
        )
        if is_cand:
            candidates.append({
                "mint": mint,
                "last_cap": info["last_cap"],
                "stale_marks": stale_marks,
                "age_min": (t_last - info["created_at_ts"]) / 60.0,
                "complete": info["last_complete"],
                "momentum": momentum
            })

    N_PATHS = 3500
    place_counts = {k: defaultdict(int) for k in range(1, 6)}
    place_caps = {k: [] for k in range(1, 6)}

    for path_idx in range(N_PATHS):
        for h_min in horizons_min:
            steps = max(1.0, h_min / 10.0)
            int_steps = int(math.ceil(steps))
            sim_results = []

            for cand in candidates:
                cap = cand["last_cap"]
                curr_complete = cand["complete"]
                survived = True
                for s in range(1, int_steps + 1):
                    step_frac = min(1.0, steps - (s - 1))
                    age_step = cand["age_min"] + (s - 0.5) * 10.0
                    if cand["stale_marks"] > 0:
                        h_prob = 0.85 * step_frac
                    elif not curr_complete:
                        if cand["momentum"] > 0.10 and age_step < 35.0 and random.random() < 0.30 * step_frac:
                            curr_complete = True
                            h_prob = 0.08 * step_frac
                        else:
                            h_prob = 0.75 * step_frac
                    else:
                        if age_step < 40.0 and cand["momentum"] > 0.02:
                            h_prob = 0.03 * step_frac
                        elif cand["momentum"] <= 0.0:
                            h_prob = 0.35 * step_frac
                        else:
                            h_prob = (0.06 + 0.002 * min(100.0, age_step)) * step_frac

                    if random.random() < h_prob:
                        survived = False
                        break

                    mom_effect = cand["momentum"] * (0.60**s) * step_frac
                    trend = (0.05 if curr_complete and age_step < 40.0 else 0.01) * step_frac
                    reversion = -0.07 * (cap - 3.8) * step_frac
                    noise = random.gauss(0.0, (0.12 + 0.03 * math.sqrt(s)) * math.sqrt(step_frac))
                    cap += (mom_effect + trend + reversion + noise)

                if survived and cap > 1.0:
                    sim_results.append((cap, cand["mint"]))

            num_new = max(1, int(random.gauss(1.8 + 0.02 * h_min, 0.8)))
            mean_new = 3.30 + 0.35 * (1.0 - math.exp(-h_min / 40.0))
            for j in range(num_new):
                sim_results.append((random.gauss(mean_new, 0.38), f"__newcomer_{j}__"))

            sim_results.sort(key=lambda x: x[0], reverse=True)
            for k in range(1, 6):
                if k - 1 < len(sim_results):
                    c_val, winner = sim_results[k - 1]
                    place_caps[k].append(c_val)
                    if winner in all_history_mints:
                        place_counts[k][winner] += 1

    total_sims = N_PATHS * len(horizons_min)
    w_emp = max(0.0, min(1.0, (n_nonempty - 1) / 3.0))
    mean_h_min = sum(horizons_min) / len(horizons_min) if horizons_min else 1440.0
    h_factor = max(0.0, min(1.0, (mean_h_min - 10.0) / 60.0))
    confidence = w_emp * (1.0 - h_factor)
    power = 0.80 + 0.50 * confidence

    prob_rows = []
    mint_scores = defaultdict(float)
    for k in range(1, 6):
        counts = place_counts[k]
        if not counts:
            continue

        named_mass_budget = sum(counts.values()) / total_sims
        t_counts = {m: (c ** power) for m, c in counts.items()}
        t_sum = sum(t_counts.values())

        cands = []
        for mint, tc in t_counts.items():
            p = (tc / t_sum) * named_mass_budget
            if p >= 0.002:
                cands.append({"mint": mint, "place": k, "p": round(p, 6)})
        
        cands.sort(key=lambda x: x["p"], reverse=True)
        cands = cands[:20]

        # v3f change: enforce adaptive minimum unnamed share
        min_unnamed = 0.50 * (1.0 - confidence) + 0.15 * confidence
        max_named_allowed = 1.0 - min_unnamed
        named_sum_now = sum(c["p"] for c in cands)
        if named_sum_now > max_named_allowed:
            scale = max_named_allowed / named_sum_now
            for c in cands:
                c["p"] = round(c["p"] * scale, 6)
        
        s_val = sum(c["p"] for c in cands)
        if s_val > 0.9999:
            scale = 0.9999 / s_val
            for c in cands:
                c["p"] = round(c["p"] * scale, 6)
        
        prob_rows.extend(cands)
        for c in cands:
            mint_scores[c["mint"]] += c["p"]

    read_diffs_h = [(parse_iso(r) - t_last) / 3600.0 for r in reads]
    h_eff_hours = sum(read_diffs_h) / len(read_diffs_h) if read_diffs_h else 24.0
    h_eff_hours = max(0.17, min(2.0, h_eff_hours))

    num_bins = len(edges) + 1
    unif_val = 0.15 / num_bins
    size_rows = []
    for k in range(1, 6):
        samples = place_caps[k]
        counts = [0] * num_bins
        for c in samples:
            idx = 0 if c < edges[0] else (num_bins - 1 if c >= edges[-1] else bisect.bisect_right(edges, c))
            counts[idx] += 1
        emp_p = [cnt / len(samples) if samples else 1.0 / num_bins for cnt in counts]
        for _ in range(2):
            new_p = [0.0] * num_bins
            for i in range(num_bins):
                m = emp_p[i] * 0.25
                new_p[i] += emp_p[i] * 0.5
                if i > 0: new_p[i-1] += m
                else: new_p[i] += m
                if i < num_bins - 1: new_p[i+1] += m
                else: new_p[i] += m
            emp_p = new_p

        mean_sample = sum(samples) / len(samples) if samples else 3.25
        c1_mu, c1_std = mean_sample, 0.18 + 0.10 * math.sqrt(h_eff_hours)
        c2_mu, c2_std = 3.25, 0.35
        norm_p = []
        for b in range(num_bins):
            bc = 1.85 if b == 0 else (6.15 if b == 81 else (edges[b-1] + edges[b]) / 2.0)
            p1 = (math.exp(-0.5 * ((bc - c1_mu) / c1_std)**2) / (c1_std * math.sqrt(2 * math.pi))) * 0.05
            p2 = (math.exp(-0.5 * ((bc - c2_mu) / c2_std)**2) / (c2_std * math.sqrt(2 * math.pi))) * 0.05
            norm_p.append(0.55 * p1 + 0.30 * p2 + unif_val)
        
        final_p = [max(0.0015, w_emp * ep + (1 - w_emp) * np) for ep, np in zip(emp_p, norm_p)]
        s_fp = sum(final_p)
        final_p = [round(p * 0.9999 / s_fp, 6) for p in final_p]
        diff = round(0.9999 - sum(final_p), 6)
        max_idx = final_p.index(max(final_p))
        final_p[max_idx] = round(final_p[max_idx] + diff, 6)
        for b_idx, p_val in enumerate(final_p):
            size_rows.append({"place": k, "bin": b_idx, "p": p_val})

    top5_mints = sorted(all_history_mints, key=lambda m: (mint_scores[m], -mint_info[m]["last_rank"]), reverse=True)[:5]
    while len(top5_mints) < 5:
        top5_mints.append(f"fallback_{len(top5_mints)}")

    os.makedirs(output_dir, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"probabilities": prob_rows, "top5": top5_mints, "sizes": size_rows}, f, indent=2)

if __name__ == "__main__":
    main()
