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
        dt = datetime.datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.timestamp()
    return 0.0


def norm_cdf(x, mu, sigma):
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def main():
    random.seed(42)

    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

    task_file = os.path.join(input_dir, "task.json")
    history_file = os.path.join(input_dir, "history.ndjson")
    out_file = os.path.join(output_dir, "forecast.json")

    with open(task_file, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    reads = task_data.get("reads", [])
    edges = task_data.get("size.edges") or task_data.get("size_edges")
    if not edges:
        edges = [round(2.00 + i * 0.05, 2) for i in range(81)]

    marks = []
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

    marks.sort(key=lambda m: parse_iso(m.get("at")))

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
            created_at_val = row.get("created_at")
            if created_at_val is None:
                created_at_val = mark.get("at")
            created_at_ts = parse_iso(created_at_val)

            mint_history[mint].append(
                (m_idx, t_mark, rank, cap_log10, venue, complete)
            )

            mint_info[mint] = {
                "last_seen_mark": m_idx,
                "last_rank": rank,
                "last_cap": cap_log10,
                "last_venue": venue,
                "last_complete": complete,
                "created_at_ts": created_at_ts,
            }

    if not marks:
        sys.exit(1)

    t_last = parse_iso(marks[-1].get("at"))
    num_marks = len(marks)

    horizons_min = [(parse_iso(r) - t_last) / 60.0 for r in reads]
    if not horizons_min:
        horizons_min = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

    candidates = []
    for mint, info in mint_info.items():
        stale_marks = (num_marks - 1) - info["last_seen_mark"]
        age_min = (t_last - info["created_at_ts"]) / 60.0

        hist = mint_history[mint]
        if len(hist) >= 2:
            dt_min = (hist[-1][1] - hist[-2][1]) / 60.0
            dt_units = max(0.5, dt_min / 10.0)
            dcap = hist[-1][3] - hist[-2][3]
            momentum = dcap / dt_units
        else:
            momentum = 0.05 if info["last_complete"] else 0.02

        momentum = max(-0.40, min(0.50, momentum))

        is_candidate = (
            (stale_marks == 0 and info["last_rank"] <= 80)
            or (stale_marks == 1 and info["last_rank"] <= 20 and info["last_complete"])
            or (info["last_rank"] <= 10 and info["last_complete"])
        )

        if is_candidate:
            candidates.append(
                {
                    "mint": mint,
                    "last_cap": info["last_cap"],
                    "stale_marks": stale_marks,
                    "age_min": age_min,
                    "complete": info["last_complete"],
                    "venue": info["last_venue"],
                    "momentum": momentum,
                    "last_rank": info["last_rank"],
                }
            )

    N_PATHS = 3500
    place_counts = {k: defaultdict(int) for k in range(1, 6)}
    place_caps = {k: [] for k in range(1, 6)}

    total_evals = N_PATHS * len(horizons_min)

    for path_idx in range(N_PATHS):
        for h_min in horizons_min:
            steps = max(1.0, h_min / 10.0)
            int_steps = int(math.ceil(steps))

            sim_results = []

            for cand in candidates:
                cap = cand["last_cap"]
                survived = True
                curr_complete = cand["complete"]

                for s in range(1, int_steps + 1):
                    step_frac = min(1.0, steps - (s - 1))
                    age_step = cand["age_min"] + (s - 0.5) * 10.0

                    if cand["stale_marks"] > 0:
                        h_prob = 0.85 * step_frac
                    elif not curr_complete:
                        if (
                            cand["momentum"] > 0.10
                            and age_step < 35.0
                            and random.random() < 0.30 * step_frac
                        ):
                            curr_complete = True
                            h_prob = 0.08 * step_frac
                        else:
                            h_prob = 0.75 * step_frac
                    else:
                        if cand["age_min"] < 40.0 and cand["momentum"] > 0.02:
                            h_prob = 0.03 * step_frac
                        elif cand["momentum"] <= 0.0:
                            h_prob = 0.35 * step_frac
                        else:
                            h_prob = (
                                0.06 + 0.002 * min(100.0, age_step)
                            ) * step_frac

                    if random.random() < h_prob:
                        survived = False
                        break

                    mom_effect = cand["momentum"] * (0.60**s) * step_frac
                    trend = (0.05 if curr_complete and cand["age_min"] < 40.0 else 0.01) * step_frac
                    reversion = -0.07 * max(0.0, cap - 3.8) * step_frac
                    noise = random.gauss(0.0, (0.12 + 0.03 * math.sqrt(s)) * math.sqrt(step_frac))

                    cap = cap + mom_effect + trend + reversion + noise

                if survived and cap > 1.0:
                    sim_results.append((cap, cand["mint"]))

            exp_new = 1.8 + 0.035 * h_min
            num_new = max(1, int(random.gauss(exp_new, 0.8)))
            for j in range(num_new):
                new_cap = random.gauss(3.20 + 0.015 * h_min, 0.45 + 0.002 * h_min)
                sim_results.append((new_cap, f"__newcomer_{j}__"))

            sim_results.sort(key=lambda x: x[0], reverse=True)

            for k in range(1, 6):
                if k - 1 < len(sim_results):
                    cap_val, winner = sim_results[k - 1]
                    place_caps[k].append(cap_val)
                    if winner in all_history_mints:
                        place_counts[k][winner] += 1

    prob_rows = []
    for k in range(1, 6):
        sorted_cand = sorted(
            place_counts[k].items(), key=lambda x: x[1], reverse=True
        )
        sorted_cand = sorted_cand[:20]

        cand_probs = []
        for mint, cnt in sorted_cand:
            p_val = cnt / total_evals
            p_round = round(p_val, 6)
            if p_round >= 0.002:
                cand_probs.append({"mint": mint, "place": k, "p": p_round})

        total_p = sum(r["p"] for r in cand_probs)
        if total_p > 0.999900 and cand_probs:
            excess = total_p - 0.999900
            max_idx = max(range(len(cand_probs)), key=lambda i: cand_probs[i]["p"])
            cand_probs[max_idx]["p"] = round(
                max(0.0, cand_probs[max_idx]["p"] - excess), 6
            )

        prob_rows.extend(cand_probs)

    size_rows = []
    read_diffs_h = [(parse_iso(r) - t_last) / 3600.0 for r in reads]
    if read_diffs_h:
        h_eff_hours = sum(read_diffs_h) / len(read_diffs_h)
    else:
        h_eff_hours = sum(horizons_min) / (60.0 * len(horizons_min))
    h_eff_hours = max(0.17, min(2.0, h_eff_hours))

    num_bins = 82
    unif_val = 0.15 / num_bins

    for k in range(1, 6):
        samples = place_caps[k]
        if samples:
            mean_sample = sum(samples) / len(samples)
        else:
            mean_sample = 3.25

        comp1_std = 0.18 + 0.10 * math.sqrt(h_eff_hours)
        comp1_mu = mean_sample

        comp2_mu = 3.25
        comp2_std = 0.35

        def get_bin_center(b_idx):
            if b_idx == 0:
                return 1.85
            elif b_idx == 81:
                return 6.15
            else:
                return (edges[b_idx - 1] + edges[b_idx]) / 2.0

        raw_bin_probs = []
        for b_idx in range(num_bins):
            bc = get_bin_center(b_idx)
            p1_density = math.exp(-0.5 * ((bc - comp1_mu) / comp1_std) ** 2) / (
                comp1_std * math.sqrt(2 * math.pi)
            )
            p2_density = math.exp(-0.5 * ((bc - comp2_mu) / comp2_std) ** 2) / (
                comp2_std * math.sqrt(2 * math.pi)
            )

            p1_prob = p1_density * 0.05
            p2_prob = p2_density * 0.05

            val = 0.55 * p1_prob + 0.30 * p2_prob + unif_val
            val = max(0.0018, val)
            raw_bin_probs.append(val)

        sum_raw = sum(raw_bin_probs)
        if sum_raw > 0:
            norm_bin_probs = [bp / sum_raw * 0.9999 for bp in raw_bin_probs]
        else:
            norm_bin_probs = [0.9999 / num_bins] * num_bins

        max_idx = max(range(len(norm_bin_probs)), key=lambda i: norm_bin_probs[i])
        for b_idx in range(len(norm_bin_probs)):
            if norm_bin_probs[b_idx] < 0.0018:
                diff = 0.0018 - norm_bin_probs[b_idx]
                norm_bin_probs[b_idx] = 0.0018
                norm_bin_probs[max_idx] -= diff

        rounded_bins = [round(bp, 6) for bp in norm_bin_probs]
        sum_rb = sum(rounded_bins)

        if sum_rb > 0.999900:
            excess = sum_rb - 0.999900
            max_b = max(range(len(rounded_bins)), key=lambda i: rounded_bins[i])
            rounded_bins[max_b] = round(max(0.0, rounded_bins[max_b] - excess), 6)

        for b_idx, p_val in enumerate(rounded_bins):
            size_rows.append({"place": k, "bin": b_idx, "p": p_val})

    mint_scores = defaultdict(float)
    for r in prob_rows:
        mint_scores[r["mint"]] += r["p"]

    sorted_top = sorted(
        all_history_mints,
        key=lambda m: (
            mint_scores[m],
            -mint_info.get(m, {}).get("last_rank", 999),
        ),
        reverse=True,
    )

    top5_mints = []
    for m in sorted_top:
        if m in all_history_mints and m not in top5_mints:
            top5_mints.append(m)
        if len(top5_mints) == 5:
            break

    if len(top5_mints) < 5:
        for m in all_history_mints:
            if m not in top5_mints:
                top5_mints.append(m)
            if len(top5_mints) == 5:
                break

    while len(top5_mints) < 5:
        top5_mints.append(f"fallback_mint_{len(top5_mints)}")

    def repair_distribution(probs_list, size_list):
        p_by_place = defaultdict(list)
        for r in probs_list:
            p_by_place[r["place"]].append(r)

        fixed_prob = []
        for pl in range(1, 6):
            r_list = p_by_place[pl]
            if not r_list:
                continue
            r_list.sort(key=lambda x: x["p"], reverse=True)
            r_list = r_list[:20]
            for item in r_list:
                item["p"] = round(float(item["p"]), 6)
            r_list = [item for item in r_list if item["p"] >= 0.002]

            s_val = sum(item["p"] for item in r_list)
            if s_val > 0.999900:
                diff = s_val - 0.999900
                r_list[0]["p"] = round(max(0.0, r_list[0]["p"] - diff), 6)
            fixed_prob.extend(r_list)

        s_by_place = defaultdict(list)
        for r in size_list:
            s_by_place[r["place"]].append(r)

        fixed_size = []
        for pl in range(1, 6):
            r_list = s_by_place[pl]
            if not r_list:
                continue
            for item in r_list:
                item["p"] = round(float(item["p"]), 6)
            s_val = sum(item["p"] for item in r_list)
            if s_val > 0.999900:
                diff = s_val - 0.999900
                max_item = max(r_list, key=lambda x: x["p"])
                max_item["p"] = round(max(0.0, max_item["p"] - diff), 6)
            fixed_size.extend(r_list)

        return fixed_prob, fixed_size

    prob_rows, size_rows = repair_distribution(prob_rows, size_rows)

    output_payload = {
        "probabilities": prob_rows,
        "top5": top5_mints,
        "sizes": size_rows,
    }

    for pl in range(1, 6):
        p_sum = sum(
            r["p"] for r in output_payload["probabilities"] if r["place"] == pl
        )
        s_sum = sum(
            r["p"] for r in output_payload["sizes"] if r["place"] == pl
        )
        assert (
            p_sum <= 1.0 + 1e-9
        ), f"FATAL: place {pl} order sum {p_sum} > 1.0"
        assert s_sum <= 1.0 + 1e-9, f"FATAL: place {pl} size sum {s_sum} > 1.0"

    assert len(output_payload["top5"]) == 5
    assert len(set(output_payload["top5"])) == 5

    os.makedirs(output_dir, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)


if __name__ == "__main__":
    main()
