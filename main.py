import os
import sys
import json
import math
from datetime import datetime, timezone

# ==============================================================================
# MODEL TUNABLE CONSTANTS
# ==============================================================================
# Placement parameters
RETENTION_MAX = 0.70            # Max prob sum for known coins at h=0
RETENTION_MIN = 0.10            # Min prob sum for known coins at large h
RETENTION_DECAY = 0.60          # Decay rate per hour of horizon
TOP_CANDIDATES_PER_PLACE = 20   # Candidates nominated per place (sharper focus)

SCORE_LOG_CAP_WEIGHT = 2.5      # Weight for logSOL market cap
SCORE_RANK_PENALTY = 0.08       # Penalty per rank distance to target place
SCORE_VENUE_POOL_BONUS = 0.8    # Score bonus for pool venue
SCORE_COMPLETE_BONUS = 0.4      # Score bonus for complete curve
SCORE_MOMENTUM_WEIGHT = 0.08    # Damped weight for recent log cap growth
SCORE_STALENESS_PENALTY = 3.0   # Penalty per missed historical step
SCORE_TAU_BASE = 0.45           # Base softmax temperature
SCORE_TAU_PER_HOUR = 0.40       # Softmax temperature growth rate per sqrt(hour)

# Size distribution parameters
SIZE_BASE_MEANS = [3.70, 3.45, 3.30, 3.20, 3.10]  # Default logSOL cap means for places 1..5
SIZE_DRIFT = 0.30                                 # LogSOL cap positive drift per hour
SIZE_SIGMA_BASE = 0.10                            # Base standard deviation
SIZE_SIGMA_PER_HOUR = 0.30                        # Standard deviation growth rate per sqrt(hour)
SIZE_UNIFORM_EPS = 0.06                           # Uniform noise mixture to protect tails
DEFAULT_EDGES = [2.00 + i * 0.05 for i in range(81)] # 81 edges -> 82 bins


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def parse_iso(dt_str: str) -> datetime:
    """Parse ISO 8601 UTC timestamp safely."""
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str).astimezone(timezone.utc)

def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ==============================================================================
# MAIN FORECASTING LOGIC
# ==============================================================================
def main():
    # 1. Resolve Input/Output Paths
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

    task_path = os.path.join(input_dir, "task.json")
    history_path = os.path.join(input_dir, "history.ndjson")
    output_path = os.path.join(output_dir, "forecast.json")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 2. Parse task.json
    with open(task_path, "r", encoding="utf-8") as f:
        task = json.load(f)

    read_times = [parse_iso(r) for r in task.get("reads", [])]
    num_places = task.get("places", 5)
    
    # Extract size bin edges
    size_cfg = task.get("size", {})
    edges = size_cfg.get("edges", task.get("edges", DEFAULT_EDGES))
    if not edges:
        edges = DEFAULT_EDGES

    # 3. Parse history.ndjson
    marks = []
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("ok", True) and obj.get("rows"):
                        marks.append(obj)
                except Exception:
                    continue

    # Sort marks by timestamp
    marks.sort(key=lambda m: m.get("at", ""))
    num_marks = len(marks)

    # Extract historical coin statistics
    coins = {}  # mint -> metadata & history
    last_mark_time = None

    if num_marks > 0:
        last_mark_time = parse_iso(marks[-1]["at"])

        for step_idx, mark in enumerate(marks):
            rows = mark.get("rows", [])
            for row in rows:
                mint = row.get("mint")
                if not mint:
                    continue

                try:
                    cap_lamports = int(row.get("market_cap_lamports", "0"))
                except ValueError:
                    cap_lamports = 0

                log_cap = math.log10(max(cap_lamports, 1)) - 9.0  # log10 cap in SOL

                if mint not in coins:
                    coins[mint] = {
                        "mint": mint,
                        "first_step": step_idx,
                        "last_step": step_idx,
                        "last_rank": row.get("rank", 200),
                        "last_log_cap": log_cap,
                        "last_cap_lamports": cap_lamports,
                        "venue": row.get("venue", "curve"),
                        "complete": row.get("complete", False),
                        "history_caps": [(step_idx, log_cap)]
                    }
                else:
                    c = coins[mint]
                    c["last_step"] = step_idx
                    c["last_rank"] = row.get("rank", 200)
                    c["last_log_cap"] = log_cap
                    c["last_cap_lamports"] = cap_lamports
                    c["venue"] = row.get("venue", c["venue"])
                    c["complete"] = row.get("complete", c["complete"])
                    c["history_caps"].append((step_idx, log_cap))

    # Handle reference timestamp fallback
    if last_mark_time is None:
        last_mark_time = parse_iso(task.get("history_until", task.get("t0", "2025-01-01T00:00:00Z")))

    # Compute short-term momentum for each coin
    for c in coins.values():
        hc = c["history_caps"]
        if len(hc) >= 2:
            c["momentum"] = hc[-1][1] - hc[-2][1]
        else:
            c["momentum"] = 0.0

    # 4. Generate Predictions Across Answer Reads
    place_coin_probs = {k: {} for k in range(1, num_places + 1)}
    place_size_probs = {k: [0.0] * 82 for k in range(1, num_places + 1)}

    num_reads = max(len(read_times), 1)

    for read_dt in read_times:
        horizon_hours = max((read_dt - last_mark_time).total_seconds() / 3600.0, 0.0)

        # Retention probability sum for historical coins at this horizon
        retention = RETENTION_MIN + (RETENTION_MAX - RETENTION_MIN) * math.exp(-RETENTION_DECAY * horizon_hours)

        # Softmax temperature scaling with horizon
        tau = SCORE_TAU_BASE + SCORE_TAU_PER_HOUR * math.sqrt(horizon_hours)

        # A. Coin Placement Probabilities
        for k in range(1, num_places + 1):
            scores = []
            for mint, c in coins.items():
                staleness = (num_marks - 1) - c["last_step"] if num_marks > 0 else 0
                
                # Composite score formula
                log_cap_val = c["last_log_cap"]
                score = (SCORE_LOG_CAP_WEIGHT * log_cap_val -
                         SCORE_RANK_PENALTY * abs(c["last_rank"] - k) +
                         (SCORE_VENUE_POOL_BONUS if c["venue"] == "pool" else 0.0) +
                         (SCORE_COMPLETE_BONUS if c["complete"] else 0.0) +
                         SCORE_MOMENTUM_WEIGHT * c["momentum"] -
                         SCORE_STALENESS_PENALTY * staleness)
                scores.append((mint, score))

            if scores:
                # Softmax over top candidates
                max_s = max(s for _, s in scores)
                exp_scores = [(m, math.exp((s - max_s) / tau)) for m, s in scores]

                # Sort candidates
                exp_scores.sort(key=lambda x: x[1], reverse=True)
                top_k_candidates = exp_scores[:TOP_CANDIDATES_PER_PLACE]

                top_exp_sum = sum(es for _, es in top_k_candidates)
                if top_exp_sum > 0:
                    for mint, es in top_k_candidates:
                        # Scaled probability bounded by retention budget
                        p = (es / top_exp_sum) * retention
                        place_coin_probs[k][mint] = place_coin_probs[k].get(mint, 0.0) + p / num_reads

        # B. Market Cap Size Distribution
        # Extract top 5 caps from last historical mark if available
        last_top_caps = []
        if num_marks > 0 and marks[-1].get("rows"):
            for r in marks[-1]["rows"][:5]:
                try:
                    cap = int(r.get("market_cap_lamports", "0"))
                    if cap > 0:
                        last_top_caps.append(math.log10(cap) - 9.0)
                except ValueError:
                    pass

        for k in range(1, num_places + 1):
            base_mean = SIZE_BASE_MEANS[k - 1]

            if k <= len(last_top_caps):
                hist_mean = last_top_caps[k - 1]
                mu = hist_mean + SIZE_DRIFT * horizon_hours
            else:
                mu = base_mean + SIZE_DRIFT * horizon_hours

            sigma = SIZE_SIGMA_BASE + SIZE_SIGMA_PER_HOUR * math.sqrt(horizon_hours)

            # Integrate Gaussian CDF across 82 bins
            # Edges: [-inf, edges[0], edges[1], ..., edges[80], +inf]
            bin_probs = [0.0] * 82
            for b in range(82):
                if b == 0:
                    lower = -1e9
                    upper = edges[0]
                elif b == 81:
                    lower = edges[80]
                    upper = 1e9
                else:
                    lower = edges[b - 1]
                    upper = edges[b]

                p_bin = normal_cdf((upper - mu) / sigma) - normal_cdf((lower - mu) / sigma)
                bin_probs[b] = max(p_bin, 0.0)

            # Uniform mixture smoothing to eliminate tail zero penalties
            u = 1.0 / 82.0
            sum_p = sum(bin_probs)
            if sum_p > 0:
                bin_probs = [(1.0 - SIZE_UNIFORM_EPS) * (p / sum_p) + SIZE_UNIFORM_EPS * u for p in bin_probs]

            for b in range(82):
                place_size_probs[k][b] += bin_probs[b] / num_reads

    # 5. Format Output Arrays
    prob_rows = []
    for k in range(1, num_places + 1):
        candidates = sorted(place_coin_probs[k].items(), key=lambda x: x[1], reverse=True)[:TOP_CANDIDATES_PER_PLACE]
        sum_p = sum(p for _, p in candidates)
        
        # Enforce sum <= 1.0
        scale = 1.0 if sum_p <= 1.0 else (1.0 / sum_p)
        for mint, p in candidates:
            if p * scale > 1e-6:
                prob_rows.append({
                    "mint": mint,
                    "place": k,
                    "p": round(p * scale, 6)
                })

    size_rows = []
    for k in range(1, num_places + 1):
        sp_list = place_size_probs[k]
        total_sp = sum(sp_list)
        if total_sp > 0:
            sp_list = [p / total_sp for p in sp_list]
        for b in range(82):
            if sp_list[b] > 1e-6:
                size_rows.append({
                    "place": k,
                    "bin": b,
                    "p": round(sp_list[b], 6)
                })

    # Determine top 5 best guesses
    mint_overall = {}
    for k in range(1, num_places + 1):
        for mint, p in place_coin_probs[k].items():
            mint_overall[mint] = mint_overall.get(mint, 0.0) + p / (k ** 0.5)

    sorted_mints = sorted(mint_overall.items(), key=lambda x: x[1], reverse=True)
    top5_mints = [m for m, _ in sorted_mints[:5]]

    # Pad top5 if history had fewer coins
    if len(top5_mints) < 5:
        existing = set(top5_mints)
        for c in coins.keys():
            if c not in existing:
                top5_mints.append(c)
                existing.add(c)
                if len(top5_mints) == 5:
                    break

    while len(top5_mints) < 5:
        top5_mints.append(f"dummy_mint_{len(top5_mints) + 1}")

    # Construct final forecast output
    forecast = {
        "probabilities": prob_rows,
        "top5": top5_mints[:5],
        "sizes": size_rows
    }

    # Write forecast.json safely
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(forecast, f, separators=(',', ':'))


if __name__ == "__main__":
    main()
