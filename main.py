import random
import time

import matplotlib.pyplot as plt
import pandas as pd

from mock_data import mock_addresses


def rule_based_filter(address_data):
    """
    Returns True if the address appears on a blacklist (threat present).
    """
    if address_data.get("in_osint_blacklist") is True:
        return True
    return False


def ml_anomaly_detection(address_data):
    """
    Checks transaction-history anomalies (e.g. zero-value bursts, sudden volume spikes).
    Returns True when anomaly heuristics fire.
    """
    features = address_data.get("transaction_features", {})
    # Simple threshold-style ML simulation
    is_spike = features.get("sudden_volume_spike", False)
    high_zero_value = features.get("zero_value_tx_1h", 0) > 10

    if is_spike or high_zero_value:
        return True
    return False


def trigger_llm(address_data):
    """
    Simulates local LLM analysis with a 1.2s delay and returns a security summary string.
    """
    time.sleep(1.2)
    return (
        "LLM analysis complete: high-risk, inconsistent transaction pattern detected."
    )


def evaluate_transaction(address_data):
    """
    Main decision flow:
    1. Rule-based filter and anomaly detection run first.
    2. If a threat is found, the LLM runs (Pending/Denied).
    3. Otherwise the LLM is skipped (Bypass / Allow).
    """
    is_rule_broken = rule_based_filter(address_data)
    is_anomaly_detected = ml_anomaly_detection(address_data)

    llm_triggered = False
    llm_response = ""
    decision = ""

    if is_rule_broken or is_anomaly_detected:
        llm_triggered = True
        llm_response = trigger_llm(address_data)

        if is_rule_broken:
            decision = "Denied"  # Hard rule violation
        else:
            decision = "Pending"  # Suspicious but not a hard deny
    else:
        decision = "Allow"  # Clean path — LLM skipped

    return {
        "decision": decision,
        "llm_triggered": llm_triggered,
        "llm_response": llm_response,
    }


def visualize_latency(df):
    """
    Reads latency values from a pandas DataFrame and plots a Matplotlib bar chart.
    """
    plt.figure(figsize=(9, 6))

    # Parse strings like "1200.00 ms" into floats
    latencies = df["Latency (ms)"].str.replace(" ms", "").astype(float)
    scenarios = df["Scenario Type"]

    # Colors: Safe -> green, Malicious -> red, Zero-day -> orange
    colors = ["#2ca02c", "#d62728", "#ff7f0e"]

    bars = plt.bar(scenarios, latencies, color=colors, edgecolor="black", alpha=0.8)

    plt.title(
        "Performance Analysis: Processing Latency by Threat Scenario",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("Scenario", fontsize=12, labelpad=10)
    plt.ylabel("Processing Time (ms)", fontsize=12, labelpad=10)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + (max(latencies) * 0.02),
            f"{yval} ms",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    plt.ylim(0, max(latencies) * 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig("latency_chart.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("\n[+] First chart saved as 'latency_chart.png'.")


def load_test_and_visualize():
    """
    Runs a 100-transaction load test to show how often the LLM is bypassed
    (resource / latency optimization).
    """
    print("\n[!] Starting 100-transaction load test...")
    print(
        "    Note: simulation may wait ~24s total for LLM paths. Please wait..."
    )

    # Batch: 80% Safe, 10% Malicious, 10% Zero-day
    batch = []
    batch.extend([mock_addresses[0]] * 80)
    batch.extend([mock_addresses[1]] * 10)
    batch.extend([mock_addresses[2]] * 10)

    random.shuffle(batch)

    llm_bypassed = 0
    llm_triggered = 0

    for tx in batch:
        evaluation = evaluate_transaction(tx)
        if evaluation["llm_triggered"]:
            llm_triggered += 1
        else:
            llm_bypassed += 1

    labels = ["LLM Bypassed\n(Safe / Fast-Track)", "LLM Triggered\n(Deep Analysis)"]
    sizes = [llm_bypassed, llm_triggered]

    colors = ["#4daf4a", "#e41a1c"]
    explode = (0.05, 0)

    plt.figure(figsize=(8, 6))

    def custom_autopct(pct, allvals):
        absolute = int(round(pct / 100.0 * sum(allvals)))
        return f"{absolute} Transactions\n({pct:.1f}%)"

    _wedges, _texts, autotexts = plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda pct: custom_autopct(pct, sizes),
        startangle=140,
        explode=explode,
        shadow=True,
        textprops=dict(color="black", fontweight="bold", fontsize=11),
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(12)

    plt.title(
        "Resource Optimization: LLM Trigger Rate over 100 Transactions",
        fontsize=15,
        pad=20,
        fontweight="bold",
    )

    plt.savefig("resource_chart.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("[+] Load test finished. Chart saved as 'resource_chart.png'.")


def run_tests():
    print("Starting system...\nLoading modules and rules...\n")
    results = []
    scenario_names = ["Safe Address", "Known Malicious", "Zero-Day Suspicious"]

    for idx, addr_data in enumerate(mock_addresses):
        start_time = time.time()
        evaluation = evaluate_transaction(addr_data)
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000

        results.append(
            {
                "Address": addr_data["address"],
                "Scenario Type": scenario_names[idx]
                if idx < len(scenario_names)
                else f"Unknown Scenario {idx}",
                "Final Decision": evaluation["decision"],
                "LLM Triggered?": "Yes" if evaluation["llm_triggered"] else "No",
                "Latency (ms)": f"{latency_ms:.2f} ms",
            }
        )

    df = pd.DataFrame(results)

    print("--- SIMULATION TEST RESULTS ---")
    print(df.to_string(index=False))

    visualize_latency(df)
    load_test_and_visualize()


if __name__ == "__main__":
    run_tests()
