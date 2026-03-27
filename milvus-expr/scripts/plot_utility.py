import json
from matplotlib import pyplot as plt
import numpy as np


def plot_io_stats(experiment_name, io_stats, num):
    plt.figure(figsize=(12, 8))

    phases = []
    data_read_mb = []
    data_write_mb = []
    milvus_read_mb = []
    milvus_write_mb = []

    for phase, stats in io_stats.items():
        phases.append(phase)
        for device_stats in stats:
            # print("Device: ", device_stats)
            if device_stats["device"] == "/dev/sda":
                data_read_mb.append(device_stats["read_mb"])
                data_write_mb.append(device_stats["write_mb"])
            elif device_stats["device"] == "/dev/mapper/ubuntu--vg-ubuntu--lv":
                milvus_read_mb.append(device_stats["read_mb"])
                milvus_write_mb.append(device_stats["write_mb"])

    # Source data storage
    plt.subplot(2, 1, 1)
    x = range(len(phases))
    width = 0.35

    plt.bar([i - width / 2 for i in x], data_read_mb, width, label="Original Data Read (MB)")
    plt.bar([i + width / 2 for i in x], data_write_mb, width, label="Original Data Write (MB)")

    plt.xlabel("Phase")
    plt.ylabel("I/O size (MB)")
    plt.title("Local Storage (/dev/sda) I/O Activity")
    plt.xticks(x, phases)
    plt.legend()
    plt.grid(True)

    # Milvus data storage
    plt.subplot(2, 1, 2)
    plt.bar([i - width / 2 for i in x], milvus_read_mb, width, label="Milvus Read (MB)")
    plt.bar([i + width / 2 for i in x], milvus_write_mb, width, label="Milvus Write (MB)")

    plt.xlabel("Phase")
    plt.ylabel("I/O size (MB)")
    plt.title("Milvus Storage (/dev/mapper/ubuntu--vg-ubuntu--lv) I/O Activity")
    plt.xticks(x, phases)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"../result_stat/{experiment_name}_io_activity_{num}.png")


def plot_inser_vectors(experiment_name, timing_stats, num, io_stats=None):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(timing_stats["insert_batches"]) + 1), timing_stats["insert_batches"])
    plt.xlabel("Batch idx (size of batch = 1000 vectors)")
    plt.ylabel("Latency (sec)")
    plt.title("Per-batch Latency")
    plt.grid(True)
    plt.savefig(f"../result_stat/insert_vectors_times_{num}.png")

    labels = ["load_data", "prepare_collection", "insert_batches", "Flush", "Sync"]
    sizes = [
        timing_stats["load_data"],
        timing_stats["prepare_collection"],
        sum(timing_stats["insert_batches"]),
        timing_stats["flush_to_disk"],
        timing_stats["sync_disk"],
    ]

    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.axis("equal")
    plt.title("Time distribution during vector insertion")
    plt.savefig(f"../result_stat/insert_vectors_time_breakdown_{num}.png")

    # CDF
    plt.figure(figsize=(10, 6))

    cumulative_times = np.cumsum(timing_stats["insert_batches"])

    total_time = cumulative_times[-1]

    x_values = cumulative_times
    y_values = np.arange(1, len(timing_stats["insert_batches"]) + 1) / len(timing_stats["insert_batches"])

    plt.plot(x_values, y_values, marker=".", linestyle="-")
    plt.xlabel("Elapsed Time (sec)")
    plt.ylabel("Fraction of Vectors Inserted")
    plt.title("CDF of Vector Insertion Completion")
    plt.grid(True)

    percentiles = [0.25, 0.5, 0.75, 0.9]
    for p in percentiles:
        idx = np.abs(y_values - p).argmin()
        plt.axhline(y=p, color="r", linestyle="--", alpha=0.3)
        plt.axvline(x=x_values[idx], color="r", linestyle="--", alpha=0.3)
        plt.annotate(
            f"{int(p*100)}%: {x_values[idx]:.2f}s",
            xy=(x_values[idx], p),
            xytext=(x_values[idx] + total_time * 0.02, p + 0.02),
            arrowprops=dict(arrowstyle="->", color="black"),
        )

    plt.savefig(f"../result_stat/insert_vectors_cdf_{num}.png")

    if io_stats is not None:
        plot_io_stats(experiment_name, io_stats, num)


def plot_build_index(experiment_name, timing_stats, num, io_stats=None):
    plt.figure(figsize=(10, 6))

    labels = [key for key in timing_stats.keys() if key != "total"]
    times = [timing_stats[key] for key in labels]

    total_time = timing_stats["total"]
    percentages = [time / total_time * 100 for time in times]

    plt.bar(0, times[0], width=0.3, label=labels[0])
    bottom = times[0]

    for i in range(1, len(times)):
        plt.bar(0, times[i], bottom=bottom, width=0.3, label=labels[i])
        bottom += times[i]

    plt.ylabel("Time (seconds)")
    plt.title(f"Index Building Time Breakdown (Total: {total_time:.2f}s)")
    plt.xticks([0], ["Index Build Process"])
    plt.legend(loc="upper right")
    plt.grid(True, axis="y")

    bottom = 0
    for i, time in enumerate(times):
        if time > 0.1:  # 너무 작은 값은 표시하지 않음
            plt.text(
                0, bottom + time / 2, f"{labels[i]}\n{time:.2f}s\n({percentages[i]:.1f}%)", ha="center", va="center"
            )
        bottom += time

    plt.savefig(f"../result_stat/build_index_times_{num}.png")

    if io_stats is not None:
        plot_io_stats(experiment_name, io_stats, num)


def plot_load_index(experiment_name, timing_stats, num, io_stats=None):
    labels = [key for key in timing_stats.keys() if key != "total"]
    times = [timing_stats[key] for key in labels]

    total_time = timing_stats["total"]
    percentages = [time / total_time * 100 for time in times]

    plt.bar(0, times[0], width=0.3, label=labels[0])
    bottom = times[0]

    for i in range(1, len(times)):
        plt.bar(0, times[i], bottom=bottom, width=0.3, label=labels[i])
        bottom += times[i]

    plt.ylabel("Time (seconds)")
    plt.title(f"Index Loading Time (Total: {total_time:.2f}s)")
    plt.xticks([0], ["Index Build Process"])
    plt.legend(loc="upper right")
    plt.grid(True, axis="y")

    bottom = 0
    for i, time in enumerate(times):
        if time > 0.1:
            plt.text(
                0, bottom + time / 2, f"{labels[i]}\n{time:.2f}s\n({percentages[i]:.1f}%)", ha="center", va="center"
            )
        bottom += time

    plt.savefig(f"../result_stat/load_index_times_{num}.png")

    with open(f"../result_stat/{experiment_name}_results_{num}.json", "r") as f:
        combined_stats = json.load(f)

    memory_before = combined_stats.get("memory_before_load_mb", 0)
    memory_after = combined_stats.get("memory_after_load_mb", 0)
    memory_increase = combined_stats.get("memory_increase_mb", 0)

    plt.figure(figsize=(8, 6))

    labels = ["Before Load", "After Load", "Increase"]
    memory_values = [memory_before, memory_after, memory_increase]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    plt.bar(np.arange(len(labels)), memory_values, color=colors, width=0.5)
    plt.xticks(np.arange(len(labels)), labels)
    plt.ylabel("Memory Usage (MB)")
    plt.title(f"Memory Usage During Index Load (Vectors: {num})")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    for i, v in enumerate(memory_values):
        plt.text(i, v + 5, f"{v:.1f} MB", ha="center")

    plt.tight_layout()
    plt.savefig(f"../result_stat/load_index_memory_{num}.png")

    if io_stats is not None:
        plot_io_stats(experiment_name, io_stats, num)


def plot_search_vectors(experiment_name, timing_stats, num, io_stats=None):
    json_output_name = f"{experiment_name}_results_{num}.json"
    result_path = f"../result_stat/{json_output_name}"

    try:
        with open(result_path, "r") as f:
            results = json.load(f)
    except Exception as e:
        print(f": {e}")
        return

    latencies = results.get("individual_latencies", [])
    if latencies:
        plt.figure(figsize=(10, 6))

        plt.hist(latencies, bins=20, alpha=0.7, color="#3498db")
        plt.xlabel("Query delay (Sec)")
        plt.ylabel("# of Query")
        plt.title(f"Query Searching times (Total {len(latencies)} queries)")
        plt.grid(True, linestyle="--", alpha=0.7)

        avg_latency = results.get("avg_latency", 0)
        p95_latency = results.get("p95_latency", 0)

        plt.axvline(x=avg_latency, color="r", linestyle="-", label=f"Avg: {avg_latency:.4f} sec")
        plt.axvline(x=p95_latency, color="g", linestyle="--", label=f"95%: {p95_latency:.4f} sec")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"../result_stat/search_vectors_latency_distribution_{num}.png")

    plt.figure(figsize=(8, 6))

    metrics = ["avg_latency", "min_latency", "max_latency", "p95_latency"]
    metric_names = ["Avg", "Min", "MAx", "95%"]
    values = [results.get(metric, 0) for metric in metrics]

    bars = plt.bar(metric_names, values, color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"], width=0.6)

    plt.ylabel("Latency (sec)")
    plt.title(f'Search Latency (# of vectors: {num}, # of quries: {results.get("total_queries", 0)})')
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, height + 0.005, f"{value:.4f}", ha="center", va="bottom", rotation=0
        )

    # plt.tight_layout()
    plt.savefig(f"../result_stat/search_latency_summary_{num}.png")

    plt.figure(figsize=(6, 5))

    qps = results.get("qps", 0)
    plt.bar(["QPS"], [qps], color="#9b59b6", width=0.4)
    plt.ylabel("Queries per Second")
    plt.title("Search Performance (QPS)", pad=20)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    plt.text(0, qps + qps * 0.05, f"{qps:.2f}", ha="center", va="bottom", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"../result_stat/search_qps_{num}.png")

    if io_stats is not None:
        plot_io_stats(experiment_name, io_stats, num)
