from matplotlib import pyplot as plt


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

    plt.savefig(f"../result_stat/index_build_times_{num}.png")

    if io_stats is not None:
        plot_io_stats(experiment_name, io_stats, num)
