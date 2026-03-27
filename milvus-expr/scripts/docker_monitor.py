import subprocess
import time
import re
import datetime
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.dates import DateFormatter


class DockerMonitor:
    def __init__(self, container_names: List[str], interval: int = 5, duration: int = 3600) -> None:
        """Docker container monitoring class"""
        self.container_names: List[str] = container_names
        self.interval: int = interval
        self.duration: int = duration
        self.data: Dict[str, List[Dict[str, Union[int, float]]]] = {name: [] for name in container_names}
        self.output_dir: str = "results"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def parse_memory(self, mem_str: str) -> int:
        """Parse memory string with converting to MB unit"""
        value = float(re.findall(r"[\d.]+", mem_str)[0])
        if "GiB" in mem_str:
            return value * 1024
        elif "MiB" in mem_str:
            return value
        elif "KiB" in mem_str:
            return value / 1024
        else:
            return value

    def parse_block_io(self, block_io_str) -> Tuple[int, int]:
        parts = block_io_str.split("/")
        read_str = parts[0].strip()
        write_str = parts[1].strip()

        read_value = float(re.findall(r"[\d.]+", read_str)[0])
        write_value = float(re.findall(r"[\d.]+", write_str)[0])

        if "GB" in read_str:
            read_value *= 1024
        elif "KB" in read_str:
            read_value /= 1024

        if "GB" in write_str:
            write_value *= 1024
        elif "KB" in write_str:
            write_value /= 1024

        return read_value, write_value

    def collect_data(self) -> None:
        """Collect Docker stats data"""
        collect_start_time = time.time()
        iterations = 0

        while time.time() - collect_start_time < self.duration:
            timestamp = datetime.datetime.now()

            try:
                docker_stat_cmd = [
                    "docker",
                    "stats",
                    "--no-stream",
                    "--format",
                    "{{.Name}},{{.MemUsage}},{{.CPUPerc}},{{.MemPerc}},{{.BlockIO}},{{.NetIO}}",
                ]
                result = subprocess.run(docker_stat_cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"Error executing docker stats: {result.stderr}")
                    time.sleep(self.interval)
                    continue

                lines = result.stdout.strip().split("\n")
                for line in lines:
                    parts = line.split(",")
                    if len(parts) >= 5:
                        container_name = parts[0]

                        if container_name in self.container_names:
                            mem_usage = parts[1]
                            cpu_perc = parts[2]
                            mem_perc = parts[3]
                            block_io = parts[4]
                            # net_io = parts[5] if len(parts) == 6 else "0B / 0B"

                            mem_mb = self.parse_memory(mem_usage.split("/")[0].strip())
                            read_mb, write_mb = self.parse_block_io(block_io)

                            self.data[container_name].append(
                                {
                                    "timestamp": timestamp,
                                    "memory_mb": mem_mb,
                                    "cpu_percent": float(cpu_perc.strip("%")),
                                    "memory_percent": float(mem_perc.strip("%")),
                                    "io_read_mb": read_mb,
                                    "io_write_mb": write_mb,
                                }
                            )
                        else:
                            print(f"Unexpected docker container name here: {container_name}")
                    else:
                        print(f"Some info is missing..: {parts}")
                        continue

                iterations += 1
                print(f"Iteration {iterations} completed")
            except Exception as e:
                print(f"Exception? {e}")

            time.sleep(self.interval)

    def plot_results(self) -> None:
        """Plot the collected docker container data"""
        for container in self.container_names:
            if not self.data[container]:
                print(f"No data collected for container: {container}")
                continue

            df = pd.DataFrame(self.data[container])

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

            # Memory Usage
            ax1.plot(df["timestamp"], df["memory_mb"], "b-", label="Memory Usage (MB)")
            ax1.set_title(f"Memory Usage - {container}")
            ax1.set_ylabel("Memory (MB)")
            ax1.grid(True)
            ax1.legend()

            # CPU Usage
            ax1_twin = ax1.twinx()
            ax1_twin.plot(df["timestamp"], df["cpu_percent"], "r-", label="CPU Usage (%)")
            ax1_twin.set_ylabel("CPU (%)")
            ax1_twin.legend(loc="upper right")

            # Disk Read
            ax2.plot(df["timestamp"], df["io_read_mb"], "g-", label="Disk Read (MB)")
            ax2.set_title(f"Disk I/O - Read - {container}")
            ax2.set_ylabel("Read (MB)")
            ax2.grid(True)
            ax2.legend()

            # Disk Write
            ax3.plot(df["timestamp"], df["io_write_mb"], "m-", label="Disk Write (MB)")
            ax3.set_title(f"Disk I/O - Write - {container}")
            ax3.set_xlabel("Time")
            ax3.set_ylabel("Write (MB)")
            ax3.grid(True)
            ax3.legend()

            date_form = DateFormatter("%H:%M:%S")
            ax3.xaxis.set_major_formatter(date_form)

            fig.tight_layout()
            plt.savefig(
                f"{self.output_dir}/{container}_monitoring_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            plt.close()

            df.to_csv(
                f"{self.output_dir}/{container}_monitoring_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                index=False,
            )


if __name__ == "__main__":
    containers: List[str] = ["milvus-standalone", "milvus-etcd", "milvus-minio"]

    # NOTE(dhmin): set monitoring info (default is to monitor 1 hour and collect info every 5 sec.)
    interval_sec = 5  # 5 sec
    duration_sec = 3600  # 1 hour

    print(f"Docker Container Monitoring ({interval_sec} interval, total {duration_sec} sec)")
    print(f"Monitoring target container: {', '.join(containers)}")

    monitor = DockerMonitor(containers, interval_sec, duration_sec)
    try:
        monitor.collect_data()
        monitor.plot_results()
        print("Completed. See resulst directory.")
    except KeyboardInterrupt:
        monitor.plot_results()
        print("Not completed but see result directory.")
