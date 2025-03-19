#!/bin/bash

OUTPUT_DIR="../milvus_io_results"
mkdir -p $OUTPUT_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEVICE="/dev/mapper/ubuntu--vg-ubuntu--lv" # target device
LOGFILE=$OUTPUT_DIR/"debug.log"

echo "Target device: $DEVICE" >> $LOGFILE
start_monitoring() {
  echo "Starting I/O monitoring for phase: $1"

  # iostat - interval 5 sec for I/O statistic
  iostat -xmt $DEVICE 5 > $OUTPUT_DIR/${TIMESTAMP}_${1}_iostat.log &
  IOSTAT_PID=$!

  # blktrace - block level IO
  sudo blktrace -d $DEVICE -o $OUTPUT_DIR/${TIMESTAMP}_${1}_blktrace &
  BLKTRACE_PID=$!

  # iotop - I/O usage per process
  sudo iotop -botqk > $OUTPUT_DIR/${TIMESTAMP}_${1}_iotop.log &
  IOTOP_PID=$!

  vmstat 5 > $OUTPUT_DIR/${TIMESTAMP}_${1}_vmstat.log &
  VMSTAT_PID=$!
}

stop_monitoring() {
  echo "Stopping I/O monitoring"
  kill $IOSTAT_PID $BLKTRACE_PID $IOTOP_PID $DSTAT_PID

  sudo blkparse -i $OUTPUT_DIR/${TIMESTAMP}_${1}_blktrace -d $OUTPUT_DIR/${TIMESTAMP}_${1}_blkparse.txt

  echo "Summary for phase: $1" > $OUTPUT_DIR/${TIMESTAMP}_${1}_summary.txt
  echo "=======================================" >> $OUTPUT_DIR/${TIMESTAMP}_${1}_summary.txt

  echo "Throughput Summary:" >> $OUTPUT_DIR/${TIMESTAMP}_${1}_summary.txt
  grep avg-cpu $OUTPUT_DIR/${TIMESTAMP}_${1}_iostat.log -A 1 | tail -n 1 >> $OUTPUT_DIR/${TIMESTAMP}_${1}_summary.txt

  echo "I/O Request Distribution:" >> $OUTPUT_DIR/${TIMESTAMP}_${1}_summary.txt
  sudo blkparse -i $OUTPUT_DIR/${TIMESTAMP}_${1}_blktrace -a issue -f "%a %S %n\n" | \
    awk '{io[$1]++; sectors[$1]+=$3} END {for (type in io) {print type, io[type], "requests,", sectors[type], "sectors"}}' \
    >> $OUTPUT_DIR/${TIMESTAMP}_${1}_summary.txt
}
