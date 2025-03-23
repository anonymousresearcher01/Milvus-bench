#!/bin/bash

# Targe devices
DATA_DEVICE="/dev/sda"                          # Source data device (원천 데이터 디바이스)
MILVUS_DEVICE="/dev/mapper/ubuntu--vg-ubuntu--lv"  # Milvus 데이터 디바이스

OUTPUT_DIR="../result_stat/io_monitoring"
EXPR=""
PHASE=""

mkdir -p $OUTPUT_DIR

init_stats() {
    cat /proc/diskstats > "$OUTPUT_DIR/$EXPR/diskstats_before_${PHASE}.txt"
}

# I/O 통계 수집 종료 및 결과 계산
calculate_stats() {
    cat /proc/diskstats > "$OUTPUT_DIR/$EXPR/diskstats_after_${PHASE}.txt"

    # 1. /dev/sda
    local data_dev_name=$(echo $DATA_DEVICE | sed 's/\/dev\///' | sed 's/\//-/g')

    # Load from "$OUTPUT_DIR/diskstats_before_${PHASE}.txt"
    local data_reads_before=$(grep " ${data_dev_name} " "$OUTPUT_DIR/$EXPR/diskstats_before_${PHASE}.txt" | awk '{print $4}')
    local data_sectors_read_before=$(grep " ${data_dev_name} " "$OUTPUT_DIR/$EXPR/diskstats_before_${PHASE}.txt" | awk '{print $6}')
    local data_writes_before=$(grep " ${data_dev_name} " "$OUTPUT_DIR/$EXPR/diskstats_before_${PHASE}.txt" | awk '{print $8}')
    local data_sectors_written_before=$(grep " ${data_dev_name} " "$OUTPUT_DIR/$EXPR/diskstats_before_${PHASE}.txt" | awk '{print $10}')

    # Load from "$OUTPUT_DIR/diskstats_after_${PHASE}.txt"
    local data_reads_after=$(grep " ${data_dev_name} " "$OUTPUT_DIR/$EXPR/diskstats_after_${PHASE}.txt" | awk '{print $4}')
    local data_sectors_read_after=$(grep " ${data_dev_name} " "$OUTPUT_DIR/$EXPR/diskstats_after_${PHASE}.txt" | awk '{print $6}')
    local data_writes_after=$(grep " ${data_dev_name} " "$OUTPUT_DIR/$EXPR/diskstats_after_${PHASE}.txt" | awk '{print $8}')
    local data_sectors_written_after=$(grep " ${data_dev_name} " "$OUTPUT_DIR/$EXPR/diskstats_after_${PHASE}.txt" | awk '{print $10}')

    # 2. /dev/mapper/ubuntu--vg-ubuntu--lv
    # local milvus_dev_name=$(echo $MILVUS_DEVICE | sed 's/\/dev\///' | sed 's/\//-/g' | sed 's/--/-/g')
    #NOTE(Dhmin): my device is "dm-0"
    # sudo dmsetup info -c
    # sudo lvs
    # sudo pvs
    local milvus_dev_name="dm-0"

    # Load from "$OUTPUT_DIR/diskstats_before_${PHASE}.txt"
    local milvus_reads_before=$(grep " ${milvus_dev_name} " "$OUTPUT_DIR/$EXPR/diskstats_before_${PHASE}.txt" | awk '{print $4}')
    local milvus_sectors_read_before=$(grep " ${milvus_dev_name} " "$OUTPUT_DIR/$EXPR/diskstats_before_${PHASE}.txt" | awk '{print $6}')
    local milvus_writes_before=$(grep " ${milvus_dev_name} " "$OUTPUT_DIR/$EXPR/diskstats_before_${PHASE}.txt" | awk '{print $8}')
    local milvus_sectors_written_before=$(grep " ${milvus_dev_name} " "$OUTPUT_DIR/$EXPR/diskstats_before_${PHASE}.txt" | awk '{print $10}')

    # Load from "$OUTPUT_DIR/diskstats_after_${PHASE}.txt"
    local milvus_reads_after=$(grep " ${milvus_dev_name} " "$OUTPUT_DIR/$EXPR/diskstats_after_${PHASE}.txt" | awk '{print $4}')
    local milvus_sectors_read_after=$(grep " ${milvus_dev_name} " "$OUTPUT_DIR/$EXPR/diskstats_after_${PHASE}.txt" | awk '{print $6}')
    local milvus_writes_after=$(grep " ${milvus_dev_name} " "$OUTPUT_DIR/$EXPR/diskstats_after_${PHASE}.txt" | awk '{print $8}')
    local milvus_sectors_written_after=$(grep " ${milvus_dev_name} " "$OUTPUT_DIR/$EXPR/diskstats_after_${PHASE}.txt" | awk '{print $10}')

    #NOTE(Dhmin): Manually calc diff is needed since I/O info from diskstats are accumulated value.
    # calc diff from accumulated digit - Source data device
    local data_read_ops=$((data_reads_after - data_reads_before))
    local data_read_sectors=$((data_sectors_read_after - data_sectors_read_before))
    local data_write_ops=$((data_writes_after - data_writes_before))
    local data_write_sectors=$((data_sectors_written_after - data_sectors_written_before))

    # calc diff from accumulated digit - Milvus data device
    local milvus_read_ops=$((milvus_reads_after - milvus_reads_before))
    local milvus_read_sectors=$((milvus_sectors_read_after - milvus_sectors_read_before))
    local milvus_write_ops=$((milvus_writes_after - milvus_writes_before))
    local milvus_write_sectors=$((milvus_sectors_written_after - milvus_sectors_written_before))

    # Sector -> Byte/KB/MB (1 sector = 512 Byte)
    local data_read_bytes=$((data_read_sectors * 512))
    local data_read_kb=$((data_read_bytes / 1024))
    local data_read_mb=$((data_read_kb / 1024))

    local data_write_bytes=$((data_write_sectors * 512))
    local data_write_kb=$((data_write_bytes / 1024))
    local data_write_mb=$((data_write_kb / 1024))

    local milvus_read_bytes=$((milvus_read_sectors * 512))
    local milvus_read_kb=$((milvus_read_bytes / 1024))
    local milvus_read_mb=$((milvus_read_kb / 1024))

    local milvus_write_bytes=$((milvus_write_sectors * 512))
    local milvus_write_kb=$((milvus_write_bytes / 1024))
    local milvus_write_mb=$((milvus_write_kb / 1024))

    # results csv
    if [ ! -f "$OUTPUT_DIR/$EXPR/io_stats_summary.csv" ]; then
        echo "phase,device,read_operations,read_bytes,read_kb,read_mb,write_operations,write_bytes,write_kb,write_mb" > "$OUTPUT_DIR/$EXPR/io_stats_summary.csv"
    fi

    echo "${PHASE},${DATA_DEVICE},${data_read_ops},${data_read_bytes},${data_read_kb},${data_read_mb},${data_write_ops},${data_write_bytes},${data_write_kb},${data_write_mb}" >> "$OUTPUT_DIR/$EXPR/io_stats_summary.csv"
    echo "${PHASE},${MILVUS_DEVICE},${milvus_read_ops},${milvus_read_bytes},${milvus_read_kb},${milvus_read_mb},${milvus_write_ops},${milvus_write_bytes},${milvus_write_kb},${milvus_write_mb}" >> "$OUTPUT_DIR/$EXPR/io_stats_summary.csv"

    # Detail results (file write)
    echo "phase,device,read_operations,read_bytes,read_kb,read_mb,write_operations,write_bytes,write_kb,write_mb" > "$OUTPUT_DIR/$EXPR/io_stats_${PHASE}.csv"
    echo "${PHASE},${DATA_DEVICE},${data_read_ops},${data_read_bytes},${data_read_kb},${data_read_mb},${data_write_ops},${data_write_bytes},${data_write_kb},${data_write_mb}" >> "$OUTPUT_DIR/$EXPR/io_stats_${PHASE}.csv"
    echo "${PHASE},${MILVUS_DEVICE},${milvus_read_ops},${milvus_read_bytes},${milvus_read_kb},${milvus_read_mb},${milvus_write_ops},${milvus_write_bytes},${milvus_write_kb},${milvus_write_mb}" >> "$OUTPUT_DIR/$EXPR/io_stats_${PHASE}.csv"

    # Display summary on console
    # echo "======== ${PHASE} phase I/O statistics ========"
    # echo "1. Source data device (${DATA_DEVICE}):"
    # echo "   Read operations: ${data_read_ops} times"
    # echo "   Data read: ${data_read_mb} MB (${data_read_kb} KB)"
    # echo "   Write operations: ${data_write_ops} times"
    # echo "   Data written: ${data_write_mb} MB (${data_write_kb} KB)"
    # echo ""
    # echo "2. Milvus data device (${MILVUS_DEVICE}):"
    # echo "   Read operations: ${milvus_read_ops} times"
    # echo "   Data read: ${milvus_read_mb} MB (${milvus_read_kb} KB)"
    # echo "   Write operations: ${milvus_write_ops} times"
    # echo "   Data written: ${milvus_write_mb} MB (${milvus_write_kb} KB)"
    # echo "======================================"
}

start_monitoring() {
    EXPR=$1
    PHASE=$2
    # echo "Phase '${EXPR}' '${PHASE}'is not being observed by I/O monitor"
    mkdir -p "$OUTPUT_DIR/$EXPR"
    init_stats
}

stop_monitoring() {
    EXPR=$1
    PHASE=$2
    # echo "Phase '${PHASE}' has ended"
    calculate_stats
}

case "$1" in
    start_monitoring)
        start_monitoring "$2" "$3"
        ;;
    stop_monitoring)
        stop_monitoring "$2" "$3"
        ;;
    *)
        echo "Usage: $0 {start_monitoring|stop_monitoring} [expr_name] [phase_name]"
        echo "Example: $0 start_monitoring insert_vectors load_data"
        echo "Example: $0 stop_monitoring insert_vectors load_data"
        exit 1
        ;;
esac

exit 0
