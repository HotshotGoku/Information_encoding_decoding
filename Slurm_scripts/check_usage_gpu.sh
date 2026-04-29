#!/usr/bin/env bash

PARTITION="youlab-gpu"
NOW=$(date "+%Y-%m-%d %H:%M:%S")

printf "PARTITION %-10s %s\n" "$PARTITION" "$NOW"
printf "%-22s %-4s %-11s %-11s %-13s %s\n" \
  "NODE" "ST" "GPU(u/t)" "CPU(u/t)" "MEM(u/t)" "TOP USERS (gpu)"
printf '%.0s─' {1..96}
printf "\n"

for node in $(sinfo -h -N -p "$PARTITION" -o "%N"); do
    info=$(scontrol show node "$node")

    state=$(echo "$info" | sed -n 's/.*State=\([A-Z]*\).*/\1/p' | head -n1)
    case "$state" in
        IDLE) st="id" ;;
        MIXED) st="mx" ;;
        ALLOCATED) st="al" ;;
        DOWN) st="dn" ;;
        DRAIN|DRAINED) st="dr" ;;
        *) st=$(echo "$state" | cut -c1-2 | tr 'A-Z' 'a-z') ;;
    esac

    cfg_cpu=$(echo "$info"   | sed -n 's/.*CfgTRES=.*cpu=\([0-9]\+\).*/\1/p')
    alloc_cpu=$(echo "$info" | sed -n 's/.*AllocTRES=.*cpu=\([0-9]\+\).*/\1/p')

    cfg_mem_m=$(echo "$info"   | sed -n 's/.*CfgTRES=.*mem=\([0-9]\+\)M.*/\1/p')
    alloc_mem_raw=$(echo "$info" | sed -n 's/.*AllocTRES=.*mem=\([0-9]\+[KMGTP]\).*/\1/p')

    cfg_gpu=$(echo "$info"   | sed -n 's/.*CfgTRES=.*gres\/gpu\(:[A-Za-z0-9_-]*\)\?=\([0-9]\+\).*/\2/p' | tail -n1)
    alloc_gpu=$(echo "$info" | sed -n 's/.*AllocTRES=.*gres\/gpu\(:[A-Za-z0-9_-]*\)\?=\([0-9]\+\).*/\2/p' | tail -n1)

    [ -z "$cfg_cpu" ] && cfg_cpu=0
    [ -z "$alloc_cpu" ] && alloc_cpu=0
    [ -z "$cfg_gpu" ] && cfg_gpu=0
    [ -z "$alloc_gpu" ] && alloc_gpu=0
    [ -z "$cfg_mem_m" ] && cfg_mem_m=0
    [ -z "$alloc_mem_raw" ] && alloc_mem_raw="0G"

    cfg_mem_g=$(awk -v m="$cfg_mem_m" 'BEGIN{printf "%d", m/1024}')

    case "$alloc_mem_raw" in
        *K) alloc_mem_g=$(awk -v m="${alloc_mem_raw%K}" 'BEGIN{printf "%.0f", m/1024/1024}') ;;
        *M) alloc_mem_g=$(awk -v m="${alloc_mem_raw%M}" 'BEGIN{printf "%.0f", m/1024}') ;;
        *G) alloc_mem_g="${alloc_mem_raw%G}" ;;
        *T) alloc_mem_g=$(awk -v m="${alloc_mem_raw%T}" 'BEGIN{printf "%.0f", m*1024}') ;;
        *)  alloc_mem_g="0" ;;
    esac

    users=$(squeue -h -w "$node" -o "%u|%b" | awk -F'|' '
        {
            user=$1
            gpu=0
            if (match($2, /gpu(:[A-Za-z0-9_-]+)?:[0-9]+/)) {
                s=substr($2, RSTART, RLENGTH)
                n=split(s, a, ":")
                gpu=a[n]
            } else if (match($2, /gres\/gpu=[0-9]+/)) {
                s=substr($2, RSTART, RLENGTH)
                sub(/.*=/, "", s)
                gpu=s
            }
            if (gpu > 0) cnt[user] += gpu
        }
        END {
            for (u in cnt) print u, cnt[u]
        }' | sort -k2,2nr -k1,1 | awk '{printf "%s(%s) ", $1, $2}')

    [ -z "$users" ] && users="-"

    printf "%-22s %-4s %-11s %-11s %-13s %s\n" \
        "$node" "$st" "${alloc_gpu}/${cfg_gpu}" "${alloc_cpu}/${cfg_cpu}" "${alloc_mem_g}/${cfg_mem_g}G" "$users"
done