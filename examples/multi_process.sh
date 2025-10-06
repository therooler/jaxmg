#!/bin/bash

export JAX_NUM_CPU_DEVICES=4
num_processes=2

range=$(seq 0 $(($num_processes - 1)))

for i in $range; do
  python multi_process.py $i $num_processes > /tmp/multi_process_$i.out &
done

wait

for i in $range; do
  echo "=================== process $i output ==================="
  cat /tmp/multi_process_$i.out
  echo
done