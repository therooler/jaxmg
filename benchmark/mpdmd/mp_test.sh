num_processes=4
# export CUDA_VISIBLE_DEVICES="0"
range=$(seq 0 $(($num_processes - 1)))
HOSTS=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
echo $HOSTS
MASTER=${HOSTS[0]}
for i in $range; do
  python -u mp_test.py "$MASTER:10001" $i $num_processes > /tmp/toy_$i.out &
done

wait

for i in $range; do
  echo "=================== process $i output ==================="
  cat /tmp/toy_$i.out
  echo
done