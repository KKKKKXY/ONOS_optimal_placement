#!/usr/bin/bash

echo $PWD
yourfilenames=`ls *.out`

for eachfile in $yourfilenames
do
    task_id=$(echo "$eachfile" | sed 's/slurm-//; s/\.out//')
    elapsed_time=$(grep "Elapsed time:" "$eachfile" | sed 's/Elapsed time://')
    echo "$task_id: $elapsed_time"  >> all_task_id.csv
done

# echo $(wc -l all_task_id.txt)