#$ -l mem=15G
#$ -pe smp 32
#$ -l h_rt=1:0:0
#$ -R y
#$ -S /bin/bash
#$ -wd /home/zongchen/
#$ -j y
#$ -N nest_bq_toy

JOB_PARAMS=$(sed "${SGE_TASK_ID}q;d" "$1")
echo "Job params: $JOB_PARAMS"

conda activate nest_bq
date

## Check if the environment is correct.
which pip
which python

python /home/zongchen/nest_bq/toy.py $JOB_PARAMS