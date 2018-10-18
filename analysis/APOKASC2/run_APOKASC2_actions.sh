#!/bin/bash
#SBATCH --job-name=actions
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=number@vtext.com
#SBATCH -p cca 
#SBATCH --mem=0
#SBATCH -N1 --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=actions_%j.log
#SBATCH --array=0-5
pwd; hostname; date

source /mnt/home/abeane/.bash_profile
source activate dyndat 
#source activate pecact 

export OMP_NUM_THREADS=1 #this prevents conflicts in parallel libraries

echo $SLURM_TASKS_PER_NODE

export temp=`echo $SLURM_TASKS_PER_NODE | tr , " "`
export newtemp=`echo $temp | tr "(" " "`
export newsample=`echo $newtemp | tr ")" " "`
export nprocs=0
for n in $newsample;do
    if [[ $n == x* ]]
    then
        export mult="${n:1}"
        ((nprocs += $prev * ($mult-1)))
    else
        ((nprocs += $n))
    fi
    export prev=$n
done

echo $SLURM_ARRAY_TASK_ID

python ../../programs/compute_actions.py -i GAIA-APOKASC2.fits -o GAIA-APOKASC2_actions_noerrors.fits -p $nprocs --no-errors --node $SLURM_ARRAY_TASK_ID --tot 6 
##python ../../programs/compute_actions.py -i GAIA-APOKASC2.fits -o GAIA-APOKASC2_actions.fits -p $nprocs --node $SLURM_ARRAY_TASK_ID --tot 6

date
