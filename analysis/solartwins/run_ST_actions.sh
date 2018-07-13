#!/bin/bash
#SBATCH --job-name=actions
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=8044322869@vtext.com
#SBATCH -p cca,gen
#SBATCH --mem=0
#SBATCH -N1 --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=ST_actions_%j.log
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

#python ../../programs/compute_actions.py -i GAIA-ST.fits -o test.fits -p 4 
python ../../programs/compute_actions.py -i GAIA-ST.fits -o GAIA-ST_actions.fits -p $nprocs

date
