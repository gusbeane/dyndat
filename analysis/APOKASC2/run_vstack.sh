#!/bin/bash
#SBATCH --job-name=actions
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=8044322869@vtext.com
###SBATCH -p cca
#SBATCH --mem=0
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --output=actions_%j.log
pwd; hostname; date

source /mnt/home/abeane/.bash_profile
source activate dyndat 
#source activate pecact 

export OMP_NUM_THREADS=1 #this prevents conflicts in parallel libraries

#python vstack_tables.py GAIA-APOKASC2_actions_noerrors.fits GAIA-APOKASC2_actions_noerrors-*
python vstack_tables.py GAIA-APOKASC2_actions.fits GAIA-APOKASC2_actions-*

date
