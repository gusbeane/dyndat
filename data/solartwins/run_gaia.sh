#!/bin/bash
#SBATCH --job-name=gaia_query
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=8044322869@vtext.com
#SBATCH --ntasks=1
#SBATCH --mem=8gb 
#SBATCH --time=2-00:00:00
#SBATCH --output=serial_test_%j.log
pwd; hostname; date

source /mnt/home/abeane/.bash_profile
source activate pecact

python gaia_query.py

date
