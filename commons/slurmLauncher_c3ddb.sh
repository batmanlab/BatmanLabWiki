#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -p defq
#SBATCH --exclusive
#SBATCH --mail-user=batmanghelich@gmail.com

./etc/profile.d/modules.sh

echo "$@"
"$@"
