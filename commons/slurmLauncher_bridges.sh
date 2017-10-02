#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM
#SBATCH --mail-user=batmanghelich@gmail.com

./etc/profile.d/modules.sh

echo "$@"
"$@"
