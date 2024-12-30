#!/bin/bash

#set job name:
#SBATCH -J opt

#128 tasks on 4 nodes:
#SBATCH -N 1

#one process for each physical core:
#SBATCH -c 2

#run on the broadwell partition (pax11):
#SBATCH -p broadwell

#runtime of 20 minutes:
#SBATCH -t 47:59:59

#copy environment variables from submit environment:
#SBATCH --get-user-env

#send mail on all occasions:
#SBATCH --mail-type=ALL

./one
