#!/bin/bash
#$ -pe smp 4
source activate tf
python /home/davide/dev/test/test.py -f=shit.txt
source deactivate
