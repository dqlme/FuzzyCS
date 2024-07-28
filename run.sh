#!/bin/bash

#JSUB -n 1
#JSUB -q gpu
#JSUB -gpgpu 1
##JSUB -m gpu06

source /apps/software/anaconda3/etc/profile.d/conda.sh
conda activate pytorch


python main.py -m random -c femnist.yaml > out/femnist/random_femnist_715.out 2>&1
python main.py -m AFL -c femnist.yaml > out/femnist/afl_femnist_715.out 2>&1
# python main.py -m random -c femnist.yaml > out/harbox/random_femnist_710.out 2>&1
python main.py -m Powd -c femnist.yaml > out/femnist/pwd_femnist_715.out 2>&1
# python main.py -m AFL -c harbox.yaml > out/harbox/afl_harbox_710.out 2>&1
python main.py -m MCDM -c femnist.yaml > out/femnist/mcdm_femnist_715.out 2>&1
python main.py -m DELTA -c femnist.yaml > out/femnist/delta_femnist_715.out 2>&1
# python main.py -m DivFL -c femnist.yaml > out/femnist/divfl_femnist_715.out 2>&1

python main.py -m random -s 1 -c femnist.yaml > out/femnist/random_femnist_715_s1.out 2>&1
python main.py -m AFL -s 1 -c femnist.yaml > out/femnist/afl_femnist_715_s1.out 2>&1
# python main.py -m random -c femnist.yaml > out/harbox/random_femnist_710.out 2>&1
python main.py -m Powd -s 1 -c femnist.yaml > out/femnist/pwd_femnist_715_s1.out 2>&1
# python main.py -m AFL -c harbox.yaml > out/harbox/afl_harbox_710.out 2>&1
python main.py -m MCDM -s 1 -c femnist.yaml > out/femnist/mcdm_femnist_715_s1.out 2>&1
python main.py -m DELTA -s 1 -c femnist.yaml > out/femnist/delta_femnist_715_s1.out 2>&1