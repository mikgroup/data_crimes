#~/bin/bash
 echo start time: 
./print_timestamp.sh
OMP_NUM_THREADS=1
# run multiple jobs
python3 3_CS_run_test.py --pad_ratio 1 --samp_type weak --num_slices 122  --logdir ./logs/weak_VD_pad_ratio_1 
python3 3_CS_run_test.py --pad_ratio 1 --samp_type strong --num_slices 122 --logdir ./logs/strong_VD_pad_ratio_1 
python3 3_CS_run_test.py --pad_ratio 1.25 --samp_type weak --num_slices 122  --logdir ./logs/weak_VD_pad_ratio_1.25 
python3 3_CS_run_test.py --pad_ratio 1.25 --samp_type strong --num_slices 122 --logdir ./logs/strong_VD_pad_ratio_1.25 
python3 3_CS_run_test.py --pad_ratio 1.5 --samp_type weak --num_slices 122  --logdir ./logs/weak_VD_pad_ratio_1.5 
python3 3_CS_run_test.py --pad_ratio 1.5 --samp_type strong --num_slices 122 --logdir ./logs/strong_VD_pad_ratio_1.5 
python3 3_CS_run_test.py --pad_ratio 1.75 --samp_type weak --num_slices 122  --logdir ./logs/weak_VD_pad_ratio_1.75 
python3 3_CS_run_test.py --pad_ratio 1.75 --samp_type strong --num_slices 122 --logdir ./logs/strong_VD_pad_ratio_1.75 
python3 3_CS_run_test.py --pad_ratio 2 --samp_type weak --num_slices 122  --logdir ./logs/weak_VD_pad_ratio_2 
python3 3_CS_run_test.py --pad_ratio 2 --samp_type strong --num_slices 122 --logdir ./logs/strong_VD_pad_ratio_2 
 echo End time of DictL runs:
./print_timestamp.sh
