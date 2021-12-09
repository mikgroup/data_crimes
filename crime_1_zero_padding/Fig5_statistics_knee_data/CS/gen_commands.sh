#~/bin/bash

echo "#~/bin/bash" > run.sh  # Notice: ">" writes into the first line of run.sh, while ">>" adds lines to it
echo " echo "start time:" " >> run.sh
echo "./print_timestamp.sh" >> run.sh
echo "OMP_NUM_THREADS=1" >> run.sh

echo "prep variables"


pad_ratio_VALS=(
1
1.25
1.5
1.75
2
)


num_slices=122

echo "run jobs"

echo "# run multiple jobs" >> run.sh

for pad_ratio in "${pad_ratio_VALS[@]}" ; do

              samp_type="weak"
	      logdir="./logs/${samp_type}_VD_pad_ratio_${pad_ratio}"       
	      echo "python3 3_CS_run_test.py --pad_ratio ${pad_ratio} --samp_type ${samp_type} --num_slices ${num_slices}  --logdir ${logdir} "  >> run.sh

	      samp_type="strong"
              logdir="./logs/${samp_type}_VD_pad_ratio_${pad_ratio}"
              echo "python3 3_CS_run_test.py --pad_ratio ${pad_ratio} --samp_type ${samp_type} --num_slices ${num_slices} --logdir ${logdir} "  >> run.sh

done


echo " echo "End time of DictL runs:"" >> run.sh
echo "./print_timestamp.sh" >> run.sh


chmod +x run.sh

echo "finished preparing the script run.sh"

#mv run.sh run_pad_${pad_ratio_VALS}_${samp_type}.sh

#echo "renamed run.sh to run_pad_${pad_ratio_VALS}_${samp_type}.sh"

#mv run.sh run_pad_${pad_ratio_VALS}_$samp.sh

#echo "renamed run.sh to run_pad_${pad_ratio_VALS}_$samp.sh"

#echo "end time:"
#timestamp



