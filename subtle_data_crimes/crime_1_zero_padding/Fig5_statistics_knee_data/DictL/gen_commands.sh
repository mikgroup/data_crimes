#~/bin/bash

echo "#~/bin/bash" > run.sh  # Notice: ">" writes into the first line of run.sh, while ">>" adds lines to it
echo " echo "start time:" " >> run.sh
echo "./print_timestamp.sh" >> run.sh
echo "OMP_NUM_THREADS=1" >> run.sh

echo "prep variables"


block_shape_MIN=8
block_shape_MAX=32
block_shape_STEP=8

num_filters_MIN=100
num_filters_MAX=300
num_filters_STEP=100

nnz_MIN=5
nnz_MAX=11
nnz_STEP=2

max_iter_MIN=7
max_iter_MAX=13
max_iter_STEP=2

num_filters_VALS=`seq ${num_filters_MIN} ${num_filters_STEP} ${num_filters_MAX}`
nnz_VALS=`seq ${nnz_MIN} ${nnz_STEP} ${nnz_MAX}`
max_iter_VALS=`seq ${max_iter_MIN} ${max_iter_STEP} ${max_iter_MAX}`
block_shape_VALS=`seq ${block_shape_MIN} ${block_shape_STEP} ${block_shape_MAX}`
lamda_VALS=(
1e-5
1e-4
1e-3
1e-2
)
pad_ratio_VALS=(1.75)
samp_type="strong"
num_slices=5
# (1,1.25,1.5,1.75,2)

echo "run jobs"

echo "# run multiple jobs" >> run.sh

for pad_ratio in "${pad_ratio_VALS[@]}" ; do
	#echo "pad_ratio $pad_ratio"
	for lamda in "${lamda_VALS[@]}" ; do
		#echo "lamda $lamda"
		for block_shape in ${block_shape_VALS[@]} ; do
			#echo "block_shape $block_shape"
			for num_filters in ${num_filters_VALS[@]} ; do
				#echo "num_filters $num_filters"
				for nnz in ${nnz_VALS[@]} ; do
					#echo "nnz $nnz"
					for max_iter in ${max_iter_VALS[@]} ; do
						#echo "max_iter $max_iter"

						#echo "$samp_type VD pad_ratio=$pad_ratio lamda=$lamda block_shape=$block_shape=num_filters $num_filters=nnz $nnz max_iter=$max_iter" 

						      logdir="./logs/${samp_type}_VD_pad_ratio_${pad_ratio}_lamda_${lamda}_block_shape_${block_shape}_num_filters_${num_filters}_nnz_${nnz}_max_iter_${max_iter}"			
					        

						#FILE=`find $logdir -name "*.npz"`

                                                #if [ -d $logdir ] &&  [ -f "$FILE" ]
						#then
						#	echo "yes - folder exists"

						
						      

						      FILE=`find $logdir -name "*.npz"`
						      if [ -f "$FILE" ]; then
						           echo "yes - .npz file exists" 
                  				      else			                         


        					      echo "python3 calib_ML_knee_data_v8.py --pad_ratio ${pad_ratio} --samp_type ${samp_type} --lamda ${lamda} --block_shape ${block_shape} --num_filters ${num_filters} --nnz ${nnz} --max_iter ${max_iter} --num_slices $num_slices --logdir ${logdir}" >> run.sh


		   				      #      echo "python3 calib_ML_knee_data_v5.py --pad_ratio ${pad_ratio} --samp_type ${samp_type} --lamda ${lamda} --block_shape ${block_shape} --num_filters ${num_filters} --nnz ${nnz} --max_iter ${max_iter} --num_slices $num_slices --logdir ${logdir}" >> run.sh
                                                      fi
						#fi
					done
				done
			done
		done
	done
done


echo " echo "End time of DictL runs:"" >> run.sh
echo "./print_timestamp.sh" >> run.sh


chmod +x run.sh

echo "finished preparing the script run.sh"

mv run.sh run_pad_${pad_ratio_VALS}_${samp_type}.sh

echo "renamed run.sh to run_pad_${pad_ratio_VALS}_${samp_type}.sh"

#mv run.sh run_pad_${pad_ratio_VALS}_$samp.sh

#echo "renamed run.sh to run_pad_${pad_ratio_VALS}_$samp.sh"

#echo "end time:"
#timestamp



