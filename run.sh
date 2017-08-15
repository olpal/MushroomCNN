#!/bin/bash

#Variables
script_location='/scratch/aolpin/testing/code/training.py'
output_dir='/scratch/aolpin/testing/results/'
datset_dir='/scratch/aolpin/testing/dataset/data/'
#datset_dir='/tmp/testing/data/'
log_dir='/scratch/aolpin/testing/logs/'
training_batch_size=( )
learning_rate=( )
dropout=( )
conv_layers=( )
epochs=( )
neurons=( )
conv_size=( )
sigmoid_shift=( )
#Loop controls
min_tbs=128
max_tbs=128
min_lr=0.001
max_lr=0.001
min_drop=0.8
max_drop=0.8
min_lay=5
max_lay=5
min_epoch=2000
max_epoch=2000
min_neurons=60
max_neurons=60
min_conv=5
max_conv=5

#Populates all data arrays
function load_arrays(){
	#Load epochs
	while [  $min_epoch -le $max_epoch ]; do
		epochs=("${epochs[@]}" "$min_epoch")
		min_epoch=$(($min_epoch+100))
	done

	loop_test=$(bc <<< "$min_tbs<=$max_tbs")
	#load training batch sizes
	while [ $loop_test == 1 ]; do
		training_batch_size=("${training_batch_size[@]}" "$min_tbs")
		min_tbs=$( bc <<< "$min_tbs+64")
		loop_test=$(bc <<< "$min_tbs<=$max_tbs")
	done

	loop_test=$(bc <<< "$min_lr<=$max_lr")
	#load learing rates
	while [ $loop_test == 1 ]; do
		learning_rate=("${learning_rate[@]}" "$min_lr")
		min_lr=$( bc <<< "$min_lr*10")
		loop_test=$(bc <<< "$min_lr<=$max_lr")
	done

	loop_test=$(bc <<< "$min_drop<=$max_drop")
	#load drop out values
	while [ $loop_test == 1 ]; do
		dropout=("${dropout[@]}" "$min_drop")
		min_drop=$( bc <<< "$min_drop+0.1")
		loop_test=$(bc <<< "$min_drop<=$max_drop")
	done

	loop_test=$(bc <<< "$min_lay<=$max_lay")
	#load training batch sizes
	while [ $loop_test == 1 ]; do
		conv_layers=("${conv_layers[@]}" "$min_lay")
		min_lay=$( bc <<< "$min_lay+1")
		loop_test=$(bc <<< "$min_lay<=$max_lay")
	done

	loop_test=$(bc <<< "$min_neurons<=$max_neurons")
	#load neurons
	while [ $loop_test == 1 ]; do
		neurons=("${neurons[@]}" "$min_neurons")
		min_neurons=$( bc <<< "$min_neurons+10")
		loop_test=$(bc <<< "$min_neurons<=$max_neurons")
	done

	loop_test=$(bc <<< "$min_conv<=$max_conv")
	#load convolutions
	while [ $loop_test == 1 ]; do
		conv_size=("${conv_size[@]}" "$min_conv")
		min_conv=$( bc <<< "$min_conv+2")
		loop_test=$(bc <<< "$min_conv<=$max_conv")
	done
	
}

function check_directory(){
	if [ -z "$1" ]                           
	then
	     echo "No directory variable passed in"  
	else
	     if [ ! -d $1 ]
		then
		    echo "Creating directory: $1"
		    mkdir -p $1
		   else
		   	echo "Directory: $1 exists"
		fi
	fi
}

function run_normal(){
	echo "Running"
	for e in "${epochs[@]}"; do
		for t in "${training_batch_size[@]}"; do
	    	for c in "${conv_layers[@]}"; do
	    		for d in "${dropout[@]}"; do
	    			for l in "${learning_rate[@]}"; do
	    				for n in "${neurons[@]}"; do
	    					for f in "${conv_size[@]}"; do
			    				date_time=`date +%Y-%m-%d--%H-%M-%S-%N`
			    				echo "$script_location $datset_dir $output_dir$date_time $t $l $d $e $c $n $f > $log_dir$date_time.log"
			    				sqsub -r 4h -q gpu -f threaded -n 16 --gpp=8 --mpp=24g -o "$log_dir$date_time.log" python $script_location $datset_dir "$output_dir$date_time" $t $l $d $e $c $n $f
		    				done
		    			done
					done
				done
			done
		done
	done
}

#Execution section
load_arrays
#check_directory $out_directory
run_normal

