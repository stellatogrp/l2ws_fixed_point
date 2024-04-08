#!/bin/bash
#SBATCH --job-name=array-job     # create a short name for your job
#SBATCH --output=slurm-%A.%a.out # STDOUT file
#SBATCH --error=slurm-%A.%a.err  # STDERR file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=40G         # memory per cpu-core (4G is default)
#SBATCH --array=0             # job array with index values 0, 1, 2, 3, 4
#SBATCH --time=00:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=rajivs@princeton.edu # 
#SBATCH --gres=gpu:1 

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='true'
# export XLA_PYTHON_CLIENT_MEM_FRACTION='0.30'
# XLA_PYTHON_CLIENT_ALLOCATOR=platform
# export xla_force_host_platform_device_count=1


python benchmarks/l2ws_train.py robust_kalman cluster
# python l2ws_train_script.py robust_kalman cluster
# python aggregate_slurm_runs_script.py robust_ls cluster


# gpu command: #SBATCH --gres=gpu:1 