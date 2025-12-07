#!/bin/bash
#SBATCH --job-name=spline_all
#SBATCH --output=report_4.out
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=cascade

module purge
module load mpi/openmpi/4.1.6/gcc/11
PYTHON_BIN=/opt/software/python/3.11.8/bin/python3

N=200000000
NE=200000000
THREADS=(2 4 8 16 48)
PROCS=(2 4 16 28)

echo "===== Job Info ====="
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "Nodes Allocated   = $SLURM_JOB_NUM_NODES"
echo "Tasks Allocated   = $SLURM_NTASKS"
echo "CPUs per Task     = $SLURM_CPUS_PER_TASK"
echo "==================="

# ----------------------------
# Clone sources
# ----------------------------
git clone https://github.com/eagerbeaver04/pararell_poisson.git 
cd pararell_poisson

# ----------------------------
# Compile all programs
# ----------------------------
mkdir build
cd build
cmake ..
make 
cd ../

echo
echo "=== RUN pthreads ==="
for t in "${THREADS[@]}"; do
  echo "Threads = $t"
  bash posix_run.sh t
done

# ----------------------------
# Run OpenMP (single node)
# ----------------------------
echo
echo "=== RUN OpenMP ==="
for t in "${THREADS[@]}"; do
  echo "Threads = $t"
  bash openmp_run.sh t
done

# ----------------------------
# Run MPI C (multi-node)
# ----------------------------
echo
echo "=== RUN MPI C ==="
for p in "${PROCS[@]}"; do
  echo "Procs = $p"
  bash mpi_run.sh p
done

# ----------------------------
# Run MPI Python (multi-node)
# ----------------------------
echo
echo "=== RUN MPI Python ==="
for p in "${PROCS[@]}"; do
  echo "Procs = $p"
  bash pu_mpi_run.sh p
done