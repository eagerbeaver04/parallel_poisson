# echo 0 | sudo tee /proc/sys/kernel/randomize_va_space -- may change parameter to fix it ( running under gdb has no segfaults)
time; mpiexec -n 4 ./build/main_executable_mpi