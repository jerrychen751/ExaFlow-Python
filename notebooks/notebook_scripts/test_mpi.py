# Parallelization

import time

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

N = 10
chunk_size = N // world_size
start_idx = rank * chunk_size
end_idx = N if rank == world_size - 1 else start_idx + chunk_size

comm.Barrier()

start_time = MPI.Wtime()

rank_local_sum = 0
for i in range(start_idx, end_idx):
    for j in range(2, N-1):
            if i % j == 0:
                rank_local_sum += 1
            break

global_sum = comm.reduce(rank_local_sum, op=MPI.SUM, root=0)

end_time = MPI.Wtime()
time_spent = end_time - start_time

print(f"Result: {global_sum}")
print(f"Time Spent: {time_spent:.3f} seconds")