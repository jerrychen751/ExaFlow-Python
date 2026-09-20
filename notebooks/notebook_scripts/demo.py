
# Suppose that we want to find sum(x^2 for x in range(0, 20,000,000))
# We compare the time it takes to parallelize this with 4 processes vs. 1

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # the identity of the process within world
world_size = comm.Get_size() # number of processes total

N = 20_000_000
chunk_size = N // world_size

start_idx = rank * chunk_size
end_idx = N if rank == world_size - 1 else start_idx + chunk_size # exclusive end

comm.Barrier() # pauses every process until each one reaches this point in code (since process startup order is random and they may finish at different paces)

start_time = MPI.Wtime()

rank_local_sum = 0
for i in range(start_idx, end_idx):
    rank_local_sum += i ** 2

global_sum = comm.reduce(rank_local_sum, op=MPI.SUM, root=0) # take sum of each rank-local sum and give it to rank=0 process
end_time = MPI.Wtime()
time_spent = end_time - start_time

if rank == 0:
    print(f"Result: {global_sum}")
    print(f"Time Spent: {time_spent:.3f} seconds")
