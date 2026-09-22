

from math import sqrt
from tracemalloc import start
from mpi4py import MPI

start_time = MPI.Wtime()

def is_prime(num: int) -> bool:
    if num == 2:
        return True

    limit = int(sqrt(num))
    for divisor in range(2, limit + 1):
        if num % divisor == 0:
            return False

    return True

def count_primes(n: int, comm: MPI.Comm) -> int:
    if n < 2:
        return 0

    world_size = comm.Get_size()
    rank = comm.Get_rank()
    chunk_size = n // world_size
    count = 0
    start_num = max(2, chunk_size * rank)
    end_num = n if rank == world_size - 1 else start_num + chunk_size

    for num in range(start_num, end_num + 1):
        if is_prime(num):
            count += 1

    return count

comm = MPI.COMM_WORLD
rank_local_count = count_primes(1_000_000, comm)
global_count = comm.reduce(rank_local_count, MPI.SUM, root=0) # None for the non-root ranks

end_time = MPI.Wtime()
rank_local_time = end_time - start_time
total_time_taken = comm.reduce(rank_local_time, op=MPI.MAX, root=0)

if comm.Get_rank() == 0:
    print(f"Count: {global_count}")
    print(f"Time: {total_time_taken:3f}")
