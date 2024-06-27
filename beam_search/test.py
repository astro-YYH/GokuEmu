from mpi4py import MPI
import numpy as np

def parallel_sum(data):
    """
    Function to compute the sum of an array in parallel using MPI.

    Parameters:
    data (numpy array): The data array to be summed up.

    Returns:
    total_sum (int): The total sum of the array (only valid on root process).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Determine the size of each chunk for each process
    chunk_size = len(data) // size

    # Create a buffer for each process to receive its chunk of the array
    recvbuf = np.empty(chunk_size, dtype='i')

    # Scatter the array to all processes
    comm.Scatter(data, recvbuf, root=0)

    # Each process computes the sum of its chunk
    local_sum = np.sum(recvbuf)

    # Gather all local sums to the root process
    total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    return total_sum

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Initialize the array in the root process (rank 0)
    data = None

    array_size = 100
    data = np.arange(array_size, dtype='i')

    # Call the parallel_sum function
    total_sum = parallel_sum(data)

    # The root process prints the total sum
    if rank == 0:
        print(f"Total sum is {total_sum}")