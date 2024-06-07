from mpi4py import MPI

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Rank of the current process
    size = comm.Get_size() # Total number of processes

    # Print rank and size
    print(f"Hello from rank {rank} out of {size}")

    # Broadcast a message from the process with rank 0 to all other processes
    if rank == 0:
        # Only rank 0 executes this block
        data = "Hello, MPI world"
    else:
        # Data will be overwritten by the broadcast for all other ranks
        data = None

    # Broadcast the data to all processes
    data = comm.bcast(data, root=0)

    # Print the data received via broadcast on all processes
    print(f"Rank {rank} received data: '{data}'")

if __name__ == "__main__":
    main()
