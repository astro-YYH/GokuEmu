import math
import numpy as np

def get_pairs_done(output_file):
    pairs_done = np.loadtxt(output_file, usecols=(0,1))
    return pairs_done

def get_num_pairs(n_sample_LF, n_sample_HF, len_slice):
    num_pairs = []
    for num_LF in range(0, n_sample_LF+1, len_slice):
        for num_HF in range(0, n_sample_HF+1, len_slice):
            if num_LF == 0 or num_HF == 0:
                continue
            num_pairs.append([num_LF, num_HF])
    return num_pairs

def write_submit(data_dir, L1HF_base, L2HF_base, num_LF, num_HF, n_optimization_restarts, partition, output_file, prefix="error_submit"):
    submit_name = f"{prefix}_{num_LF}LF_{num_HF}HF.sh"
    with open(submit_name, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name=err{num_LF}L{num_HF}H\n")
        f.write("#SBATCH --time=2-00:00:00\n")
        f.write("#SBATCH --mem=7G\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --cpus-per-task=1\n")
        f.write("#SBATCH --ntasks=1\n")
        # f.write("#SBATCH --mail-type=ALL\n")
        # f.write("#SBATCH --mail-user=<email>\n")
        f.write("#SBATCH --partition={}\n".format(partition))
        f.write("\n")
        f.write("hostname\n")
        f.write("date\n")
        f.write("echo "+f"job-name=err{num_LF}L{num_HF}H\n")
        f.write("python -u dgmgp_error.py ")
        f.write(f"--data_dir={data_dir} ")
        f.write(f"--L1HF_base={L1HF_base} ")
        f.write(f"--L2HF_base={L2HF_base} ")
        f.write(f"--num_LF={num_LF} ")
        f.write(f"--num_HF={num_HF} ")
        f.write(f"--n_optimization_restarts={n_optimization_restarts} ")
        f.write(f"--parallel=0 ")
        f.write(f"--output_file={output_file} ")
        f.write("\n")
        f.write("date\n")

def write_submit_frontera(data_dir, L1HF_base, L2HF_base, num_pairs, n_optimization_restarts, output_file, ntasks: int = 28,partition="small",submit_prefix='error_submit'):
    # print(len(num_pairs))
    # print(ntasks)
    n_submit = math.ceil(len(num_pairs)/ntasks)
    for i in range(n_submit):
        submit_name = f"{submit_prefix}_frontera_{i}.sh"
        with open(submit_name, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --job-name=err_{i}\n")
            f.write("#SBATCH --time=1-00:00:00\n")
            # f.write("#SBATCH --mem=200G\n")
            f.write("#SBATCH --nodes=1\n")
            # f.write("#SBATCH --cpus-per-task=2\n")
            f.write("# SBATCH --ntasks-per-node=56\n")
            f.write("#SBATCH -A AST21005\n")
        # f.write("#SBATCH --mail-user=<email>\n")
            f.write("#SBATCH --partition={}\n".format(partition))
            f.write("\n")
            f.write("hostname\n")
            f.write("source ~/.bashrc\n")
            f.write("conda activate gpy-env\n")
            f.write("date\n")
            f.write("echo "+f"job-name=err_{i}\n")
            for num_pair in num_pairs[i*ntasks:min(i*ntasks+ntasks, len(num_pairs))]:
                num_LF = num_pair[0]
                num_HF = num_pair[1]
                f.write("python -u dgmgp_error.py ")
                f.write(f"--data_dir={data_dir} ")
                f.write(f"--L1HF_base={L1HF_base} ")
                f.write(f"--L2HF_base={L2HF_base} ")
                f.write(f"--num_LF={num_LF} ")
                f.write(f"--num_HF={num_HF} ")
                f.write(f"--n_optimization_restarts={n_optimization_restarts} ")
                f.write(f"--parallel=0 ")
                f.write(f"--output_file={output_file} &")
                f.write("\n")
            f.write("wait\n")
            f.write("date\n")

def find_row(array, target_row):
    for row in array:
        if row == target_row:
            return True
    return False

data_dir="../data/narrow" # hpcc
# data_dir="/work2/01317/yyang440/frontera/tentative_sims/data_for_emu" # frontera
L1HF_base="matter_power_378_Box100_Part75_27_Box100_Part300"
L2HF_base="matter_power_378_Box25_Part75_27_Box100_Part300"

n_sample_LF = 378
n_sample_HF = 27
len_slice = 3
n_optimization_restarts = 20
output_file = "error_function_goku_narrow_pre_frontera.txt"
cluster = "frontera"  # frontera hpcc
submit_prefix = "error_submit_narrow"
restart = False

pairs_done = get_pairs_done(output_file).tolist() if restart == True else []

num_pairs = get_num_pairs(n_sample_LF, n_sample_HF, len_slice) # [[],[],...]

partition_threshold = 4/4 * len(num_pairs)
partition = "intel"

if cluster != "frontera":
    for num_pair in num_pairs:
        if find_row(pairs_done, num_pair):
            continue
        # print("writing")
        if num_pairs.index(num_pair) > int(partition_threshold):
            partition = "batch"
        assert len(num_pair) == 2
        num_LF = num_pair[0]
        num_HF = num_pair[1]
        assert num_LF%len_slice == 0
        assert num_HF%len_slice == 0
        write_submit(data_dir, L1HF_base, L2HF_base, num_LF, num_HF, n_optimization_restarts, partition, output_file, prefix=submit_prefix)
else:
    write_submit_frontera(data_dir, L1HF_base, L2HF_base, num_pairs, n_optimization_restarts, output_file, ntasks=50,submit_prefix=submit_prefix)

