import os
import argparse
parser = argparse.ArgumentParser(description="training codes")
parser.add_argument("--datasets", type=str, default="/path/to/your/datasets",
                    help="datasets path")
args = parser.parse_args()
## Set get_all_args.datadir
for file in os.listdir("./configs"):
    print(f"configs/{file}")
    with open(f"configs/{file}","r") as f:
        lines = f.readlines()
        sub_lines = [lin for lin in lines if "get_args.data_dir" in lin]
        if len(sub_lines) != 0:
            index = lines.index(sub_lines[0])
            # lines[index] = f'get_all_args.datadir = "{/path/to/your/datasets/}"\n'
            lines[index] = f'get_args.data_dir = "{args.datasets}"\n'
    with open(f"configs/{file}","w+") as f:
        f.writelines(lines)
