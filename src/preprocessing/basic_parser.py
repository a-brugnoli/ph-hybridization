import argparse
import os 
os.environ['OMP_NUM_THREADS'] = "1"
os.environ['DEIGEN_STACK_ALLOCATION_LIMIT'] = "0"

parser = argparse.ArgumentParser(description="Basic Parser for simulation options")

# Define the expected arguments
parser.add_argument("--quad", action="store_true", help="Boolean for quadrilateral or hexahedral mesh (true if specified, false otherwise)")
parser.add_argument("--save_out", action="store_true", help="Boolean to save possible output files (true if specified, false otherwise)")

# Parse the command-line arguments
args, unknown = parser.parse_known_args()

quad = args.quad
save_out = args.save_out

if save_out:
    print("File will be saved in your home in the directory StoreResults")
