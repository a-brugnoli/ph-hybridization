import argparse
import os 
import math
os.environ['OMP_NUM_THREADS'] = "1"
os.environ['DEIGEN_STACK_ALLOCATION_LIMIT'] = "0"

parser = argparse.ArgumentParser(description="Full Parser for simulation options")

# Define the expected arguments
parser.add_argument("--model", type=str, default="Wave", choices=["Wave", "Maxwell"], help="The considered system (Wave or Maxwell)")
parser.add_argument('--nel', type=int, nargs='+', default=[4], help="Array containing the number of elements of each side")
parser.add_argument('--degree', type=int, default=1, help="An integer parameter for the polynomial degree")
parser.add_argument('--ndim', type=int, default=3, choices=[2, 3])

parser.add_argument("--dt", type=float, default=0.01, help="Time step")
parser.add_argument("--t_end", type=float, default=1, help="Final time of the simulation")

parser.add_argument("--quad", action="store_true", help="Boolean for quadrilateral or hexahedral mesh (true if specified, false otherwise)")
parser.add_argument("--save_out", action="store_true", help="Boolean to save possible output files (true if specified, false otherwise)")

# Parse the command-line arguments
args, unknown = parser.parse_known_args()

model = args.model

nx = args.nel[0] if len(args.nel) else 1
ny = args.nel[1 if len(args.nel) > 1 else -1]
nz = args.nel[2 if len(args.nel) > 2 else -1]

pol_degree = args.degree
dim=args.ndim

quad = args.quad
save_out = args.save_out

time_step = args.dt
t_end = args.t_end
n_time_iter = math.ceil(t_end/time_step)
