""" Skript to convert previous buffer format to new one.
"""

import numpy as np
import sys
import argparse

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path")

    args = parser.parse_args(argv)

    state = np.load(f"{args.path}_state.npy")
    action = np.load(f"{args.path}_action.npy")
    next_state = np.load(f"{args.path}_next_state.npy")
    reward = np.load(f"{args.path}_reward.npy")
    not_done = np.load(f"{args.path}_not_done.npy")
    ptr = np.load(f"{args.path}_ptr.npy")

    print(f"Replay Buffer loaded with {state.shape[0]} elements.")

    contrainer = { "state": 	 state,
                    "action": 	 action,
                    "next_state": next_state,
                    "reward":	 reward,
                    "not_done":   not_done,
                    "ptr": 		 ptr}

    np.savez(args.path, **contrainer)
    
if __name__ == "__main__":
    main(sys.argv[1:])

