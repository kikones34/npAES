from aes import ShiftRows, SubBytes, sbox, pmul, gen_params, generate_keys, RotWord, SubWord, rcon
import numpy as np
from functools import lru_cache

# this optimization reduces execution time by an order of magnitude
pmul = lru_cache(maxsize=256*4)(pmul)

""" File parsing """

def parse_faults(file):
    with open(file) as f:
        correct, *faults = [np.fromiter(map(lambda n: int(n, 16), line.strip().split(",")), dtype=np.uint8).reshape(4, 4).T for line in f]
    return correct, faults


""" Fault classification """

ones_col = np.array([1, 1, 1, 1], dtype=np.uint8, ndmin=2).T
patterns = [ShiftRows(np.pad(ones_col, ((0, 0), (i, 3-i)))) for i in range(4)]

# classify differential faults depending on the pattern they match
# each pattern reveals a different column of the key
def classify_diff_faults(diff_faults):
    classd_faults = [[],[],[],[]]
    for fault in diff_faults:
        for pidx, pattern in enumerate(patterns):
            if np.all((fault != 0) == pattern):
                classd_faults[pidx].append(fault[pattern == 1].tolist())
                break
        else:
            print(f"Fault doesn't match any pattern!\n{fault}")
    return classd_faults


""" Internal state solvers """

coefs_list = [[2, 1, 1, 3], [3, 2, 1, 1], [1, 3, 2, 1], [1, 1, 3, 2]]

def check_formula(coef, x, fault, diff_fault):
    return sbox[x ^ pmul(coef, fault)] == sbox[x] ^ diff_fault

def find_solutions(diff_fault, fault_row):
    coefs = coefs_list[fault_row]
    solutions = []
    for fault in range(256):
        for x1 in range(256):
            if check_formula(coefs[0], x1, fault, diff_fault[0]):
                for x2 in range(256):
                    if check_formula(coefs[1], x2, fault, diff_fault[1]):
                        for x3 in range(256):
                            if check_formula(coefs[2], x3, fault, diff_fault[2]):
                                for x4 in range(256):
                                    if check_formula(coefs[3], x4, fault, diff_fault[3]):
                                        solutions.append((x1, x2, x3, x4))
    return set(solutions)

# Intersect solutions until there is only 1 left (deterministic method)
total_faults = 0 # variable only used for printing
def find_unique_solution_rec(remaining_faults, solution_set):
    if not remaining_faults:
        return None
    
    for fault_row in range(4):
        print(f"Trying fault {total_faults - len(remaining_faults)} at row {fault_row}")
        new_solution_set = find_solutions(remaining_faults[0], fault_row)
        if not new_solution_set:
            continue
        
        if solution_set != None:
            new_solution_set &= solution_set
            if not new_solution_set:
                continue

            if len(new_solution_set) == 1:
                return new_solution_set
        
        final_solution_set = find_unique_solution_rec(remaining_faults[1:], new_solution_set)
        if final_solution_set:
            return final_solution_set
    
    return None

def find_unique_solution(diff_faults):
    global total_faults
    total_faults = len(diff_faults)
    solution = find_unique_solution_rec(diff_faults, None)
    if not solution:
        return None
    return solution.pop()

# Alternative probabilistic method to solve the state matrix
import collections
import random
def find_most_common_solution(diff_faults):
    solutions = []
    for i, df in enumerate(diff_faults):
        fault_row = random.randint(0, 3)
        print(f"Finding solutions for fault {i} at row {fault_row}...")
        solutions.extend(find_solutions(df, fault_row))
        
    tally = collections.Counter(solutions).items()
    mc_sol = max(tally, key=lambda t: t[1])
    
    return mc_sol[0]

# Combines all solutions into the full state matrix
def find_full_solution(classd_diff_faults, solver):
    full_solution = np.zeros((4,4), dtype=np.uint8)
    for c in range(4):
        print(f"Finding solutions in column {c}...")
        sol = solver(classd_diff_faults[c])
        if sol:
            print(f"Found solution: {sol}")
        else:
            print(f"Could not find solution.")
        full_solution[:,c] = sol
    return full_solution

# TODO: make it work for 192 and 256 bits
def reverse_key_schedule(last_key, last_key_idx, Nk, Nr):
    nwords = (Nr + 1) * 4
    curr_key = np.copy(last_key)
    for i in reversed(range(Nk, nwords)):
        pos = i % Nk
        if pos == 0:
            curr_key[:,pos] ^= (SubWord(RotWord(curr_key[:,3])) ^ rcon[i//Nk]).astype(np.uint8)
        else:
            curr_key[:,pos] ^= curr_key[:,pos-1]
    return curr_key

# helper function to retrieve a 128 bit key from the state matrix solution
def get_key(full_solution):
    k10 = ShiftRows(SubBytes(full_solution)) ^ correct
    return reverse_key_schedule(k10, 10, 4, 10)

if __name__ == "__main__":

    in_file = "outputs_DFA_AES128.dat"
    solver = find_unique_solution
    rk_tests = False

    vhex = np.vectorize("0x{:02x}".format)

    correct, faults = parse_faults(in_file)
    diff_faults = list(map(lambda fault: correct ^ fault, faults))
    classd_diff_faults = classify_diff_faults(diff_faults)
    full_solution = find_full_solution(classd_diff_faults, solver)
    print(f"State matrix solution:\n{full_solution}")
    key = get_key(full_solution)
    print(f"Recovered cipher key:\n{vhex(key)}")

    if rk_tests:
        """ Reverse key schedule tests """
        def do_reverse_key_test(cipher_key):
            Nk, Nr = gen_params(cipher_key)
            round_keys = generate_keys(cipher_key, Nk, Nr)
            recovered_key = reverse_key_schedule(round_keys[-1], Nr, Nk, Nr)
            np_cipher_key = np.array(cipher_key, dtype=np.uint8).reshape(4,4).T
            if np.all(np_cipher_key == recovered_key):
                print("Correctly recovered key!")
            else:
                print(f"Could not recover key.\nCorrect key:\n{np_cipher_key}\nRecovered key:\n{recovered_key}")

        # 128 bits
        print("Testing reverse key scheduling with 128 bits...")
        do_reverse_key_test(range(16))
