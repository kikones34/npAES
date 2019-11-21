from aes import SubBytes
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from textwrap import wrap

""" Helper functions """
str2bytes = lambda s: [int(b, 16) for b in wrap(s, 2)]

""" Hardcoded data """
# approx range in which all round1's lie 
round1_start = 15200
round1_end = 16200
round1_len = round1_end - round1_start

# precise PC in which round1 starts
# manually obtained, could automate
round1_start_pc = 68408

""" Struct-like class to hold traces data """
class Trace:
    def __init__(self, inpt, outpt):
        self.inpt = inpt
        self.outpt = outpt
        self.regs = {"pc": [], "rs": [[] for _ in range(13)]}

""" Parses all data, useful to gather metrics """
def parse_traces_full(folder):
    traces = []
    for file in glob.glob(os.path.join(folder, "*.idat")):
        print(f"Reading {file}")
        instr, outstr = file.split(".")[0].split("-")[1:]
        trace = Trace(str2bytes(instr), str2bytes(outstr))
        with open(file) as f:
            # discard first line (reg names)
            next(f)
            # reads lines until reaching round1_end
            for line in f:
                pc, _, _, _, *rs = line.strip().split()
                trace.regs["pc"].append(int(pc))
                for i, r in enumerate(rs):
                    trace.regs["rs"][i].append(int(r))
        traces.append(trace)
    return traces

""" Plot PCs to find out where the 1st round is approx. for all traces
    Conclusion (a ojímetro): start = 15200, end = 16200
"""
def plot_pcs(traces):
    for i, trace in enumerate(traces):
        #plt.yticks([])
        plt.plot(trace.regs["pc"])
        plt.savefig(f"pc-trace-{i}.png", dpi=150)
        plt.clf()

""" Plot general purpose registers to see if we can discard any
    Conclusion: can discard everything other than r0, r1, r2, r3
"""
def plot_regs(traces):
    for tidx, trace in enumerate(traces):
        for ridx, r in enumerate(trace.regs["rs"]):
            plt.title(f"min: {min(r)}, max: {max(r)}")
            plt.plot(r)
            plt.savefig(f"plots/r{ridx}/r{ridx}-trace-{tidx}.png", dpi=150)
            plt.clf()


""" Faster and more memory-efficient parsing which only reads the necessary data
    deduced from the full data metrics """
def parse_traces_min(folder):
    traces = []
    for file in glob.glob(os.path.join(folder, "*.idat")):
        print(f"Reading {file}")
        instr, outstr = file.split(".")[0].split("-")[1:]
        trace = Trace(str2bytes(instr), str2bytes(outstr))
        with open(file) as f:
            # discard first line (reg names)
            next(f)
            # skip to round1_start
            for _ in range(round1_start):
                next(f)
            # reads lines until reaching round1_end
            for _ in range(round1_end - round1_start):
                line = next(f)
                pc, _, _, _, *rs, _, _, _, _, _, _, _, _, _ = line.strip().split()
                trace.regs["pc"].append(int(pc))
                for i, r in enumerate(rs):
                    trace.regs["rs"][i].append(int(r))
        traces.append(trace)
    return traces

""" Aligns all traces so that the first pc is round1_start_pc """
def align_traces(traces):
    start_offsets = [trace.regs["pc"].index(round1_start_pc) for trace in traces]
    min_round1_len = round1_len - max(start_offsets)
    for trace, offset in zip(traces, start_offsets):
        trace.regs["pc"] = trace.regs["pc"][offset:offset+min_round1_len]
        trace.regs["rs"] = [trace.regs["rs"][i][offset:offset+min_round1_len] for i in range(4)]

""" Calculates Pearson correlation coefficient """
def correlate_vars(X, Y):
    xmean = np.mean(X)
    ymean = np.mean(Y)
    xdiffs = X - xmean
    ydiffs = Y - ymean

    num = np.sum(xdiffs*ydiffs)
    den = np.sqrt(np.sum(xdiffs**2)) * np.sqrt(np.sum(ydiffs**2))

    return num / den

""" Brute-forces a single byte of the key """
def find_key_byte(inpts, rs, nbyte):
    for byte in range(256):
        guesses = SubBytes(inpts[nbyte] ^ byte)
        for n, rn in enumerate(rs):
            for rnt in rn:
                c = correlate_vars(guesses, rnt)
                if abs(c) > 0.9:
                    return byte, c, n
    return None


""" Data organization:
    inpts[n] = byte n of input of each trace (list)
        len(inptss) = 16
        len(inptss[n]) = traces
    rs[n][t] = register rn at time t of each trace (list)
        len(rs) = 4
        len(rs[n]) = round1_len
        len(rs[n][t]) = traces
"""
if __name__ == "__main__":
    
    in_folder = os.path.join("samples", "traces")

    traces = parse_traces_min(in_folder)
    align_traces(traces)

    inpts = np.array([[trace.inpt[i] for trace in traces] for i in range(16)], dtype=np.uint32)
    rs = np.array([list(zip(*[trace.regs["rs"][i] for trace in traces])) for i in range(4)], dtype=np.uint32)

    key = [0]*16
    for i in range(16):
        print(f"Finding key byte {i}...")
        sol, c, n = find_key_byte(inpts, rs, i)
        if sol:
            print(f"Found! k_{i} = {sol} in r{n} with correlation {c:.2f}")
            key[i] = sol
        else:
            print(f"Coult not find a solution for k_{i}")
    print(f"Full key: {key}")
