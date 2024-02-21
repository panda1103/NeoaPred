import os

def seq2pdb(seq, out):
    L_seq = seq.upper()
    cmd = "PCcli -s "+L_seq+" -o " + out
    print(cmd)
    os.system(cmd)

