from mvpp_event import MVPPEvent
from mvpp_sequence import MVPPSequence

def parse_mvpp_event(s):
    ss = s.split("*")
    evt = MVPPEvent(dim=int(ss[0]), time = float(ss[1]))
    return evt

def parse_mvpp_sequence(s):
    seqId,seq_str = s.split(":")
    seqId = int(seqId)
    return MVPPSequence(seqId,map(parse_mvpp_event, seq_str.split(",")))
    
def read_sequences(filepath):
    lst_seq = []
 
    f = open(filepath,"r")
    for line in f:
        lst_seq.append(parse_mvpp_sequence(line.rstrip("\n")))
    return lst_seq

def read_sequences_rdd(sc,filepath,num_parts):
    rdd = sc.textFile(filepath,num_parts)
    return rdd.map(parse_mvpp_sequence)

def read_dim_dict(dictFilePath):
    f = open(dictFilePath,"r")
    dict_idx_dim = {}
    for line in f:
        s = line.rstrip("\n").split("*")
        dict_idx_dim[int(s[0])] = s[2]
    f.close()
    return dict_idx_dim