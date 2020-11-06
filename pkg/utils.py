import csv
import os

def load_full(t):
    if t == 'full':
        DATA_FILE = "../data/tweets/trump.full"
    else:
        DATA_FILE = '../data/tweets/trump.mini'
        if os.path.isfile(DATA_FILE) is False:
            return write_mini(dat=load_full(t='full'))
        else:
            pass
    stream = open(DATA_FILE, mode='r')
    data = csv.DictReader(stream, fieldnames=["id","link","content","date",
                                              "retweets","favorites",
                                              "mentions","hashtags","geo"])
    next(data)
    return data

def write_mini(dat, n=100, rtn=False):
    MINI_FILE = '../data/tweets/trump.mini'
    ministream = open(MINI_FILE, mode='w')
    writr = csv.DictWriter(ministream, 
                           fieldnames=["id","link","content","date",
                                       "retweets","favorites",
                                       "mentions","hashtags","geo"])
    i, data = 0, []
    while i <= n:
        d.append(next(dat))
        i+=1
    writr.writerows(data)
    return data


