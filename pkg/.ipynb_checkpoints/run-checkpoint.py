import json
import csv
import re, os
from string import punctuation
import spacy
from collections import Counter
from tqdm import tqdm
from spacy_syllables import SpacySyllables
import numpy as np
from pandas import DataFrame

from datetime import datetime
from sklearn.cluster import OPTICS
import pickle


import argparse

base = "../data/tweets/tmp/data"

_stops = [s.strip() for s in 
          open('../data/shared/stopwords.en.basic',
               mode='r').readlines()]

_puncts = list(punctuation)
excl=['@', "#"]
for e in excl:
    _puncts.remove(e)

def build_nlp():

    nlp = spacy.load(f"en_core_web_sm")

    sentencizer = spacy.pipeline.Sentencizer(punct_chars=[".", "?",
                                                          "!", "。",
                                                          ":", "..."])

    nlp.add_pipe(sentencizer, name="sentencizer", before='tagger')

    merge_nps = nlp.create_pipe("merge_noun_chunks")
    nlp.add_pipe(merge_nps, after='tagger')

    merge_ents = nlp.create_pipe("merge_entities")
    nlp.add_pipe(merge_ents, after='tagger')

    syllables = SpacySyllables(nlp)
    nlp.add_pipe(syllables, name='syllables')
    return nlp


def remove_puncts(x, puncts=_puncts.copy()):
    if len(puncts) == 0:
        return x
    else:
        p = puncts.pop(0)
        x = x.replace(p, "")
        return remove_puncts(x, puncts)

def clean(s):
    patt0 = re.compile("http.*\S")
    patt1 = re.compile("www\..*?\.com")
    s = re.sub(patt0, "", s)
    s = re.sub(patt1, "", s)
    s = s.replace("@ ", "@").replace("# ", "#")
    s = re.sub(r'(…|\.\.\.|RT\W)', "", s)
    s = re.sub(r'(-|–)', ' ', s)
    s = s.replace("  ", " ").replace("   ", " ").replace("’", "")
    s = re.sub(r'(&amp;|&)', "and", s)
    s = re.sub(r'\d(\W|)(am|a.m.|A.M.|pm|p.m.|P.M.)', "", s)
    s = re.sub(r'(CST|PST|EST|PT|CT|ET|CDT|EDT|PDT)', "", s)
    s = re.sub(" \/ ", "", s)
    if s is not "" and s is not None:
        return s
    else:
        pass
    
def flesch_kincaid(n_words, n_sents, n_syllables, out):
    fa = (n_words/n_sents)
    fb = (n_syllables/n_words)
    calc = 206.835 - (1.015*fa) - (84.6*fb)
    if out == 'score':
        return calc
    
    elif out == 'level':
        conv = [(90.0, "5th Grade"), (80.0, "6th Grade"), 
                (70.0, "7th Grade"), (60.0, "8th/9th Grade"),
                (50.0, "10th - 12th Grade"), (30.0, "Some College"), 
                (10.0, "College Grad"), (0.0, "Professional")]
        grade = None
        while not grade:
            gtest = conv.pop(0)
            if calc < gtest[0]:
                pass
            else:
                grade = gtest[1]
        return grade

    
def tokenize(nlp, data):
    doc_arrs = []
    doc_sents_raw = []

    docs = []
    for didx in tqdm(range(len(data))):
        ddx = {}
        d = data[didx]['content']
        clean_str = clean(d)
        if clean_str is not None and clean_str.strip() != "":
            try:
                doc = nlp(clean_str)
            except Exception as err:
                
                print(err, clean_str)
                pass
            else:
                sent = [[x.text for x in s] for s in doc.sents]
                # in case you need it 
                n_sents = len(sent)
                tkns = []
                syllables_total = 0
                word_length_total = 0
                n_mentions = 0
                n_hashtags = 0

                doc_length_clean = 0
                doc_length_raw = 0
                for t in doc:
                    doc_length_raw += 1
                    if t.text.find("#") > -1:
                        # add to number of hashtags
                        n_hashtags += 1
                    elif t.text.find("@") > -1:
                        # add to number of mentions
                        n_mentions += 1
                    else:
                        doc_length_clean += 1
                        syllables_total += (0 if t._.syllables_count is None 
                                            else t._.syllables_count)
                        word_length_total += len(t.text)
                        if t.lower in _stops or t.lower in _puncts:
                            # remove extraneous punctuations and stop words
                            pass
                        else:
                            # remove punctuations from individual tokens
                            tx = remove_puncts(t.text)
                            tkns.append((tx, t.tag_))
                if doc_length_clean == 0:
                    pass
                else:
                    fk = flesch_kincaid(n_words=doc_length_clean,
                                        n_syllables=syllables_total,
                                        n_sents=n_sents, out='score')

                    avg_syllables = syllables_total/doc_length_clean
                    avg_word_length = word_length_total/doc_length_clean
                    doc_arr = [n_hashtags, n_mentions, avg_syllables, 
                               avg_word_length, n_sents, fk, doc_length_raw]

                    dateparts = data[didx]['date'].split(" ")
                    date = datetime.strptime(dateparts[0], '%Y-%m-%d')
                    time = datetime.strptime(dateparts[1], '%H:%M:%S')


                    doc_arrs.append(doc_arr)
                    ddx.update({"id":data[didx]['id'], "tokens": tkns, 
                                'content_raw': data[didx]['content'],
                                "date": dateparts[0], "time": dateparts[1], 
                                "lang_arr": doc_arr})

                    docs.append(ddx)
    return docs

               
def load_data(UPDATE=False):

    def flatten_arrs(arrs, n=0):
        if type(arrs[n]) is list:
            arr = arrs.pop(n)
            arrs.extend(arr)
        else:
            n = n+1
            flatten(arrs[n])

    arrs_file = os.path.join(base, "arrs.json")
    tkns_file = os.path.join(base, "tkns.json")
    cnts_file = os.path.join(base, "cnts.json")


    if all([os.path.isfile(a) for a in 
            [arrs_file, tkns_file, cnts_file]]) and UPDATE is False:
        arrs = json.load(open(arrs_file, mode='r'))
        tkns = json.load(open(tkns_file, mode='r'))
        cnts = json.load(open(cnts_file, mode='r'))
        print("data loaded.")

    else:

        dfile = os.path.join("../data/tweets/trump.full")
        fstream = open(dfile, mode='r')
        nlp = build_nlp()

        readr = csv.DictReader(fstream, 
                               fieldnames=["id","link","content","date",
                                           "retweets","favorites",
                                           "mentions","hashtags","geo"])

        next(readr)
        content = [r for r in readr]
        docs = tokenize(nlp, content)
        tkns, arrs = [], []
        for doc in docs:
            tkns.append(doc['tokens'])
            arrs.append(doc['lang_arr'])

        counts = Counter(flatten_arrs(tkns))
        cnts = dict(counts)
        json.dump(arrs, open(arrs_file, mode='w'))
        json.dump(tkns, open(tkns_file, mode='w'))
        json.dump(cnts, open(cnts_file, mode='w'))
        print("data saved.")
    return arrs




def cluster(data, UPDATE=False, *args, **kwargs):




    m = OPTICS(*args,**kwargs)
    modfile =os.path.join('../data/tweets/tmp/models',
                          f'{repr(m)}.fitted')
    modtext = repr(m)
    if os.path.isfile(modfile) and UPDATE is False:
        m = pickle.load(open(modfile, mode='rb'))
        print(f"{modtext} Loaded.")
    else:
        print(f"Fitting {modtext}...")
        m.fit(data)
        print(f"Fitted.\n")
        print(f"Pickling...")
        pickle.dump(m, open(modfile, mode='wb'))
        print(f'Saved.')
        
    return m


def summary(data, model):
    cols = ["avg_hashtags", "avg_mentions", 
            "avg_avg_syllables", "avg_avg_wordlen", "avg_sents", 
            "avg_readability", "avg_doclen_raw"]

    df = DataFrame(arrs, index=0, columns=['n_hashtags', 'n_mentions', 
                                           'avg_syllables', "avg_wordlen",
                                           'n_sents', 'readability', 
                                           'doclen_raw'])

    df.insert(0, "cluster", model.labels_)
    groups = df.groupby('cluster')

    print(groups[0].count())

    print(groups.describe())
    
    

if __name__ == "__main__":
    args_dict = dict()
    models_dir = "../data/tweets/tmp/models/"
    def set_args(arg_vals):
        available_args = [{"name":"min_samples", "type": int}, 
                          {"name":"max_eps", "type":float}, 
                          {"name":"metric", "type": str}, 
                          {"name": "p", "type":float}, 
                          {"name": "cluster_method", "type": str}, 
                          {"name": "eps", "type": float}, 
                          {"name": "xi", "type": float}, 
                          {"name":"min_cluster_size", "type": float}]
        print("Select number to set argument, select again to edit value.")
        for i in range(len(available_args)):
            print(f"{i})   {available_args[i]['name']}")
        argi = input(f"Enter Argument Number or 'done' to exit: ")
        if argi == 'done':
            return arg_vals
        else:
            argii = int(argi)
            argi_v = input(f"Enter in  {available_args[argii]['type']} value for argument {available_args[argii]['name']}: ")
            arg_vals.update({f"{available_args[argii]['name']}": available_args[argii]['type'].__call__(argi_v)})
            return set_args(arg_vals)
    
    def run():
        mods = [m for m in os.listdir(models_dir) if m != '.ipynb_checkpoints']
        for midx in range(len(mods)):
            print(f"{midx})  {mods[midx]}")
    
        mod_select = input("Select existing model or 'new' to create a new one: ")
        if mod_select.isdigit():
            mod_file = models_dir + mods[int(mod_select)]
            cluster_model = pickle.load(open(mod_file, mode='rb'))
        else:
            upd_data = input("Update Data (y/N)? ")
            upd_mod = input("Update Model (refit if exists) (y/N)? ")
            arrs = load_data(UPDATE=(True if upd_data == 'y' else False))
            print("\n\n Set Model Arguments \n\n")
            args = set_args(args_dict.copy())
            cluster_model = cluster(data=arrs, 
                                    UPDATE=(True if upd_mod == 'y' else False), 
                                    **args)
        
        mod_labels = cluster_model.labels_
        print(f"Clusters: {np.unique(mod_labels)}\nCounts: {Counter(mod_labels)}")

        cont_inp = input("[c]ontinue, [b]ack, or [q]uit? ").lower()
        if cont_inp == 'c':
            summary(data=arrs, model=cluster_model)
        elif cont_inp == 'b':
            return run()
        else:
            print("Done.")
    
    run()