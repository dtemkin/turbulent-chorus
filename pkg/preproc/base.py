import os, sys
from collections import Counter, UserDict, UserList
import json, csv, pickle
import random
import re
from datetime import datetime
from string import punctuation
import spacy
from spacy_syllables import SpacySyllables
import numpy as np
from pandas import DataFrame
from sklearn.cluster import OPTICS
import seaborn as sns
from tqdm import tqdm
from pkg.preproc import utils


sys.setrecursionlimit(60000)
dirname=os.path.dirname

base_tmp = os.path.join(dirname(dirname(dirname(__file__))), 'tmp')
base_data = os.path.join(base_tmp, 'data')
base_3rdparty = os.path.join(base_data, '3rdparty')
base_models = os.path.join(base_tmp, 'models')
_stops = [s.strip() for s in open(os.path.join(base_data, 'stopwords.en.basic'),
                                  mode='r').readlines()]


def fetch_tweets(twitter_handle, syr=2009, eyr=2020, 
                 excl_retweets=True, js=[], max_retries=3):
    final_js = js
    base_url = "http://www.trumptwitterarchive.com/data/{twitter_handle}/{yr}.json"
    for yr in range(syr, eyr+1):
        print(f"Getting {twitter_handle} - {yr}")
        url = base_url.format(twitter_handle=twitter_handle, yr=yr)
        try:
            resp = requests.get(url)
        except Exception as err:
            print(f"Invalid year for {twitter_handle}. {err}")
            pass
        else:
            try:
                jsx = resp.json()
            except Exception as err2:
                print(f"Error could not find {resp.url}")
            else:
                if excl_retweets:
                    final_js.extend([row for row in jsx if 
                                     row['text'][0] != '"' or 
                                     row['text'][:2] not in ["RT", "rt"]])
                else:
                    final_js.extend(jsx)
    json.dump(final_js, open(f"../tmp/3rdparty/{twitter_handle}-{('wort' if excl_retweets else 'wrt')}"+"_twitter_archive.json", mode='w'))


class Data(object):
    
    def __init__(self):
        self._sample = []
        self._data = []
    
    @property
    def data(self):
        return self._data
    
    @property
    def sample(self):
        return self._sample
    
    @staticmethod
    def is_retweet(s):
        try:
            is_retweet = s.pop("is_retweet")
        except:
            is_retweet = (True if s['text'][0:4].find("RT") > -1 
                          else False)
        else:
            if not is_retweet:
                is_retweet = (True if s['text'][0:4].find("RT") > -1 
                              else is_retweet)
            else:
                pass
        return is_retweet

    @staticmethod
    def get_mentions(s):
        mention_patt = re.compile("@\w*\s")
        men = re.findall(mention_patt, s)
        return men

    @staticmethod
    def get_hashtags(s):
        hashtag_patt = re.compile("#\w*\s")
        has = re.findall(hashtag_patt, s)
        return has

    @staticmethod
    def clean_mentions_hashtags(s):
        mention_patt = re.compile("@(\s*\w)")
        hashtag_patt = re.compile("#(\s*\w)")

        mentions = re.search(mention_patt, s)
        hashtags = re.search(hashtag_patt, s)


        if mentions is not None:
            s = re.sub("@\s*", " @", s)
        if hashtags is not None:
            s = re.sub("#\s*", "#", s)

        mentions_list = [m.strip() for m in get_mentions(s)]
        hashtags_list = [h.strip() for h in get_hashtags(s)]
        s = re.sub("@\w+\s*", " ", s)
        s = re.sub("#\w+\s*", " ", s)
        return s.strip(), mentions_list, hashtags_list
    
    @staticmethod
    def remove_link(s):
        s = re.sub("http:.*\s*", "", s)
        s = re.sub("https:.*\s*", "", s)
        s = re.sub("www\..*\s*", "", s)
        return s
    
    @staticmethod
    def remove_nlchars(s):
        s = re.sub("\n+", " ", s)
        s = re.sub("\s+"," ", s)
        s = re.sub("\t+", " ", s)
        return s.strip()
    
    @staticmethod
    def cleaner(s):
        s = Data.remove_link(s)
        s = Data.remove_nlchars(s)
        s = re.sub(r'(&amp;|&)', "and", s)
        s = re.sub(r'\d+\s*(a|p)\.*m\.*', "", s, flags=re.I)
        s = re.sub(r'(C|P|E)\W*T', "", s)
        s = re.sub(r'\(\)', "", s)
        s = re.sub(" \/ ", "", s)
        if s is not "" and s is not None:
            return s
        else:
            pass
    
    @staticmethod
    def strip_source(s):
        src = s.replace("Twitter ","").replace("for ", "").strip().lower()
        if src.find("android") > -1:
            return 'android', 1
        elif src.find("iphone") > -1:
            return "iphone", 2
        elif src.find("web") > -1:
            return "browser", 3
        else:
            return "other", 4
    
    @staticmethod
    def remove_twitlang(x):
        x = re.sub("@.*?\s", " ", x)
        x = re.sub("#.*?\s", " ", x)
        return x
    
    @staticmethod
    def is_quote(x):
        q = False
        if x[0] in ['"', "“", "-"]:
            q=True
        elif x.find('\"') > -1 or x.find('"') > -1:
            q=True
        elif re.search('\s+–\s*\w', x) is not None:
            q=True
        elif re.search('\s+-\s*\w', x) is not None:
            q=True
        elif re.search('\s+--\s*\w', x) is not None:
            q=True
        return q
        
    
    @staticmethod
    def _load_kaggle__():
        f = os.path.join(base_3rdparty, 'kaggle_original.csv')
        fields = ['id', 'link', 'content', 'date', 
                  'retweets', 'favorites', 'mentions', 'hashtags']
        kagglereadr = csv.DictReader(open(f, encoding='utf-8', mode='r'), 
                                     fieldnames=fields)
        next(kagglereadr)
        kaggle_rows = [r for r in kagglereadr]
        return kaggle_rows

    @staticmethod
    def _load_clarkgrieves__():
        
        rm_fields = ["TWEETID", "MENTION", "HASHTAG", 
                     "URL", 'JOB', 'SOURCE', 'RETWEET', 'FAV']
        cgdate_idx = " ".join([cg.pop('DATE'), cg.pop('TIME')])
        f = os.path.join(base_3rdparty, 'clarkgrieves_data.txt')
        cg_readr = csv.reader(open(f, encoding='utf-8', mode='r'))
        cg_header = next(cg_readr)
        cg_rows = [{cg_header[h]: row[h]
                    for h in range(len(cg_header))} 
                   for row in cg_readr if cg_header[h] not in rm_fields]
        _rows = []
        for cgx in cg_rows:
            cgx.update({"idx": " ".join([cgx.pop("DATE"), cgx.pop("TIME")]),
                        "word_count": cg.pop("WORDCOUNT", 0), 
                        'text_feats': {k.lower(): 
                                       (0 if cgx.pop(k) == 'A' else 1) 
                                       for k in cgx}})
            _rows.append(cgx)
            
        return _rows
    
    @staticmethod
    def _load_twitarchive__():
        f = os.path.join(base_3rdparty, 
                         'realdonaldtrump-wort_twitter_archive.json')
        tta = json.load(open(f, encoding='utf-8', mode='r'))
        
        def format_data(twitter_archive):
            jss = {}

            retweets = 0
            for j in twitter_archive:
                if Data.is_retweet(j):
                    retweets += 1
                    pass
                else:
                    if Data.is_quote(j['text']):
                        pass
                    else:
                        clean = Data.cleaner(j['text'])

                        if clean is not None:
                            created_date = datetime.strptime(
                                j.pop('created_at'), "%a %b %d %H:%M:%S %z %Y")
                            created_date = created_date.strftime(
                                "%Y-%m-%d %H:%M:%S")
                            plat_name, plat_id = Data.strip_source(j.pop('source'))
                            j.update({"platform_name": plat_name, 
                                      'platform_id': plat_id,
                                      'utc_date': created_date, 
                                      "contains_url": 
                                      (1 if j['text'].find("http")>-1 else 0),
                                      "text": clean})
                            _id = j.pop("id_str")
                            jss.update({_id: j})
            print(f'eliminated {retweets} retweets')    
            return jss
        return format_data(tta)
    
    @staticmethod
    def _merge_kaggle_twitterarchive__(kaggle_data, twitter_archive):
        new_rows = []
        quotes = 0
        for rowidx in tqdm(range(len(kaggle_data))):
            row = dict(kaggle_data[rowidx])
            try: 
                jdata = twitter_archive.pop(row['id'])
            except KeyError as err:
                pass
            else:

                row.update(**jdata)
                if is_quote(row.pop('content')):
                    pass

                else:
                    text, mentions, hashtags = Data.clean_mentions_hashtags(
                        row['text'])
                    fav_count = int(row.pop('favorites', 0))
                    retweets_count = int(row.pop("retweets",0))
                    dt =  datetime.strptime(row.pop('date'), 
                                            "%Y-%m-%d %H:%M:%S")
                    row.update({"mentions": [mxx for mxx in mentions 
                                             if len(mxx) > 0], 
                                "mentions_count": len(mentions), 
                                "hashtags": [hxx for hxx in hashtags 
                                             if len(hxx) > 0], 
                                "hashtags_count": len(hashtags), 
                                "favorites_count": fav_count, 
                                "retweets_count": retweets_count, 
                                'text': text, 
                                "local_date": dt.strftime("%Y-%m-%d"), 
                                'local_time': dt.strftime("%H:%M:%S")})
                    new_rows.append(row)

        print(f"{quotes} quotes eliminated")
        return new_rows

    @staticmethod
    def _merge_cg_other__(dat, cg_data):
        dd = []
        ix = 0
        for cgix in tqdm(range(len(cg_data))):

            cg = cg_data[cgix]
            for d in dat:
                if cg.pop('idx') == d['utc_date']:
                    if ix > 0:
                        pass
                    else:
                        ix += 1
                    dd.append(d)
        return dd
    
    def get_sample(self, load_local=True):
        f = os.path.join(base_data, "realdonaldtrump.sample.final")
        if os.path.isfile(f) and load_local:
            data = json.load(f)
        else:
            n = kwargs.get("n", 5000)
            full_data = Data.get()
            data = random.choices(full_data, k=n)
            json.dump(data, open(f, mode='w'))
        self._sample = data
    
    def get_data(self, load_local=True):

        f = os.path.join(base_data, "realdonaldtrump.full.final")
        
        if os.path.isfile(f) and load_local:
            data = json.load(open(f, mode='r'))
        else:
            
            kg = Data._load_kaggle__()
            cg = Data._load_clarkgrieves__()
            ta = Data._load_twitarchive__()
        
            kta = Data._merge_kaggle_twitterarchive__(kg, ta)
            data = Data._merge_cg_other__(kta, cg)
        
            json.dump(data, open(f, mode='w'))
        self._data = data
    
    
class Document(object):
    
    def __init__(self, doc):
        super().__init__()
        # see readme for field descriptions
        self._doc = doc

        fields = ["n_hashtags", "n_mentions", "avg_syllables",
                  "avg_word_length", "fk", 'n_sents', 
                  "n_ents", "n_uppers", "platform_id",'amplifier',
                  'analneg', 'attribadj', 'auxdo', 'bemv',
                  'bracket', 'caps', 'cconj', 'cntrstconj',
                  'colon', 'comma', 'defart', 'detquan',
                  'exclam', 'fstpp', 'fulstop', 'gerund',
                  'havemv', 'imperative', 'indefart', 
                  'infinitive', 'it', 'mdnec', 
                  'mdposs', 'mdpred', 'multiwvb',
                  'nomin', 'numdet', 'numnoun', 'objpro',
                  'otheradv', 'othrintj', 'othrnoun',
                  'othrverb', 'passive', 'past', 'perceptvb',
                  'perfect', 'posesprpn', 'possdet', 'predadj', 
                  'prep', 'procontract', 'progressive', 
                  'proquan', 'provdo', 'prpn', 'prvv', 'pubv', 
                  'ques', 'relclausesubgap', 'sinflect', 
                  'sndpp', 'stancevb', 'subjpro', 'superlative', 
                  'thrdpp', 'timeadv', 'whw', 'initialmention']
        
        self._mentions, self._hashtags = [], []
        self._length, self._n_words, self._n_sents = 0, 0, 0
        self._n_syllables, self._word_length_total = 0, 0
        self._n_faves, self._n_retweets = 0, 0
        self._pos_tags = []
        self._stops = [s.strip() for s in
                       open(os.path.join(base_data, 'stopwords.en.basic'),
                                         mode='r').readlines()]
        self._sentences, self._tokens, self._tokens_merged_ents = [], [], []     
        self._platform_map = {}
        self._platform_id = None
        self._platform_txt = None
        self._n_uppers = 0
        self._cg_features, self._feature_array = [], []
        self._local_date, self._local_time = None, None
        self._utc_date = None
        self._id = None
        
    @property
    def ID(self):
        return self._id
    
    @ID.setter
    def ID(self, x):
        self._id = x
    
    @property
    def favorites(self):
        return self._n_faves
        
    @favorites.setter
    def favorites(self, x):
        self._n_faves = x
        
    @property
    def retweets(self):
        return self._n_retweets
    
    @retweets.setter
    def retweets(self, x):
        self._n_retweets = x
        
    @property
    def platform(self):
        return self._platform_txt
    
    @platform.setter
    def platform(self, x):
        self._platform_txt = x
    
    @property
    def platform_id(self):
        return self._platform_id
    
    @platform_id.setter
    def platform_id(self, x):
        self._platform_id = x
        
    @property
    def cg_features(self):
        return self._cg_features
    
    @cg_features.setter
    def cg_features(self, x):
        self._cg_features = x
        
    
    @property
    def word_count(self):
        return self._n_words
    
    @property
    def doc_chars(self):
        return sum([len(t) for t in self._tokens])
   
    @word_count.setter
    def word_count(self, x):
        self._n_words = x
    
        
    @property
    def pos_tags(self):
        return self._pos_tags
    

    def is_stop(self, t):
        if t.lower() in self._stops:
            return True
        else:
            return False
    
    def set_tokens(self, x):
        self._tokens = x
    
    def set_tokens_merged(self, x):
        self._tokens_merged_ents = x
    
    
    def get_tokens_merged(self, keep_stops=False, lowercase=False):
        return [(t if lowercase is False else t.lower()) 
                for t in self._tokens_merged_ents 
                if self.is_stop(t) is keep_stops]
    
    def get_tokens(self, merge_ents=True, keep_stops=False, lowercase=False):
        if merge_ents:
            return self.get_tokens_merged_ents(keep_stops, lowercase)
        else:
            return [(t if lowercase is False else t.lower()) 
                    for t in self._tokens
                    if self.is_stop(t) is keep_stops]
    
    @property
    def length(self):
        return self._length
    
    @property
    def sentences(self):
        return self._sentences
    
    @sentences.setter
    def sentences(self, x):
        if type(x[0]) is str:
            self._sentences.append(x)
        elif type(x[0]) is list:
            self._sentences = x
            
    @property
    def n_sentences(self):
        return len(self._sentences)
    
    @property
    def entities(self):
        return self._doc.ents
    
    @property
    def n_entities(self):
        return len(self._doc.ents)
    
    @property
    def n_syllables(self):
        return self._n_syllables
    
    @property
    def mentions(self):
        return self._mentions
    
    @mentions.setter
    def mentions(self, x):
        self._mentions = x
    
    @property
    def hashtags(self):
        return self._hashtags
    
    @hashtags.setter
    def hashtags(self, x):
        self._hashtags = x
    
    @property
    def flesch_kincaid(self):
        fa = (self.word_count /self.n_sentences)
        fb = (self.n_syllables/self.word_count)
        calc = 206.835 - (1.015*fa) - (84.6*fb)
        return calc
    
    
    
    @staticmethod
    def source_text2id(platform):
        try:
            platform_id = self._platform_map[platform]
        except KeyError as errmsg:
            platform_id = len(self._platform_map) + 1
            self._platform_map.update({platform: len(self._platform_map)+1})
        return platform_id
        
    def extract_features(self, puncts):
        syllables_tot, word_len_tot = 0, 0 
        tkns_low, tags = [], []
        for t in self._doc:
            # drops mentions and hashtags
            if t.text not in self.mentions or t.text not in self.hashtags:
                self._n_syllables += (0 if t._.syllables_count is None 
                                      else t._.syllables_count)
                self._length += 1
                tx = t.text.strip()
                if tx in puncts:
                    # remove extraneous punctuations and stop words
                    pass
                else:
                    tx = utils.clean2(tx, puncts)
                    if len(tx) > 0:
                        self._word_length_total += len(tx)
                        self._tokens.append(tx)
                        self._pos_tags.append(t.tag_)
                        
    @property
    def n_uppers(self):
        return self._n_uppers

    def set_n_uppers(self, x=None):
        if x is None:
            n_uppers = 0
            for tkn in self._doc:
                txt = tkn.text
                if tkn.text.isupper():
                    n_uppers += 1
            self._n_uppers = n_uppers
        else:
            self._n_uppers = x
    
    @property
    def feature_array(self):
        return self._feature_array
    
    @feature_array.setter
    def feature_array(self, x):
        self._feature_array = x
    
    @property
    def local_date(self):
        return self._local_date
    
    @local_date.setter
    def local_date(self, x):
        self._local_date = x
      
    @property
    def local_time(self):
        return self._local_time
    
    @local_time.setter
    def local_time(self, x):
        self._local_time = x
    
    @property
    def utc_date(self):
        return self._utc_date
    
    @utc_date.setter
    def utc_date(self, x):
        self._utc_date = x
        

class Preprocessor(object):
    
    def __init__(self):

        self._stops = [s.strip() for s in
                       open(os.path.join(base_data, 'stopwords.en.basic'),
                                         mode='r').readlines()]
        self._puncts = list(punctuation)
        excl=['@', "#"]
        for e in excl:
            self._puncts.remove(e)
        self._puncts.extend(['”', '’', '“'])
        
        self._nlp = spacy.load("en_core_web_sm")
        sentencizer = spacy.pipeline.Sentencizer(punct_chars=[".", "?", "!", 
                                                              "。", ":", "..."])
        self._nlp.add_pipe(sentencizer, 
                                name="sentencizer", before='tagger')
        
        syllables = SpacySyllables(self._nlp)
        self._nlp.add_pipe(syllables, name='syllables')
        
        self._docs = []
        self._tkns_merged = []
        self._tkns_merged_all = []
        self._arrs = []
    
    @property
    def docs(self):
        return self._docs
    
    @property
    def tokens(self):
        return self._tkns_merged
    
    @property
    def tokens_flat(self):
        return self._tkns_merged_all
    
    @property
    def arrs(self):
        return self._arrs
    
    @staticmethod
    def update_nlp(nlp):

        merge_ents = nlp.create_pipe("merge_entities")
        nlp.add_pipe(merge_ents)
        return nlp

    def tokenize(self, data):
        nlp_x = Preprocessor.update_nlp(self._nlp)
        doc_sents_raw = []
        for didx in tqdm(range(len(data))):
            d = data[didx]['text']
            clean_str = utils.clean(d)

            if clean_str is not None and clean_str.strip() != "":
                try:
                    doc_0 = self._nlp(clean_str)
                    doc_1 = nlp_x(clean_str)

                except Exception as err:
                    print(err, clean_str)
                    pass
                else:
                    
                    tkns_merged, tkns_merged_all = [], []
                    for t in doc_1:
                        tt = utils.clean2(t.text, puncts=self._puncts)
                        if len(tt) == 0:
                            pass
                        else:
                            tkns_merged.append(tt)
                    
                    doc = Document(doc_0)
                    doc.set_tokens([utils.clean2(x.text, self._puncts) 
                                    for x in doc_0 
                                    if x.text not in self._puncts])
                    doc.sentences = [[utils.clean2(x.text, self._puncts) 
                                      for x in s 
                                      if x.text not in self._puncts] 
                                     for s in doc_0.sents]
                    
                    doc.set_tokens_merged(tkns_merged)
                    doc.mentions = data[didx].pop('mentions', [])
                    doc.hashtags = data[didx].pop('hashtags', [])
                    doc.platform = data[didx].pop('platform_name')
                    doc.platform_id = data[didx].pop('platform_id')
                    doc.set_n_uppers()
                    doc.extract_features(puncts=self._puncts)
                    
                    # in case you need it 
                    
                    wc = data[didx].pop("word_count", 0)
                    doc.word_count = (len(doc.get_tokens(
                        merge_ents=False, keep_stops=True)) 
                                      if wc == 0 else float(wc))
                    if doc.length == 0:
                        pass
                    else:
                        avg_syllables = doc.n_syllables/doc.word_count
                        avg_word_length = doc.doc_chars/doc.word_count
                        
                        doc.cg_features = [int(x) for x in 
                                           data[didx]['text_feats'].values()]

                        doc_arr = [len(doc.hashtags), 
                                   len(doc.mentions),
                                   avg_syllables, avg_word_length, 
                                   doc.flesch_kincaid, doc.n_sentences, 
                                   doc.n_entities, doc.n_uppers,
                                   doc.platform_id, *doc.cg_features]

                        if len(doc_arr) == 69:
                            doc.feature_array = doc_arr
                            doc.local_date = data[didx]['local_date']
                            doc.local_time = data[didx]['local_time']
                            doc.utc_date = data[didx]['utc_date']
                            doc.ID = didx
                            self._docs.append(doc)
                            
                        else:
                            print("Length Error: ")
                            print(f"{didx}) {data[didx]['text']}")