import spacy
from spacy_syllables import SpacySyllables
from collections import Counter


def build_pipe(corpus_size='sm'):
    nlp = spacy.load(f"en_core_web_{corpus_size}", )

    sentencizer = spacy.pipeline.Sentencizer(punct_chars=[".", "?",
                                                          "!", "。",
                                                          ":", "..."])
    nlp.add_pipe(sentencizer, name="sentencizer", before='tokenizer')

    merge_nps = nlp.create_pipe("merge_noun_chunks")
    nlp.add_pipe(merge_nps)

    merge_ents = nlp.create_pipe("merge_entities")
    nlp.add_pipe(merge_ents)

    syllables = SpacySyllables(nlp)
    nlp.add_pipe(syllables, name='syllables')
    return nlp


class Preprocessor(object):

    def __init__(self, docs, pipe):
        self._pipeline = pipe
        self._raw_docs = docs

        self._docs = []
        self._tokens_docs_no_stops = []

        self._tokens_docs = []
        self._tokens_flat = []
        self._sents_docs = []
        self._sents_flat = []
        self._tokens_counts = None

        tokens_dir = "../data/tweets/tokens/"
        self._tokens_files = {"flat": os.path.join(tokens_dir,
                                                   'tokens_flat.json'),
                              "docs": os.path.join(tokens_dir,
                                                   'tokens_docs.json'),
                              "counts": os.path.join(tokens_dir,
                                                     'tokens_counts.json')}
        
        self._sents_files = {"flat": os.path.join(tokens_dir, 'sents_flat.json'),
                             "docs": os.path.join(tokens_dir, 'sents_docs.json')}
        
        self._dump_files = [tflat_file, tdoc_file, counts_file, sflat_file, sdoc_file]

        self._stops = [s.strip() for s
                       in open('../data/shared/stopwords.en.basic',
                               mode='r').readlines()]
        
    @property
    def files(self):
        tokens_dir = "../data/tweets/tokens/"
        tflat_file = os.path.join(tokens_dir, 'tokens_flat.json')
        tdoc_file = os.path.join(tokens_dir, 'tokens_docs.json')
        counts_file = os.path.join(tokens_dir, 'tokens_counts.json')
        
        sflat_file = os.path.join(tokens_dir, 'sents_flat.json')
        sdoc_file = os.path.join(tokens_dir, 'sents_docs.json')
        
        files = [tflat_file, tdoc_file, counts_file, sflat_file, sdoc_file]

    def remove_puncts(self, excl=['@', "#"]):
        def __func_(itm, puncts):
            p = puncts.pop(0)
            itm = itm.replace(p, "")
            if len(puncts) == 0:
                return itm
            else:
                return remove_puncts(itm, puncts=puncts)
        puncts = list(punctuation)
        for e in excl:
            puncts.remove(e)

        self._docs = [__func_(d, puncts) for d in self._docs]

    def clean_pass1(self, s):
        patt0 = re.compile("http.*\S")
        patt1 = re.compile("www\..*?\.com")
        s = re.sub(patt0, "", s)
        s = re.sub(patt1, "", s)
        s = re.sub('…', "", s)
        s = re.sub("\.\.\.", "", s)
        s = re.sub("-", ' ', s)
        s = re.sub("–", ' ', s)
        s = s.replace("@ ", "@").replace("# ", "#")
        s = s.replace("  ", " ").replace("   ", " ").replace("’", "")
        s = re.sub("P.M.", "pm", s)
        s = re.sub("A.M.", 'am', s)
        s = re.sub("RT ", "", s)
        s = re.sub("&amp;", "and", s)
        s = re.sub("&", "and", s)
        s = re.sub(" CT", '', s)
        s = re.sub(" PT", "", s)
        s = re.sub(" ET", "", s)
        s = re.sub(" ET/PT", "", s)
        s = re.sub(" ETPT", "", s)
        s = re.sub(" EST", "", s)
        s = re.sub(" EDT", "", s)

        return s

    def clean_pass2(self, s):
        return s.strip().lower()

    def clean_strings(self, p=0):
        if p == 0:
            fn = lambda x: self.clean_pass2(self.clean_pass1(x))
        elif p == 1:
            fn = self.clean_pass1
        else:
            fn = self.clean_pass2
        self._docs = [fn(d) for d in self._docs]

    def remove_stops(self, incl_=[]):
        stops_list = self._stops.copy()
        if type(incl_) == str:
            stops_list.append(incl_)
        elif type(incl_) == list:
            stops_list.extend(incl_)
        else:
            raise TypeError("Invalid Type.")

        self._docs_no_stops = [[(w.lower(), t) for w, t in d
                                if w.lower() not in stops_list]
                               for d in self._docs_tokend]

    def remove_spec(self, tokens=None, as_='flat', char_='@'):

        if as_ == 'flat':
            if tokens is None:
                tokens = self._tokens_flat
            else:
                pass
            tokens_no_spec = [(w.lower(), t) for w, t
                              in tokens if w.lower()[0] != char_]
        elif as_ == 'docs':
            if tokens is None:
                tokens = self._tokens_docs
            else:
                pass
            tokens_no_spec = [[(w.lower(), t) for w, t
                               in doc if w.lower()[0] != char_]
                              for doc in tokens]
        return tokens_no_spec

    def tokenize(self, UPDATE=False):
        if UPDATE or not all([os.path.isfile(f) for f in ]):
            for doc in self._docs:
                d = self._pipeline(doc)
                tokens = [(t.text, t.tag_) for t in d]
                sents = d.sents
                self._sents_flat.extend(sents)
                self._sents_docs.append(sents)
                self._tokens_flat.extend(tokens)
                self._tokens_docs.append(tokens)

    def set_counts(self, tokens=None):
        if tokens is not None:
            pass
        else:
            tokens = self._tokens_flat
        self._tokens_counts.update(tokens)

    def get_counts(self):
        return self._tokens_counts
    
    def set_sentences(self, sents, as_='flat'):
        if as_ == 'flat':
            self._sents_flat.extend(sents)
        
        elif as_ == 'doc':
            self._sents_docs.append(sents)
    
    def get_sentences(self, as_='flat'):
        return (self._sents_flat if as_ == 'flat' else self._sents_docs)
    
    def get_tokens(self, as_='flat'):
        return (self._tokens_flat if as_ == 'flat' else self._tokens_docs)
    
    def set_tokens(self, tokens, as_='flat'):
        if as_ == 'flat':
            self._tokens_flat.extend(tokens)
        
        elif as_ == 'doc':
            self._tokens_docs.append(tokens)
        
    
    def load(self, grp, itm):
        tokens_dir = "../data/tweets/tokens/"
        
        f = os.path.join(tokens_dir, f'{grp}_{itm}.json')
        return json.load(f)
    
    def dump(self, grp, itm):
        tokens_dir = "../data/tweets/tokens/"
        tflat_file = os.path.join(tokens_dir, 'tokens_flat.json')
        tdoc_file = os.path.join(tokens_dir, 'tokens_docs.json')
        counts_file = os.path.join(tokens_dir, 'tokens_counts.json')
        
        sflat_file = os.path.join(tokens_dir, 'sents_flat.json')
        sdoc_file = os.path.join(tokens_dir, 'sents_docs.json')
        
        files = [tflat_file, tdoc_file, counts_file, sflat_file, sdoc_file]
    
        