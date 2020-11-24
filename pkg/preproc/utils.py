import re

def clean(string):
    string = re.sub(r'(&amp;|&)', "and", string)
    string = re.sub(r'\d(\W|)(am|a.m.|A.M.|pm|p.m.|P.M.)', "", string)
    string = re.sub(r'(CST|PST|EST|PT|CT|ET|CDT|EDT|PDT)', "", string)
    string = re.sub(" \/ ", "", string)
    if string is not None:
        if len(string) > 0:
            return string

def clean2(string, puncts):
    string = re.sub("\.(?=.*\.)", "", string)
    string = string.replace("’ve", " have").replace("’s", "")
    string = string.replace("’ll", ' will').replace("n’t", " not")
    string = string.replace("‘18", '2018').replace("w\/", "with")
    string = string.replace("'s", "").replace("'ve", " have")
    string = string.replace("'ll", ' will')
    string = string.replace("…", "")
    string = string.replace("u.s.", 'usa')
    string = string.replace("U.S.", 'usa')
    string = string.replace("US", 'usa')
    string = string.replace("us.", 'us')
    string = string.replace("🇺", "")
    string = string.replace("n't", " not").replace("--","").replace("—","")
    string = string.strip()
    if len(string) > 0 and string not in puncts:
        return string
    else:
        return ""