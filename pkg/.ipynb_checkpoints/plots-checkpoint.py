import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

def make_distplots(df, labs):
    
    fig, axs = plt.subplots(nrows=len(labs), figsize=[20.0, 130.0])
    for lix in range(len(labs)):
        sns.histplot(data=df[labs[lix]], stat='count', ax=axs[lix])


