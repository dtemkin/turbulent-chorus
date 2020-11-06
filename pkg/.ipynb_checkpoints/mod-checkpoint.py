from datetime import datetime
from sklearn.cluster import OPTICS, DBSCAN
import pickle
import os

dirname=os.path.dirname
def cluster(data, cluster_meth, load_local=False, *args, **kwargs):
    models_dir = os.path.join(dirname(dirname(__file__)), 'tmp', 'models', 'cluster')
    mods = os.listdir(models_dir)
    sel = -1
    if load_local:
        sel = -1
        while sel < 0:
            
            print("\n############  Available Models  ################\n")
            for modidx in range(len(mods)):
                modf = mods[modidx]
                print(f"{modidx})   {modf}")
            usrin = input("Enter Selection: ")
            if usrin >= len(mods):
                print("Invalid Selection")
                continue
            else:
                sel = usrin
        mod = mods[sel]
        selmod = os.path.join(models_dir, f'{mod}') 
        m = pickle.load(open(selmod, mode='rb'))
        modtext = repr(m)
        print(f"{modtext} Loaded.")
    else:
        if cluster_meth == 'optics':
            m = OPTICS(*args, **kwargs)
        elif cluster_meth == 'dbscan':
            m = DBSCAN(*args, **kwargs)
        elif hasattr(cluster_meth, "__call__"):
            m = cluster_meth.__call__(*args, **kwargs)
        else:
            raise TypeError("cluster_method must be 'optics', 'dbscan' or callable")
        modfile =os.path.join(dirname(dirname(__file__)), 'tmp', 'models', 'cluster',
                              f'{repr(m)}.fitted')
        modtext = repr(m)
    
        if os.path.isfile(modfile):
            m = pickle.load(open(modfile, mode='rb'))
            print(f"{modtext} Loaded.")
        else:
            print(f"Fitting {modtext}...")
            m.fit_predict(X=data)
            print(f"Fitted.\n")
            print(f"Pickling...")
            pickle.dump(m, open(modfile, mode='wb'))
            print(f'Saved.')

    return m


def summary(arrs, colnames, model):
    df = DataFrame(arrs, index=0, columns=colnames)

    df.insert(0, "cluster", model.labels_)
    groups = df.groupby('cluster')

    print(groups[0].count())

    print(groups.describe())
    