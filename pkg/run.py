import argparse

base = "../data/tweets/tmp/data"


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