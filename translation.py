import numpy as np
import translators.server as ts
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

LANG = 'fr'
API = 'google'

lang_dict = {'Chinese': 'cht', 'French': 'fr', 'Italian': 'it', 'Portuguese': 'pt', 'Spanish': 'spa', 'English': 'en'}


def translator_constructor(api):
    if api == 'google':
        return ts.google
    elif api == 'bing':
        return ts.bing
    elif api == 'baidu':
        return ts.baidu
    elif api == 'sogou':
        return ts.sogou
    elif api == 'youdao':
        return ts.youdao
    elif api == 'tencent':
        return ts.tencent
    elif api == 'alibaba':
        return ts.alibaba
    else:
        raise NotImplementedError(f'{api} translator is not realised!')


def translate(x):
    #print(x[2])
    print([x[0], x[1],translator_constructor(API)(x[2], 'fr', 'en',if_ignore_limit_of_length=True),translator_constructor(API)(x[3], 'fr', 'en',if_ignore_limit_of_length=True)])
    #return[x[0], x[1],translator_constructor(API)(x[2], 'en', 'fr',if_ignore_limit_of_length=True),translator_constructor(API)(x[3], 'en', 'fr',if_ignore_limit_of_length=True)]
    # return [translator_constructor(API)(x[0], lang_dict[x[2]], 'en'), x[1], x[2]]
    try:
        print([x[0], x[1],translator_constructor(API)(x[2], 'fr', 'en',if_ignore_limit_of_length=True),translator_constructor(API)(x[3], 'fr', 'en',if_ignore_limit_of_length=True)])
        return[x[0], x[1],translator_constructor(API)(x[2], 'fr', 'en',if_ignore_limit_of_length=True),translator_constructor(API)(x[3], 'fr', 'en',if_ignore_limit_of_length=True)]
        print(translator_constructor(API)(x[2],x[3], 'fra'))
    except:
        return [x[0], x[1], x[2],x[3]]


def imap_unordered_bar(func, args, n_processes: int = 48):
    p = Pool(n_processes, maxtasksperchild=100)
    res_list = []
    with tqdm(total=len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list


def main():
    df = pd.read_csv('./trainingdata/translated_train.csv',dtype={'Label': np.int})
    df1=pd.read_csv('./trainingdata/traindata.csv',dtype={'Label': np.int})
    # tqdm.pandas('Translation progress')
    df[['uuid','Label','Statement','Premise']] = imap_unordered_bar(translate, df[['uuid','Label','Statement', 'Premise']].values)
    file=[df1,df]
    train=pd.concat(file)
    train.to_csv(f'./trainingdata/translated_traindata.csv', index=False)
    #df.to_csv(f'./trainingdata/translated_train.csv', index=False)


if __name__ == '__main__':
    main()