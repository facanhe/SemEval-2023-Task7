import translators as ts
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

LANG = 'fr'
API = 'baidu'

lang_dict = {'Chinese': 'cht', 'French': 'fra', 'Italian': 'it', 'Portuguese': 'pt', 'Spanish': 'spa', 'English': 'en'}


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
    # return [translator_constructor(API)(x[0], lang_dict[x[2]], 'en'), x[1], x[2]]
    try:
        return [translator_constructor(API)(x[0], lang_dict[x[2]], 'en'), x[1], x[2]]
    except:
        return [x[0], x[1], x[2]]


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
    df = pd.read_csv('./trainingdata/traindata.csv', header=0, sep=',', quoting=2)
    # tqdm.pandas('Translation progress')
    df[['Statement','Premise']] = imap_unordered_bar(translate, df[['Statement', 'Premise']].values)
    df.to_csv(f'./trainingdata/translated_train.csv', index=False)


if __name__ == '__main__':
    main()