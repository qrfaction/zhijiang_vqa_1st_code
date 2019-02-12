import pandas as pd
from collections import Counter
from glob import glob
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import cv2
from config import *
from tqdm import tqdm
import json
import pickle
import numpy as np

def get_embedding_matrix(w2i,wordvec):
    print('get embedding matrix')

    num_words = len(w2i) + 1
    embedding_matrix = np.random.uniform(-0.2,0.2,(num_words,300))
    embedding_matrix[0]=0

    with open(word_vec_dir+wordvec+'w2v.pkl','rb') as f:
        w2v = pickle.load(f)

    noword = {}
    num_noword = 0
    for word, i in w2i.items():
        vec = w2v.get(word)
        if vec is None:
            if word not in noword:
                noword[word] = 0
            noword[word]+=1
            num_noword+=1
        else:
            embedding_matrix[i] = vec

    noword = sorted(noword.items(),key=lambda item:item[1],reverse=True)


    with open(data_path + wordvec+'noword.json','w') as f:
        f.write(json.dumps(noword,indent=4, separators=(',', ': ')))
    print('miss:', num_noword)
    return embedding_matrix


def read_worker(wordmat):
    result={}
    for line in wordmat:
        wvec = line.strip('\n').strip(' ').split(' ')
        result[wvec[0]] = np.asarray(wvec[1:], dtype='float32')
    return result

def read_wordvec(file,name):

    import multiprocessing as mlp

    with open(word_vec_dir+file) as f:
        wordmat=[line for line in f.readlines()]
        if wordmat[-1]=='':
            wordmat = wordmat[:-1]
        if wordmat[0] == '':
            wordmat = wordmat[1:]

    results = []
    pool = mlp.Pool(mlp.cpu_count())

    aver_t = int(len(wordmat)/mlp.cpu_count()) + 1
    for i in range(mlp.cpu_count()):
        result = pool.apply_async(read_worker, args=(wordmat[i*aver_t:(i+1)*aver_t],))
        results.append(result)
    pool.close()
    pool.join()

    word_dict={}
    for result in results:
        word_dict.update(result.get())
    print('num of words:',len(word_dict))
    with open(word_vec_dir+name+'w2v.pkl','wb') as f:
        pickle.dump(word_dict, f, 4)
    return word_dict


def load_video_attr(topk=100):
    from collections import Counter
    with open(data_path+'video_attr.json','r') as f:
        video_attr = json.loads(f.read())

    v_attr = {}
    num_attr = []
    attr_text = []
    for k,frames_attr in video_attr.items():
        attr_count = [attr for attrs in frames_attr for attr in attrs]
        attr_count = sorted(Counter(attr_count).items(), key=lambda x: x[1], reverse=True)
        attrs = [a for a, freq in attr_count[:topk]]
        num_attr.append(len(attr_count))
        v_attr[k] = attrs
        attr_text+=attrs

    # print(sorted(Counter(num_attr).items(), key=lambda x: x[1], reverse=True))
    attr_text = list(set(attr_text))
    attr2i = {a:i for i,a in enumerate(attr_text)}
    return v_attr,attr_text,attr2i

def question2seq(texts_list,wordvec,seq_len):

    if isinstance(texts_list[0],str):
        texts = texts_list
    elif isinstance(texts_list[0],list):
        texts = [text for texts in texts_list for text in texts]
    else:
        raise RuntimeError("texts error")

    tokenizer = Tokenizer(num_words=1e6)
    tokenizer.fit_on_texts(texts)
    word2i = tokenizer.word_index
    embed_matrix = get_embedding_matrix(word2i,wordvec)


    if isinstance(texts_list[0],str):
        q_seq = tokenizer.texts_to_sequences(texts)
        q_seq = pad_sequences(q_seq, maxlen=seq_len, truncating='post')
        return np.array(q_seq),embed_matrix
    elif isinstance(texts_list[0],list):
        seq_list = []
        for i,t in enumerate(texts_list):
            q_seq = tokenizer.texts_to_sequences(t)
            q_seq = pad_sequences(q_seq, maxlen=seq_len[i], truncating='post')
            seq_list.append(np.array(q_seq))
        return seq_list,embed_matrix
    else:
        raise RuntimeError("texts error")

def get_prior():

    with open(data_path + 'ans2i.json', 'r') as f:
        ans2i = json.loads(f.read())
    with open(data_path + 'i2ans.json', 'r') as f:
        i2ans = json.loads(f.read())

    tr = pd.read_csv(data_path + 'Train_frameqa_question.csv', sep='\t')
    te = pd.read_csv(data_path + 'Test_frameqa_question.csv', sep='\t')
    data = tr.append(te).reset_index(drop=True)

    num_type = data['type'].max()+1
    c2v = np.zeros((num_type,len(i2ans)))

    for _, row in data.iterrows():
        c = row['type']
        a = row['answer']
        if a in ans2i:
            i = ans2i[a]
            c2v[c][i] = 1
    print(c2v.sum(axis=1))
    with open(data_path+'c2v.pkl', 'wb') as f:
        pickle.dump(c2v, f, 4)

def update_json(tagret_f,files):

    with open(data_path+tagret_f, 'r') as f:
        target = json.loads(f.read())
    print(len(target))
    for file in files:
        with open(data_path+file, 'r') as f:
            data = json.loads(f.read())
        target.update(data)
    print(len(target))
    with open(data_path+tagret_f, 'w') as f:
        f.write(json.dumps(target,indent=4, separators=(',', ': ')))


if __name__=='__main__':
    update_json('video_attr.json',[
         'video_attr_h.json',
    ])