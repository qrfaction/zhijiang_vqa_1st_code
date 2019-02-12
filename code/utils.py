from config import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from collections import Counter
import pandas as pd
import pickle
import json
import re
from tqdm import tqdm


def data2table(file,return_data=False):
    if file == 'train':
        file = data_path + 'train.txt'
    elif file == 'test':
        file = data_path + 'test.txt'
    else:
        raise RuntimeError("data error")
    with open(file,'r') as f:
        data = f.read().split('\n')
        data = [sample for sample in data if sample!='']
        data = [sample.split(',') for sample in data]

    dataset = []
    for record in data:
        try:
            assert (len(record)-1) % 4 == 0
        except AssertionError as e:
            print(record)
        video_id = record[0]
        for i in range(1,len(record),4):
            dataset.append([video_id,record[i],record[i+1],record[i+2],record[i+3]])
    dataset = pd.DataFrame(dataset,columns=['video_id','question','ans0','ans1','ans2'])

    if return_data:
        return dataset
    dataset.to_csv(file[:-4] + '.csv',index=False)

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

    if os.path.exists('./info/')==False:
        os.makedirs('./info/')
    with open('./info/'+wordvec+'noword.json','w') as f:
        f.write(json.dumps(noword,indent=4, separators=(',', ': ')))
    print('miss:', num_noword)
    return embedding_matrix

def question2seq(texts_list,wordvec,seq_len):
    if isinstance(texts_list[0],str):
        texts = texts_list
    elif isinstance(texts_list[0],list):
        texts = [text for texts in texts_list for text in texts]
    else:
        raise RuntimeError("texts error")

    if os.path.exists(data_path+'tokenizer1.pkl'):
        with open(data_path + 'tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        tokenizer = Tokenizer(num_words=1e8,oov_token=1)
        tokenizer.fit_on_texts(texts)
        with open(data_path+'tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer,f)
    word2i = tokenizer.word_index
    embed_matrix = get_embedding_matrix(word2i, wordvec)

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

def read_worker(wordmat):
    result={}
    for line in wordmat:
        wvec = line.strip('\n').strip(' ').split(' ')
        result[wvec[0]] = np.asarray(wvec[1:], dtype='float32')
    return result

def read_wordvec(file,name):

    import multiprocessing as mp

    with open(word_vec_dir+file) as f:
        wordmat=[line for line in f.readlines()]
        if wordmat[-1]=='':
            wordmat = wordmat[:-1]
        if wordmat[0] == '':
            wordmat = wordmat[1:]

    results = []
    pool = mp.Pool(mp.cpu_count())

    aver_t = int(len(wordmat)/mp.cpu_count()) + 1
    for i in range(mp.cpu_count()):
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

def split_data(data,idx):
    return {k:v[idx] for k,v in data.items()}

def select_label(y,multi_mode=False):
    with open(data_path + 'i2ans.json', 'r') as f:
        i2ans = json.loads(f.read())
    with open('./info/get_ansfreq.json', 'r') as f:
        ans_freq = json.loads(f.read())
    answers = []
    if multi_mode:
        for multi_ans in y:
            min_freq = 1000000
            best_i = -1
            for label in multi_ans:
                ans_idx = np.where(label != 0)[0]
                for i in ans_idx:
                    a = i2ans[i]
                    freq = ans_freq[a]
                    if min_freq > freq:
                        min_freq = freq
                        best_i = i
            assert best_i>-1
            answers.append(best_i)
    else:
        for label in y:
            ans_idx = np.where(label != 0)[0]
            best_i = ans_idx[0]
            min_freq = ans_freq[i2ans[best_i]]

            for i in ans_idx:
                a = i2ans[i]
                freq = ans_freq[a]
                if min_freq > freq:
                    min_freq = freq
                    best_i = i
            answers.append(best_i)
    print(len(answers))
    return answers

def get_result(y_pred, output,multi_mode=True,is_tr=False,te=None):

    with open(data_path + 'i2ans.json', 'r') as f:
        i2ans = json.loads(f.read())

    if multi_mode:
        result = []
        for multi_y in y_pred:
            result.append([i2ans[y.argmax()] for y in multi_y])
        if te is None:
            if is_tr:
                te = pd.read_csv(data_path + 'train.csv')
            else:
                te = pd.read_csv(data_path + 'test.csv')

        submit = []
        for (i,row),y in zip(te.iterrows(),result):
            record = [row['video_id']]
            for j in range(5):
                record += [row['q'+str(j)],y[j]]
            submit.append(','.join(record))
        submit = '\n'.join(submit)
        with open(output,'w') as f:
            f.write(submit)
    else:
        y_pred = [y.argmax() for y in y_pred]

        if te is None:
            if is_tr:
                te = pd.read_csv(data_path + 'train.csv', usecols=['video_id', 'question'])
            else:
                te = pd.read_csv(data_path + 'test.csv', usecols=['video_id', 'question'])
        te = te[['video_id', 'question']]

        y_pred = [i2ans[i] for i in y_pred]
        te['ans'] = y_pred

        def merge_ans(x):
            result = []
            for i, row in x.iterrows():
                result += [row['question'], row['ans']]
            return pd.Series(result)

        te = te.groupby(by=['video_id']).apply(merge_ans)
        te.to_csv(output, header=False)

def load_video_attr(topk=100):
    from collections import Counter
    with open(data_path+'video_attr.json','r') as f:
        video_attr = json.loads(f.read())

    v_attr = {}
    # num_attr = []
    attr_text = []
    for k,frames_attr in video_attr.items():
        attr_count = [attr for attrs in frames_attr for attr in attrs]
        attr_count = sorted(Counter(attr_count).items(), key=lambda x: x[1], reverse=True)
        attrs = [a for a, freq in attr_count[:topk]]
        # num_attr.append(len(attr_count))
        v_attr[k] = attrs
        attr_text+=attrs

    # print(sorted(Counter(num_attr).items(), key=lambda x: x[1], reverse=True))
    # print(sorted(Counter(num_attr).items(), key=lambda x: x[0], reverse=True))
    attr_text = list(set(attr_text))
    attr2i = {a:i for i,a in enumerate(attr_text)}
    return v_attr,attr_text,attr2i

def get_prior():
    def clean_questions(q):
        q = q.replace(' ot ', ' or ')
        q = q.replace(' pr ', ' or ')
        q = q.replace(" the ", " ") + " "
        q = q.replace(" a ", " ")
        q = q.replace(" short hear ", " short hair ")
        q = q.replace('where are color of', 'what are color of')
        return q

    with open(data_path + 'ans2i.json', 'r') as f:
        ans2i = json.loads(f.read())
    with open(data_path + 'i2ans.json', 'r') as f:
        i2ans = json.loads(f.read())
    tr = data2table('train',return_data=True)
    te = data2table('test',return_data=True)

    question = tr['question'].append(te['question']).tolist()
    question = list(set(question))

    def get_keyword(x):
        x_copy = clean_questions(x)
        if ' or not' in x:
            kw = 'yes&no'
        elif ' or ' in x_copy:
            x_seq = x_copy.split(' ')
            or_i = x_seq.index('or')
            kw = []
            for a in i2ans:
                reg_1 = ' ' + a + ' or ' in x_copy
                reg_2 = ' or ' + a + ' ' in x_copy
                reg_3 = ' or in ' + a + ' ' in x_copy
                if ' ' in a:
                    space_i = a.index(' ')
                    reg_4 = a[:space_i] == x_seq[or_i-1] and a[space_i+1:] == x_seq[or_i+2]
                    reg_5 = a[:space_i] == x_seq[or_i-2] and a[space_i+1:] == x_seq[or_i+1]
                    reg_6 = a[:space_i] == x_seq[or_i+1] and a[space_i+1:] == x_seq[or_i-1]
                else:
                    reg_4 = False
                    reg_5 = False
                    reg_6 = False
                if reg_1 | reg_2 | reg_3 | reg_4 | reg_5 | reg_6:
                    kw.append(a)

            kw = sorted(kw)
            if len(kw) <= 1:  # or 一般至少能找到俩
                kw = '&'
            else:
                kw = '&'.join(kw)
        else:
            kw = '&'
        return kw

    q2key = {q:get_keyword(q) for q in question}
    c2a = {key:set(key.split('&')) for key in q2key.values() if key!='&'}
    for _,row in tr.iterrows():
        ans_proposals = get_keyword(row['question'])
        if ans_proposals in c2a:
            for a in row[['ans'+str(i) for i in range(3)]]:
                c2a[ans_proposals].add(a)

    q2a = {q:'&'.join(c2a.get(key,[key])) for q,key in q2key.items()}

    questions = tr['question'].tolist() + te['question'].tolist()
    questions = set(questions)
    c2a.update({
        'color': set(),
        'doing': set(),
        'where': set(),
        'y/n': set(),
        'many': set(),
        'other': set(),
    })

    q2c = {}
    c2q = {c:set() for c in c2a.keys()}
    for q_ in questions:
        kw = q2key[q_]
        q = '$' + clean_questions(q_)
        if kw != '&':
            q2c[q_] = kw
            c2q[kw].add(q_)
        elif 'what' in q and 'color' in q:
            q2c[q_] = 'color'
            c2q['color'].add(q_)
        elif 'what' in q and 'doing' in q or re.match('what do.*? do ', q + ' '):
            q2c[q_] = 'doing'
            c2q['doing'].add(q_)
        elif 'where' in q:
            q2c[q_] = 'where'
            c2q['where'].add(q_)
        elif 'whether' in q or '$does ' in q or \
            '$do ' in q or '$is ' in q or '$are ' in q:
            q2c[q_] = 'y/n'
            c2q['y/n'].add(q_)
        elif 'how many' in q:
            q2c[q_] = 'many'
            c2q['many'].add(q_)
        else:
            q2c[q_] = 'other'
            c2q['other'].add(q_)

    c2v = {c: np.zeros(len(i2ans)) for c in c2q.keys()}
    for _, row in tr.iterrows():
        c = q2c[row['question']]
        answers = [row['ans0'], row['ans1'], row['ans2']]
        for a in answers:
            if a in ans2i:
                i = ans2i[a]
                c2v[c][i] = 1
                c2a[c].add(a)

    for k in c2q.keys():
        c2q[k] = list(c2q[k])
        c2a[k] = list(c2a[k])


    with open('./info/c2v.pkl', 'wb') as f:
        pickle.dump(c2v, f, 4)
    with open('./info/q2c.json', 'w') as f:
        f.write(json.dumps(q2c, indent=4, separators=(',', ': ')))
    with open('./info/c2q.json', 'w') as f:
        f.write(json.dumps(c2q, indent=4, separators=(',', ': ')))
    with open('./info/c2a.json', 'w') as f:
        f.write(json.dumps(c2a, indent=4, separators=(',', ': ')))
    with open(data_path+'q2a.json','w') as f:
        f.write(json.dumps(q2a,indent=4, separators=(',', ': ')))
    return q2c, c2q


def get_useless_ans(freq):
    tr = data2table('train', return_data=True)

    ans2q = {}

    with open('./info/get_ansfreq.json', 'r') as f:
        ansfreq = json.loads(f.read())

    for i, row in tr.iterrows():
        for j in range(3):
            if ansfreq[row['ans' + str(j)]] < freq:
                continue
            if row['ans' + str(j)] not in ans2q:
                ans2q[row['ans' + str(j)]] = set()
            ans2q[row['ans' + str(j)]].add(i)

    ans2numq = {a: len(questions) for a, questions in ans2q.items()}
    ans2numq = sorted(ans2numq.items(), key=lambda x: x[1], reverse=True)

    ans_subset = {}
    for i in range(len(ans2numq)):
        ans1, freq1 = ans2numq[i]
        if ans1 in ans_subset:
            continue
        ans_subset[ans1] = []
        for j in range(i + 1, len(ans2numq)):
            ans2, freq2 = ans2numq[j]
            if ans2 in ans_subset:
                continue
            if ans2q[ans1].issuperset(ans2q[ans2]):
                ans_subset[ans1].append(ans2)

    ans_subset = [a for v in ans_subset.values() if len(v) >= 2 for a in v]
    return ans_subset

def concat_dataset(files,outputfile):

    data = ''
    for file in files:
        with open(data_path+file,'r') as f:
            data = data + f.read().strip().strip('\n') + '\n'

    data = data.strip('\n')
    with open(data_path+outputfile,'w') as f:
        f.write(data)

def prepocess(min_freq=10):

    names = ['video_id']
    for i in range(5):
        names += ['q'+str(i)]+['q'+str(i)+'_ans'+str(j) for j in range(3)]

    tr = pd.read_csv(data_path+'train.txt',names=names)
    te = pd.read_csv(data_path+'test.txt',names=names)


    # --------------------------------- cal freq ans ------------------------------------

    ans_counter = {}
    for i in range(5):
        cols = ['q'+str(i)+'_ans'+str(j) for j in range(3)]
        for ans in tr[cols].values:
            for i in list(set(ans)):
                ans_counter[i] = ans_counter.get(i,0) + 1

    if os.path.exists('./info/')==False:
        os.makedirs('./info/')
    with open('./info/ans_freq.json', 'w') as f:
        f.write(json.dumps(sorted(ans_counter.items(), key=lambda x:x[1], reverse=True),indent=4, separators=(',', ': ')))
    with open('./info/get_ansfreq.json', 'w') as f:
        f.write(json.dumps(ans_counter,indent=4, separators=(',', ': ')))


    # ------------------------------- filter sample --------------------------------------
    useless_ans = get_useless_ans(min_freq)
    print('useless a',len(useless_ans))
    num_questions = 0
    data = []
    for _, sample in tr.iterrows():
        questions = []
        for i in range(5):
            cols = ['q' + str(i) + '_ans' + str(j) for j in range(3)]
            ans1 = sample[cols[0]]
            ans2 = sample[cols[1]]
            ans3 = sample[cols[2]]
            q = sample['q' + str(i)]
            q_ans_pair = []
            if ans_counter[ans1] >= min_freq and ans1 not in useless_ans:
                q_ans_pair.append(ans1)
            if ans_counter[ans2] >= min_freq and ans2 not in useless_ans:
                q_ans_pair.append(ans2)
            if ans_counter[ans3] >= min_freq and ans3 not in useless_ans:
                q_ans_pair.append(ans3)

            q_ans_pair = list(set(q_ans_pair))
            num_ans = len(q_ans_pair)
            if num_ans>0:
                q_ans_pair = [q]+q_ans_pair+(3-num_ans)*['_UNKA_']
                assert len(q_ans_pair)==4
                questions += q_ans_pair

        num_q = len(questions)//4
        num_questions+=num_q
        if num_q >= 3:
            s = [sample['video_id']] + questions + (5-num_q)*['_UNKQ_','_UNKA_','_UNKA_','_UNKA_']
            assert len(s) == 21
            data.append(s)
    print(num_questions/5)
    tr = pd.DataFrame(data, columns=tr.columns)
    questions = set(tr['q0'])|set(tr['q1'])|set(tr['q2'])|set(tr['q3'])|set(tr['q4'])
    questions = questions|set(te['q0'])|set(te['q1'])|set(te['q2'])|set(te['q3'])|set(te['q4'])

    i2q = [q for q in questions if q!='_UNKQ_']
    q2i = {q:i for i, q in enumerate(i2q)}

    print('filter tr shape ', len(data))

    tr.to_csv(data_path + 'train.csv', index=False)
    te.to_csv(data_path + 'test.csv', index=False)
    with open(data_path+'i2q.json','w') as f:
        f.write(json.dumps(i2q,indent=4, separators=(',', ': ')))
    with open(data_path+'q2i.json','w') as f:
        f.write(json.dumps(q2i,indent=4, separators=(',', ': ')))

    # ----------------------------  ans label encode ------------------------------------
    answers = set()
    for i in range(5):
        answers = answers | set(tr['q' + str(i) + '_ans0']) | set(tr['q' + str(i) + '_ans1']) | set(tr['q' + str(i) + '_ans2'])
    answers = [a for a in list(answers) if a != '_UNKA_']
    print('num ans', len(answers))

    i2ans = sorted(answers)
    ans2i = {a:i for i,a in enumerate(i2ans)}
    useless_ans = {a: freq for a, freq in ans_counter.items() if a not in answers}


    with open(data_path+'ans2i.json','w') as f:
        f.write(json.dumps(ans2i,indent=4, separators=(',', ': ')))
    with open(data_path+'i2ans.json','w') as f:
        f.write(json.dumps(i2ans,indent=4, separators=(',', ': ')))
    with open(data_path + 'useless_ans.json', 'w') as f:
        f.write(json.dumps(useless_ans, indent=4, separators=(',', ': ')))

    # ----------------------------  create dataset ------------------------------------
    y = np.zeros((len(data),5,len(ans2i)))
    for i,samples in tr.iterrows():
        for j in range(5):
            for k in range(3):
                ans = samples['q'+str(j)+'_ans'+str(k)]
                if ans != '_UNKA_':
                    y[i,j,ans2i[ans]] = 1
    np.save(data_path+'label.npy',y)

    #------------------------------ make reg ------------------------------------
    get_prior()

    # -----------------------------  deal  wordvec  --------------------------------------
    read_wordvec('glove.42B.300d.txt','glove42')

def load_data(wordvec,cfg):

    tr = pd.read_csv(data_path + 'train.csv')
    te = pd.read_csv(data_path + 'test.csv')
    y = np.load(data_path+'label.npy')

    topk = cfg['attr_num']
    video_attr,attr_text,attr2i = load_video_attr(topk)
    with open(data_path+'i2q.json','r') as f:
        i2q = json.loads(f.read())
    with open(data_path+'q2i.json','r') as f:
        q2i = json.loads(f.read())

    questions = i2q
    [questions,attr_seq],embed_matrix = question2seq([questions,attr_text],wordvec,(TEXT_LEN,ATTR_LEN))
    print(questions.shape,attr_seq.shape)

    tr_data = []
    for i,samples in tr.iterrows():
        ques = []
        ans = []
        q_str = []
        for q_i in range(5):
            q = samples['q'+str(q_i)]
            q_seq = questions[q2i[q]] if q != '_UNKQ_' else q

            ques.append(q_seq)
            a_ = [samples['q'+str(q_i)+'_ans'+str(j)] for j in range(3)]
            a_ = [a for a in a_ if a!= '_UNKA_']
            ans.append(a_)
            q_str.append(q)

        need_q = [i for i,q in enumerate(ques) if q!='_UNKQ_']
        v_id = samples['video_id']

        attr = [attr_seq[attr2i[a]] for a in video_attr[v_id]]
        if len(attr) < topk:
            attr += (topk-len(attr))*[np.zeros(ATTR_LEN)]
        tr_data.append([v_id,ques,ans,q_str,need_q,attr])

    te_data = []
    for i,samples in te.iterrows():
        ques = []
        q_str = []
        for q_i in range(5):
            q = samples['q'+str(q_i)]
            q_seq = questions[q2i[q]] if q!='_UNKQ_' else q
            ques.append(q_seq)
            q_str.append(q)
        v_id = samples['video_id']
        attr = [attr_seq[attr2i[a]] for a in video_attr[v_id]]
        if len(attr) < topk:
            attr += (topk-len(attr))*[np.zeros(ATTR_LEN)]
        te_data.append([v_id,ques,None,q_str,None,attr])

    cfg['num_ans'] = y.shape[2]
    return tr_data,te_data,y,embed_matrix,cfg

class DataLoader:

    def __init__(self,training,data,y,cfg,q2c=None):

        with open(data_path + 'i2ans.json', 'r') as f:
            self.i2ans = json.loads(f.read())
        with open(data_path + 'ans2i.json', 'r') as f:
            self.ans2i = json.loads(f.read())
        if q2c is None:
            with open('./info/q2c.json', 'r') as f:
                self.q2c = json.loads(f.read())
        else:
            self.q2c = q2c

        with open('./info/c2v.pkl', 'rb') as f:
            c2v = pickle.load(f)

        if y is None:
            self.video_dir = data_path + 'rcnn/'
            self.get_data = self.get_data_te_
        else:
            self.video_dir = data_path + 'rcnn/'
            self.get_data = self.get_data_tr_

        self.y = y
        self.num_samples = len(data)
        self.data = {
            'video':[],
            'questions':[],
            'q_str':[],
            'ans':[],
            'valid_q':[],
            'attr':[],
            'prior':[]
        }

        self.data_distr = {'num_q':0}
        for i in range(len(data)):
            self.data['video'].append(self.video_dir + data[i][0] + '.npy')
            self.data['questions'].append(data[i][1])
            self.data['ans'].append(data[i][2])
            self.data['q_str'].append(data[i][3])
            self.data['valid_q'].append(data[i][4])
            self.data['attr'].append(data[i][5])
            prior = []
            for q in data[i][3]:
                if q != '_UNKQ_':
                    c = self.q2c[q]
                    self.data_distr[c] = self.data_distr.get(c,0) + 1
                    self.data_distr['num_q'] += 1
                    prior.append(c2v[c])
                else:
                    prior.append(np.zeros(len(self.i2ans)))
            self.data['prior'].append(prior)
        self.data['prior'] = np.array(self.data['prior'])
        self.data['video'] = np.array(self.data['video'])
        self.data['attr'] = np.array(self.data['attr'])

        self.batchsize = cfg['bs']
        self.curr_index = 0
        self.num_frame = cfg['num_frame']

        self.training = training


    def read_videos(self,files,split=1,offset=0,need_frame=None):
        def frame_sampling(frames,need_frame,offset=0):
            num_frame = frames.shape[0]
            step = num_frame / need_frame
            v_flow = []

            #  down sampling
            if len(frames)<need_frame:
                v_flow = frames.copy()
            elif self.training:
                for i in np.linspace(0+offset, num_frame+offset-1, need_frame):
                    if len(v_flow) >= need_frame or i>(num_frame-2):
                        break
                    j = np.random.randint(int(i),min(num_frame,int(i+step)))
                    v_flow.append(frames[j])
            else:
                if split > 1:
                    assert offset<split
                    offset_step = (step + split -1)//split
                    offset = min(offset*offset_step,step)
                else:
                    offset_step = 0
                for i in np.linspace(0+offset, num_frame-1+offset, need_frame):
                    if len(v_flow) >= need_frame or i>(num_frame-1):
                        break
                    v_flow.append(frames[int(i)])

            #  add padding
            v_flow = np.array(v_flow)
            if len(v_flow) < need_frame:
                shape = list(v_flow.shape)
                shape[0] = need_frame - shape[0]
                padding = np.zeros(shape)
                v_flow = np.concatenate([padding,v_flow],axis=0)
            return v_flow

        videos = []
        if need_frame is None:
            need_frame = self.num_frame
        for f in files:
            v = frame_sampling(np.load(f), need_frame, offset)
            videos.append(v)
        return np.array(videos)

    def random_concat(self):
        need_frame = self.num_frame // 2
        num_attr = self.data['attr'].shape[1]//2
        idx = np.arange(self.num_samples)

        data = {
            'video':[],
            'attr':[],
            'q0':[],
            'q1':[],
            'q2':[],
            'q3':[],
            'q4':[],
            'prior':[]
        }
        y = []
        while True:
            np.random.shuffle(idx)
            for i in range(0,self.num_samples,2):
                if i == self.num_samples-1:
                    continue
                pair1 = idx[i]
                pair2 = idx[i+1]

                videos = self.read_videos(self.data['video'][[pair1,pair2]],need_frame=need_frame)
                videos = np.concatenate(videos)
                data['video'].append(videos)

                labels = []
                prior = []
                num_q = np.random.randint(2, 4)
                valid_i = np.random.choice(self.data['valid_q'][pair1], num_q, replace=False)
                valid_i = np.concatenate([valid_i,np.random.choice(self.data['valid_q'][pair2], 5-num_q, replace=False)])
                for j in range(5):
                    q_i = valid_i[j]
                    if j < num_q:
                        use_pair = pair1
                    else:
                        use_pair = pair2
                    data['q' + str(j)].append(self.data['questions'][use_pair][q_i])
                    labels.append(self.y[use_pair, q_i])
                    prior.append(self.data['prior'][use_pair,q_i])
                y.append(np.array(labels))
                data['prior'].append(np.array(prior))

                attr = self.data['attr'][[pair1,pair2]][:,:num_attr]
                data['attr'].append(np.concatenate(attr))

                if len(data['video']) == self.batchsize:
                    for k in data.keys():
                        data[k] = np.array(data[k])
                    y = np.array(y)

                    yield data,y
                    data = {
                        'video': [],
                        'attr': [],
                        'q0': [],
                        'q1': [],
                        'q2': [],
                        'q3': [],
                        'q4': [],
                        'prior': []
                    }
                    y = []

    def sampling_question(self,idx):
        if self.y is None:
            data = {
                'q0': [],
                'q1': [],
                'q2': [],
                'q3': [],
                'q4': [],
                'prior':[]
            }
            for i in idx:
                q = self.data['questions'][i]
                for j in range(5):
                    data['q' + str(j)].append(q[j])
                data['prior'].append(self.data['prior'][i])
            for k in data.keys():
                data[k] = np.array(data[k])
            return data
        else:
            data = {
                'q0':[],
                'q1':[],
                'q2':[],
                'q3':[],
                'q4':[],
                'prior': []
            }
            labels = []
            for i in idx:
                q = self.data['questions'][i]
                y = self.y[i].copy()
                prior = self.data['prior'][i].copy()
                for j in range(5):
                    if q[j] == '_UNKQ_':
                        q_i = np.random.choice(self.data['valid_q'][i])
                        data['q'+str(j)].append(q[q_i])
                        y[j] = y[q_i]
                        prior[j] = prior[q_i]
                    else:
                        data['q'+str(j)].append(q[j])
                labels.append(y)
                data['prior'].append(prior)
            for k in data.keys():
                data[k] = np.array(data[k])
            return np.array(labels),data

    def get_data_tr_(self,split=1,offset=0,training=None,idx=None):
        if training is None:
            training = self.training
        if idx is None:
            idx = np.arange(self.num_samples)
        if training:
            np.random.shuffle(idx)
        for i in range(0,len(idx),self.batchsize):
            batch_idx = idx[i:i+self.batchsize]
            y,data = self.sampling_question(batch_idx)
            data.update({
                    'video':self.read_videos(self.data['video'][batch_idx],split,offset),
                    'attr':self.data['attr'][batch_idx],
                })
            yield data,y


    def get_data_te_(self,split=1,offset=0):
        self.curr_index = 0
        while self.curr_index < self.num_samples:
            next_i = self.curr_index+self.batchsize
            curr_i = self.curr_index
            batch_idx = list(range(curr_i,min(next_i,self.num_samples)))
            data = self.sampling_question(batch_idx)
            data.update({
                'video': self.read_videos(self.data['video'][curr_i:next_i], split, offset),
                'attr': self.data['attr'][curr_i:next_i],
            })
            self.curr_index = next_i
            yield data

    def evaluate(self,y_pred):
        score = 0
        for i,pred in enumerate(y_pred):
            for q_id in self.data['valid_q'][i]:
                ans = self.data['ans'][i][q_id]
                pred_ans = pred[q_id,:].argmax()
                pred_ans = self.i2ans[pred_ans]
                if pred_ans in ans:
                    score += 1
        score /= self.data_distr['num_q']
        print(score)
        return score

    def test_aug(self,model,split=2):
        steps = (self.num_samples + self.batchsize - 1) // self.batchsize
        y_pred = 0
        for offset in range(split):
            y_pred += model.predict_generator(self.get_data(split=split,offset=offset),steps=steps,workers=32)
            # if offset in [0]:
            #     print('offset',str(offset),self.evaluate(y_pred))
        y_pred/=split
        return y_pred
    
if __name__ == '__main__':

    # get_prior()
    # prepocess(10)

    cfg['atten_k'] = 0
    cfg['bs'] = 8
    cfg['seed'] = 47
    cfg['num_frame'] = 16
    cfg['patience'] = 5
    cfg['dim_q'] = 256
    cfg['dim_v'] = 384
    cfg['dim_a'] = 256
    cfg['dim_attr'] = 300
    cfg['attr_num'] = 96
    cfg['use_rc'] = True
    cfg['rc_max_iter'] = 12
    cfg['alpha'] = 0.65

    cfg['attention_avepool'] = True
    cfg['attention'] = 'coa'
    cfg['lr'] = 0.0002

    cfg['model_name'] = 'superguts'
    load_data('glove42',cfg)


