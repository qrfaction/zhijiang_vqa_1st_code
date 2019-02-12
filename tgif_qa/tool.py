from utils import *
from tqdm import tqdm
import time
import numpy as np
import pickle
from collections import Counter
from config import *
import json
import os

def prepocess(min_freq=7,num_q=5):

    tr = pd.read_csv(data_path+'Train_frameqa_question.csv',sep='\t')
    te = pd.read_csv(data_path+'Test_frameqa_question.csv',sep='\t')

    # --------------------------------- cal freq ans ------------------------------------
    num_samples = len(tr)
    ans_counter = dict(Counter(tr['answer'].tolist()+te['answer'].tolist()))
    tr['freq'] = tr['answer'].map(ans_counter)
    te['freq'] = te['answer'].map(ans_counter)
    tr = tr[tr['freq']>=min_freq]
    tr = tr[['gif_name','question', 'answer', 'type','freq']]
    te = te[['gif_name','question', 'answer', 'type', 'freq']]
    print(len(te))
    print(len(tr)/num_samples)
    if os.path.exists(data_path)==False:
        os.makedirs(data_path)
    with open(data_path+'ans_freq.json', 'w') as f:
        f.write(json.dumps(sorted(ans_counter.items(), key=lambda x:x[1], reverse=True),indent=4, separators=(',', ': ')))
    with open(data_path+'get_ansfreq.json', 'w') as f:
        f.write(json.dumps(ans_counter,indent=4, separators=(',', ': ')))

    questions = tr['question'].tolist()+te['question'].tolist()
    i2q = [q for q in set(questions) if q != '_UNKQ_']
    q2i = {q: i for i, q in enumerate(i2q)}

    tr.to_csv(data_path + 'train.csv', index=False)
    te.to_csv(data_path + 'test.csv', index=False)
    with open(data_path + 'i2q.json', 'w') as f:
        f.write(json.dumps(i2q, indent=4, separators=(',', ': ')))
    with open(data_path + 'q2i.json', 'w') as f:
        f.write(json.dumps(q2i, indent=4, separators=(',', ': ')))

    # --------------------------------- grouping -----------------------------------------
    tr_data = []
    for v_id,g in tr.groupby('gif_name'):
        data = g.values[:,1:].tolist()
        for i in range(0, len(g), num_q):
            samples = []
            for d in data[i:i+num_q]:
                samples += d
            if len(data[i:i+num_q])<num_q:
                samples += ['_UNKQ_','_UNKA_',-1,0]*(num_q-len(data[i:i+num_q]))

            record = [v_id] + samples
            tr_data.append(record)

    te_data = []
    for v_id, g in te.groupby('gif_name'):
        data = g.values[:,1:].tolist()
        for i in range(0, len(g), num_q):
            samples = []
            for d in data[i:i + num_q]:
                samples += d
            if len(data[i:i+num_q]) < num_q:
                samples += ['_UNKQ_', '_UNKA_', -1, 0] * (num_q - len(data[i:i+num_q]))

            record = [v_id] + samples
            te_data.append(record)

    columns = ['gif_name']
    for i in range(num_q):
        columns += ['q'+str(i),'ans'+str(i),'type'+str(i),'freq'+str(i)]
    tr = pd.DataFrame(tr_data,columns=columns)
    te = pd.DataFrame(te_data,columns=columns)
    tr.to_csv(data_path+'train.csv',index=False)
    te.to_csv(data_path+'test.csv',index=False)

    # ----------------------------  ans label encode ------------------------------------

    i2ans = [a for a,freq in ans_counter.items() if freq>=min_freq]
    ans2i = {a:i for i,a in enumerate(i2ans)}
    print('num ans',len(i2ans))

    with open(data_path+'ans2i.json','w') as f:
        f.write(json.dumps(ans2i,indent=4, separators=(',', ': ')))
    with open(data_path+'i2ans.json','w') as f:
        f.write(json.dumps(i2ans,indent=4, separators=(',', ': ')))


    # ----------------------------  create dataset ------------------------------------
    tr_y = np.zeros((len(tr_data),num_q,len(ans2i)))
    for i,samples in tr.iterrows():
        for j in range(num_q):
            ans = samples['ans'+str(j)]
            if ans != '_UNKA_':
                if ans not in ans2i:
                    print(ans,'---------------')
                tr_y[i,j,ans2i[ans]] = 1
    np.save(data_path+'tr_y.npy',tr_y)

    te_y = np.zeros((len(te_data), num_q, len(ans2i)))
    for i, samples in te.iterrows():
        for j in range(num_q):
            ans = samples['ans' + str(j)]
            if ans in ans2i:
                te_y[i,j,ans2i[ans]] = 1

    np.save(data_path + 'te_y.npy', te_y)
    # -----------------------------  deal  wordvec  --------------------------------------
    # read_wordvec('glove.42B.300d.txt','glove42')

    get_prior()
    if os.path.exists(model_path) == False:
        os.mkdir(model_path)

def load_data(wordvec,cfg):

    tr = pd.read_csv(data_path + 'train.csv')
    te = pd.read_csv(data_path + 'test.csv')
    tr_y = np.load(data_path+'tr_y.npy')
    te_y = np.load(data_path+'te_y.npy')

    num_q = cfg['num_q']
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
        qtype = []
        for q_i in range(num_q):
            q = samples['q'+str(q_i)]
            q_seq = questions[q2i[q]] if q != '_UNKQ_' else q
            ques.append(q_seq)

            qtype.append(samples['type'+str(q_i)])
            ans.append(samples['ans'+str(q_i)])


        need_q = [i for i,q in enumerate(ques) if q!='_UNKQ_']
        v_id = samples['gif_name']

        attr = [attr_seq[attr2i[a]] for a in video_attr[v_id]]
        if len(attr) < topk:
            attr += (topk-len(attr))*[np.zeros(ATTR_LEN)]
        tr_data.append([v_id,ques,ans,need_q,attr,qtype])

    te_data = []
    for i,samples in te.iterrows():
        ques = []
        ans = []
        qtype = []
        for q_i in range(num_q):
            q = samples['q'+str(q_i)]
            q_seq = questions[q2i[q]] if q != '_UNKQ_' else q
            ques.append(q_seq)
            qtype.append(samples['type' + str(q_i)])
            ans.append(samples['ans' + str(q_i)])
        need_q = [i for i, q in enumerate(ques) if q != '_UNKQ_']
        v_id = samples['gif_name']
        attr = [attr_seq[attr2i[a]] for a in video_attr[v_id]]
        if len(attr) < topk:
            attr += (topk-len(attr))*[np.zeros(ATTR_LEN)]
        te_data.append([v_id,ques,ans,need_q,attr,qtype])

    cfg['num_ans'] = tr_y.shape[-1]
    return tr_data,tr_y,te_data,te_y,embed_matrix,cfg

class DataLoader:

    def __init__(self,training,data,y,cfg):

        with open(data_path + 'i2ans.json', 'r') as f:
            self.i2ans = json.loads(f.read())
        with open(data_path + 'ans2i.json', 'r') as f:
            self.ans2i = json.loads(f.read())
        with open(data_path+'c2v.pkl', 'rb') as f:
            c2v = pickle.load(f)

        self.num_q = cfg['num_q']
        self.video_dir = data_path + 'rcnn/'
        self.need_frame = cfg['num_frame']
        self.y = y
        self.data = {
            'video':[],
            'questions':[],
            'ans':[],
            'valid_q':[],
            'attr':[],
            'prior':[]
        }

        self.num_samples = 0
        for i in range(len(data)):
            self.data['video'].append(self.video_dir + data[i][0] + '.npy')
            self.data['questions'].append(data[i][1])
            self.data['ans'].append(data[i][2])
            self.data['valid_q'].append(data[i][3])
            self.data['attr'].append(data[i][4])
            prior = []
            for q,c in zip(data[i][1],data[i][5]):
                if q != '_UNKQ_':
                    self.num_samples +=1
                    assert c!=-1
                    prior.append(c2v[c])
                else:
                    assert c==-1
                    prior.append(np.zeros(len(self.i2ans)))
            self.data['prior'].append(prior)

        self.data['prior'] = np.array(self.data['prior'])
        self.data['video'] = np.array(self.data['video'])
        self.data['attr'] = np.array(self.data['attr'])

        self.batchsize = cfg['bs']
        self.curr_index = 0

        self.training = training
        self.num_gifs = len(self.data['video'])



    def read_videos(self,files,read_half=False):
        videos = []
        if read_half:
            for f in files:
                init = np.random.randint(0,2)
                videos.append(np.load(f)[init::2])
        else:
            for f in files:
                videos.append(np.load(f))
        return np.array(videos)

    def random_concat(self):
        num_attr = self.data['attr'].shape[1]//2
        idx = np.arange(self.num_gifs)

        data = {
            'video':[],
            'attr':[],
            'prior':[]
        }
        for i in range(self.num_q):
            data['q'+str(i)] = []
        y = []
        while True:
            np.random.shuffle(idx)
            for i in range(0,self.num_gifs,2):
                if i == self.num_gifs-1:
                    continue
                pair1 = idx[i]
                pair2 = idx[i+1]

                videos = self.read_videos(self.data['video'][[pair1,pair2]],read_half=True)
                videos = np.concatenate(videos)
                data['video'].append(videos)

                labels = []
                prior = []
                num_q = np.random.randint(self.num_q//2,2+self.num_q//2)
                valid_i = np.random.choice(self.data['valid_q'][pair1], num_q, replace=True)
                valid_i = np.concatenate([valid_i,np.random.choice(self.data['valid_q'][pair2], self.num_q-num_q, replace=True)])
                for j in range(self.num_q):
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
                        'prior': []
                    }
                    for i in range(self.num_q):
                        data['q' + str(i)] = []
                    y = []

    def sampling_question(self,idx):
        data = {
            'video': [],
            'attr': [],
            'prior': []
        }
        for i in range(self.num_q):
            data['q' + str(i)] = []

        labels = []
        for i in idx:
            q = self.data['questions'][i]
            y = self.y[i].copy()
            prior = self.data['prior'][i].copy()
            for j in range(self.num_q):
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

    def get_data(self,training=None,idx=None):
        if training is None:
            training = self.training
        if idx is None:
            idx = np.arange(self.num_gifs)

        while True:
            if training == True:
                np.random.shuffle(idx)
            for i in range(0,len(idx),self.batchsize):
                batch_idx = idx[i:i+self.batchsize]
                y,data = self.sampling_question(batch_idx)
                data.update({
                        'video':self.read_videos(self.data['video'][batch_idx]),
                        'attr':self.data['attr'][batch_idx],
                    })
                yield data,y

    def evaluate(self,y_pred):
        score = 0
        for i,pred in enumerate(y_pred):
            for q_id in self.data['valid_q'][i]:
                ans = self.data['ans'][i][q_id]
                pred_ans = pred[q_id,:].argmax()
                pred_ans = self.i2ans[pred_ans]
                if pred_ans == ans:
                    score += 1
        return score/self.num_samples




if __name__ == '__main__':
    prepocess(num_q=4)
    # get_info()









