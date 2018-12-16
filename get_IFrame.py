import os
from tqdm import tqdm
import json
from glob import glob
import multiprocessing as mp



def worker(files,worker_id):
    def extract_all(file):
        path = file[:-4] + str(worker_id) +'temp.txt'
        shell_cmd = \
            "ffprobe -select_streams v -show_frames -show_entries frame=pict_type -of csv {} | grep -n I | cut -d ':' -f 1 > {}".format(
                file, path)
        os.system(shell_cmd)
        with open(path,'r') as f:
            idx = f.read().split('\n')
            idx = [int(i)-1 for i in idx if i!='']
        os.remove(path)
        return idx

    v = {}
    for f in tqdm(files):
        video_id = f.split('/')[-1][:-4]
        v[video_id] = extract_all(f)
    return v

def get_IFrame_idx(files=None):
    if files is None:
        data_path = '../data/'
        # files = glob(data_path + 'DatasetA/train/*.mp4') + \
        #         glob(data_path + 'DatasetA/test/*.mp4') + \
        #         glob(data_path + 'DatasetB/train/*.mp4') + \
        #         glob(data_path + 'DatasetB/test/*.mp4')
        files = glob(data_path + 'test/*.mp4')
    

    results = []
    pool = mp.Pool(mp.cpu_count())
    aver_t = int(len(files) / mp.cpu_count()) + 1

    for i in range(mp.cpu_count()):
        result = pool.apply_async(worker,
                        args=(files[i * aver_t: (i + 1) * aver_t],i))
        results.append(result)
    pool.close()
    pool.join()


    v = {}
    for result in results:
        v.update(result.get())
    if os.path.exists('./info/') == False:
        os.mkdir('./info/')
    with open('./info/I_Frame_idx.json','w') as f:
        f.write(json.dumps(v,indent=4, separators=(',', ': ')))

    print('done !')

if __name__ == '__main__':
    get_IFrame_idx()


