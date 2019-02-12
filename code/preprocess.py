

from utils import *
def pipeline():

    # -----------------------  检查文件是否存在 ----------------------------
    assert os.path.exists(word_vec_dir + 'glove.42B.300d.txt'),'请将glove.42B.300d.txt放到 /data/word_vec/ 文件夹下'
    assert os.path.exists(data_path + 'DatasetA/train.txt'),'data/DatasetA/train.txt 不存在'
    assert os.path.exists(data_path + 'DatasetB/train.txt'),'data/DatasetB/train.txt 不存在'


    if os.path.exists('./info/') == False:
        os.mkdir('./info/')



    # ------------------------ 清洗训练集 -----------------------------------
    concat_dataset(['DatasetA/train.txt', 'DatasetB/train.txt'], 'train.txt')
    with open(data_path + 'train.txt', 'r') as f:
        tr = f.read()
    lines = tr.split('\n')
    assert len(lines)>10000,'训练集样本理应大于10000'

    for error, correct in fix_error.items():
        tr = tr.replace(error, correct)

    with open(data_path + 'train.txt', 'w') as f:
        f.write(tr)
    concat_dataset(['DatasetB/test.txt'], 'test.txt')

    prepocess(10)

if __name__ == "__main__":

    pipeline()

