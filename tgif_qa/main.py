import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)
from tool import *
from model import *
import warnings
warnings.filterwarnings('ignore')

def main(cfg):

    bs = cfg['bs']
    patience = cfg['patience']

    tr, tr_y, te, te_y, embed_matrix, cfg = load_data('glove42',cfg)

    te_g = DataLoader(False,te,te_y,cfg)
    tr_g = DataLoader(True,tr,tr_y,cfg)
    print(cfg)
    print('tr num gifs', tr_g.num_gifs, '\t', 'num question', tr_g.num_samples)
    print('te num gifs', te_g.num_gifs, '\t', 'num question', te_g.num_samples)

    model = cfg['model'](embed_matrix, cfg)

    best_iter = 0
    best_score = 0
    for i in range(1,21):
        print(i)
        if cfg['use_mixup'] and i < cfg['mix_up_max_iter']:
            if i % 2 == 1:
                g = tr_g.random_concat()
            else:
                g = tr_g.get_data()
            steps = (tr_g.num_gifs + bs - 1) // bs
        else:
            g = tr_g.get_data()
            steps = (tr_g.num_gifs + bs - 1) // bs
        model.fit_generator(
            g,
            steps_per_epoch=steps,
            epochs=1,
            verbose=cfg['verbose'],
            workers=32,
        )
        score = te_g.evaluate(model.predict_generator(
            te_g.get_data(),
            steps=(te_g.num_gifs + bs - 1) // bs,
            workers=32
        ))
        print(score)
        print(tr_g.evaluate(model.predict_generator(
            tr_g.get_data(training=False),
            steps=(tr_g.num_gifs + bs - 1) // bs,
            workers=32
        )))
        if score > best_score:
            best_iter = i
            best_score = score
            model.save(model_path+cfg['model_name']+'.h5')


if __name__ == '__main__':

    # cfg = {}
    cfg['bs'] = 8
    cfg['num_frame'] = 16
    cfg['patience'] = 4
    cfg['dim_q'] = 256
    cfg['dim_v'] = 384
    cfg['dim_a'] = 256

    cfg['attr_num'] = 128
    cfg['verbose'] = 0
    cfg['use_mixup'] = True

    cfg['mix_up_max_iter'] = 6
    cfg['alpha'] = 0.65
    cfg['lr_decay'] = 1
    cfg['atten_k'] = 96
    cfg['lr'] = 0.0002

    cfg['num_q'] = 4
    cfg['loss'] = focal_loss_fixed

    cfg['model'] = attention_model

    cfg['q_share'] = True


    # cfg['loss'] = 'binary_crossentropy'
    # cfg['model_name'] = 'bce'
    # main(cfg)
    #
    # cfg['q_share'] = separate_model
    # cfg['q_share'] = False
    # cfg['loss'] = 'binary_crossentropy'
    # cfg['model_name'] = 'nosbce'
    # main(cfg)

    cfg['mix_up_max_iter'] = 0
    cfg['q_share'] = attention_model
    cfg['q_share'] = True
    cfg['model_name'] = 'nosnorc'
    main(cfg)

    cfg['mix_up_max_iter'] = 0
    cfg['q_share'] = separate_model
    cfg['q_share'] = False
    cfg['model_name'] = 'nosnorc'
    main(cfg)



