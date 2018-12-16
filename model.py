from keras.layers import *
from keras.models import Model
from config import *
from keras.optimizers import *
from keras.regularizers import *
import tensorflow as tf

def focal_loss_fixed(y_true, y_pred,gamma=2):
    alpha = cfg['alpha']
    z = K.sum(y_true) + K.sum(1-y_true)
    pt_1 = tf.where(tf.less_equal(0.5,y_true), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.less_equal(y_true, 0.5),y_pred, tf.zeros_like(y_pred))
    pos_loss = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))
    neg_loss = -K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return (pos_loss+neg_loss)/z

def softmax(x, axis=2):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)

def acc(y_true_, y_pred_):

    acc = 0
    for i in range(5):
        y_true = y_true_[:,i,:]

        correct_mask = y_true
        error_mask = 1-correct_mask

        y_pred = y_pred_[:,i,:]
        correct_score = K.max(y_pred*correct_mask,axis=1)
        error_score = K.max(y_pred*error_mask,axis=1)

        score = correct_score - error_score
        score = tf.greater(score,0)
        acc += K.mean(score)

    return acc / 5

def share_layer(q,layer):
    q_out = []
    for q_i in q:
        q_out.append(layer(q_i))
    return q_out

def distance(q,v,dist,axis=2):

    if dist == 'cos':
        dis_f = lambda x: x[0] * x[1]

    elif dist == 'h_mean':
        def dis_f(x):
            return x[0]*x[1]/(K.sum(K.abs(x[0]),axis=axis,keepdims=True)+K.sum(K.abs(x[1]),axis=axis,keepdims=True))
    elif dist == 'dice':
        def dis_f(x):
            return x[0]*x[1]/(K.sum(x[0]**2,axis=axis,keepdims=True)+K.sum(x[1]**2,axis=axis,keepdims=True))
    elif dist == 'jaccard':
        def dis_f(x):
            return  x[0]*x[1]/(
                    K.sum(x[0]**2,axis=axis,keepdims=True)+
                    K.sum(x[1]**2,axis=axis,keepdims=True)-
                    K.sum(K.abs(x[0]*x[1]),axis=axis,keepdims=True))
    elif dist == 'dice_dot':
        def dis_f(x):
            x0 = x[0]
            x1 = K.permute_dimensions(x[1],(0,2,1))
            x_norm = (K.sum(x0**2,axis=2,keepdims=True)+K.sum(x1**2,axis=1,keepdims=True))
            return K.batch_dot(x0,x1)/x_norm
    elif dist == 'cos_dot':
        def dis_f(x):
            x0 = x[0]
            x1 = K.permute_dimensions(x[1],(0,2,1))
            return K.batch_dot(x0,x1)
    else:
        raise RuntimeError("dist error")

    return Lambda(dis_f)([q,v])

def temporal_attention(q,v,use_conv=False):

    def get_atten_w(q_list, v_encode,fca):
        w_list = []
        for q in q_list:
            merged = distance(q,v_encode,'dice')
            w = fca(merged)
            w = Flatten()(w)
            w = Activation('softmax')(w)
            w_list.append(w)
        w = average(w_list)
        w = Reshape((-1, 1, 1))(w)
        return w

    fcq = Dense(cfg['dim_a'],activation='relu')

    q = share_layer(q,GlobalAveragePooling1D())
    q = share_layer(q,fcq)

    if cfg['attention_avepool']:
        fca = Dense(1, use_bias=False)
        q = share_layer(q, Reshape((1, -1)))
        v_encode = TimeDistributed(GlobalAveragePooling1D())(v)  # 16 *2048
        v_encode = Dense(cfg['dim_a'], activation='tanh')(v_encode)
        w = get_atten_w(q, v_encode, fca)
    else:
        fca = Dense(1, use_bias=False,activation='sigmoid')
        q = share_layer(q, Reshape((1,1,-1)))
        v_encode = Conv2D(filters=cfg['dim_a'], kernel_size=[1,1],activation='tanh')(v)
        w = []
        for q_i in q:
            merged = distance(q_i, v_encode, 'dice',axis=3)
            w_i = fca(merged)
            w.append(w_i)
        w = average(w)

    if use_conv:
        v = Lambda(lambda x: x[0] * x[1])([v, w])
        v = Conv2D(filters=cfg['dim_v'], kernel_size=[1,1], padding='same')(v)
        v = BatchNormalization()(v)
        v = Activation('relu')(v)
        # v = Lambda(lambda x: x[0] * x[1])([v, w])
    else:
        v = Lambda(lambda x: x[0] * x[1])([v, w])
    return v

def time_spatial_attention(q,v):

    fcaq = Dense(1)
    fcav = Dense(1)
    fcq = Dense(units=cfg['dim_a'],activation='relu')
    v = Reshape((-1, cfg['dim_v']))(v)
    v_encode = Dense(units=cfg['dim_a'], activation='tanh')(v)
    v_encode = SpatialDropout1D(0.3)(v_encode)

    q_out = []
    v_out = []
    if cfg['attention'] == 'coa':
        for i in range(len(q)):
            q_encode = fcq(q[i])
            q_encode = SpatialDropout1D(0.3)(q_encode)
            atten = Lambda(lambda x: K.batch_dot(x[0], x[1]))([q_encode, Permute((2, 1))(v_encode)])  # 14 * 40


            atten_q = atten
            atten_q = fcaq(atten_q)  # 14*1
            atten_q = Flatten()(atten_q)
            atten_q = Activation('softmax')(atten_q)
            atten_q = Reshape((1, -1))(atten_q)  # 1*14
            q_i = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten_q, q[i]])  # 1*300
            q_out.append(Flatten()(q_i))


            atten_v = atten
            atten_v = fcav(Permute((2, 1))(atten_v))
            atten_v = Flatten()(atten_v)

            if cfg['atten_k']>0:
                v_i = Lambda(top_k_ave, arguments={'k': cfg['atten_k']})([atten_v, v])
            else:
                atten_v = Activation('softmax')(atten_v)
                atten_v = Reshape((1, -1))(atten_v)
                v_i = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten_v, v])
                v_i = Lambda(lambda x: K.squeeze(x, axis=1))(v_i)
            v_out.append(v_i)
    elif cfg['attention']=='mlb':
        q = share_layer(q,GlobalAveragePooling1D())
        q_out = q
        for i in range(len(q)):
            q_encode = fcq(q[i])
            q_encode = Dropout(0.3)(q_encode)
            q_encode = Reshape((1,-1))(q_encode)
            atten = distance(q_encode,v_encode,'cos')  # 14 * 40

            atten = fcav(atten)
            atten = Flatten()(atten)
            if cfg['atten_k'] > 0:
                v_i = Lambda(top_k_ave, arguments={'k': cfg['atten_k']})([atten, v])
            else:
                atten = Activation('softmax')(atten)
                atten = Reshape((1, -1))(atten)
                v_i = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten, v])
                v_i = Lambda(lambda x: K.squeeze(x, axis=1))(v_i)
            v_out.append(v_i)
    elif cfg['attention']=='top_down':
        q = share_layer(q, GlobalAveragePooling1D())
        q_out = q
        fca1 = Dense(cfg['dim_a'],activation='relu')
        fca2 = Dense(1)
        v_encode = SpatialDropout1D(0.3)(v)
        for i in range(len(q)):
            q_encode = q[i]
            q_encode = Dropout(0.3)(q_encode)
            q_encode = RepeatVector(cfg['num_frame']*36)(q_encode)
            atten = concatenate([q_encode,v_encode],axis=-1)
            atten = fca1(atten)
            atten = fca2(atten)
            atten = Flatten()(atten)

            if cfg['atten_k'] > 0:
                v_i = Lambda(top_k_ave, arguments={'k': cfg['atten_k']})([atten, v])
            else:
                atten = Activation('softmax')(atten)
                atten = Reshape((1, -1))(atten)
                v_i = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten, v])
                v_i = Lambda(lambda x: K.squeeze(x, axis=1))(v_i)
            v_out.append(v_i)
    elif cfg['attention'] == 'san':
        q = share_layer(q, GlobalAveragePooling1D())
        q_out = q
        fcq1 = Dense(cfg['dim_a'])
        fcq2 = Dense(cfg['dim_a'])

        fca1 = Dense(1)
        fca2 = Dense(1)

        v = Dense(units=cfg['dim_a'], activation='tanh')(v)
        v_encode1 = Dense(cfg['dim_a'])(v)
        v_encode1 = SpatialDropout1D(0.3)(v_encode1)
        v_encode2 = Dense(cfg['dim_a'])(v)
        v_encode2 = SpatialDropout1D(0.3)(v_encode2)
        for i in range(len(q)):
            q_i = fcq(q[i])
            q_i = Dropout(0.3)(q_i)
            q_encode = fcq1(q_i)
            q_encode = RepeatVector(cfg['num_frame'] * 36)(q_encode)
            atten = Activation('tanh')(add([q_encode,v_encode1]))
            atten = fca1(atten)
            atten = Flatten()(atten)

            if cfg['atten_k'] > 0:
                v_i = Lambda(top_k_ave, arguments={'k': cfg['atten_k']})([atten, v])
            else:
                atten = Activation('softmax')(atten)
                atten = Reshape((1, -1))(atten)
                v_i = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten, v])
                v_i = Lambda(lambda x: K.squeeze(x, axis=1))(v_i)

            u_i = add([q_i,v_i])
            q_encode = fcq2(u_i)
            q_encode = RepeatVector(cfg['num_frame'] * 36)(q_encode)
            atten = Activation('tanh')(add([q_encode, v_encode2]))
            atten = fca2(atten)
            atten = Flatten()(atten)

            if cfg['atten_k'] > 0:
                v_i = Lambda(top_k_ave, arguments={'k': cfg['atten_k']})([atten, v])
            else:
                atten = Activation('softmax')(atten)
                atten = Reshape((1, -1))(atten)
                v_i = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten, v])
                v_i = Lambda(lambda x: K.squeeze(x, axis=1))(v_i)
            v_i = add([v_i,u_i])
            v_out.append(v_i)
    else:
        raise RuntimeError("don't have this attention")
    return q_out, v_out

def temporal_attention_single(q,v,use_conv=False):
    def get_atten_w(q, v, fca):
        merged = distance(q,v,'dice')
        w = fca(merged)
        w = Flatten()(w)
        w = Activation('softmax')(w)
        return w
    assert len(q) == len(v)

    fcq = Dense(cfg['dim_a'], activation='relu')
    fcv = Dense(cfg['dim_a'], activation='tanh')
    fca = Dense(1, use_bias=False)

    q = share_layer(q, GlobalAveragePooling1D())
    q = share_layer(q, fcq)
    q = share_layer(q, Reshape((1, -1)))

    encode_cnn = Conv2D(filters=cfg['dim_v'],kernel_size=[1,1],padding='same')
    bn = BatchNormalization()
    for i in range(len(q)):
        v_encode = TimeDistributed(GlobalAveragePooling1D())(v[i])  # 16 *2048
        v_encode = fcv(v_encode)
        w = get_atten_w(q[i], v_encode,fca)
        w = Reshape((-1, 1, 1))(w)

        if use_conv:
            v[i] = Lambda(lambda x:x[0]*x[1])([v[i],w])
            v[i] = encode_cnn(v[i])
            v[i] = bn(v[i])
            v[i] = Activation('relu')(v[i])
        else:
            v[i] = Lambda(lambda x: x[0] * x[1])([v[i], w])

    return v

def time_spatial_attention_single(q,v):

    fcaq = Dense(1)
    fcav = Dense(1)
    fcq = Conv1D(filters=cfg['dim_a'],kernel_size=1,activation='relu')
    fcv = Dense(units=cfg['dim_a'], activation='tanh')


    v = share_layer(v,Reshape((-1, cfg['dim_v'])))
    v_encode = share_layer(v,fcv)
    v_encode = share_layer(v_encode,SpatialDropout1D(0.3))

    q_out = []
    v_out = []

    if cfg['attention'] == 'coa':
        for i in range(len(q)):
            q_encode = fcq(q[i])
            q_encode = SpatialDropout1D(0.3)(q_encode)
            atten = Lambda(lambda x: K.batch_dot(x[0], x[1]))([q_encode, Permute((2, 1))(v_encode[i])])  # 14 * 40

            atten_q = atten
            atten_q = fcaq(atten_q)  # 14*1
            atten_q = Flatten()(atten_q)
            atten_q = Activation('softmax')(atten_q)
            atten_q = Reshape((1, -1))(atten_q)  # 1*14
            q_i = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten_q, q[i]])  # 1*300
            q_out.append(Flatten()(q_i))

            atten_v = atten
            atten_v = fcav(Permute((2, 1))(atten_v))
            atten_v = Flatten()(atten_v)

            if cfg['atten_k'] > 0:
                v_i = Lambda(top_k_ave, arguments={'k': cfg['atten_k']})([atten_v, v[i]])
            else:
                atten_v = Activation('softmax')(atten_v)
                atten_v = Reshape((1, -1))(atten_v)
                v_i = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten_v, v[i]])
                v_i = Lambda(lambda x: K.squeeze(x, axis=1))(v_i)
            v_out.append(v_i)
    elif cfg['attention'] == 'mlb':
        q = share_layer(q, GlobalAveragePooling1D())
        q_out = q
        for i in range(len(q)):
            q_encode = fcq(q[i])
            q_encode = Dropout(0.3)(q_encode)
            q_encode = Reshape((1, -1))(q_encode)
            atten = distance(q_encode, v_encode[i], 'cos')  # 14 * 40

            atten = fcav(atten)
            atten = Flatten()(atten)
            if cfg['atten_k'] > 0:
                v_i = Lambda(top_k_ave, arguments={'k': cfg['atten_k']})([atten, v[i]])
            else:
                atten = Activation('softmax')(atten)
                atten = Reshape((1, -1))(atten)
                v_i = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten, v[i]])
                v_i = Lambda(lambda x: K.squeeze(x, axis=1))(v_i)
            v_out.append(v_i)
    elif cfg['attention'] == 'top_down':
        q = share_layer(q, GlobalAveragePooling1D())
        q_out = q
        fca1 = Dense(cfg['dim_a'], activation='relu')
        fca2 = Dense(1)
        v_encode = share_layer(v,SpatialDropout1D(0.3))
        for i in range(len(q)):
            q_encode = q[i]
            q_encode = Dropout(0.3)(q_encode)
            q_encode = RepeatVector(cfg['num_frame'] * 36)(q_encode)
            atten = concatenate([q_encode, v_encode[i]], axis=-1)
            atten = fca1(atten)
            atten = fca2(atten)
            atten = Flatten()(atten)

            if cfg['atten_k'] > 0:
                v_i = Lambda(top_k_ave, arguments={'k': cfg['atten_k']})([atten, v[i]])
            else:
                atten = Activation('softmax')(atten)
                atten = Reshape((1, -1))(atten)
                v_i = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten, v[i]])
                v_i = Lambda(lambda x: K.squeeze(x, axis=1))(v_i)
            v_out.append(v_i)
    elif cfg['attention'] == 'san':
        q = share_layer(q, GlobalAveragePooling1D())
        q_out = q
        fcq1 = Dense(cfg['dim_a'])
        fcq2 = Dense(cfg['dim_a'])

        fca1 = Dense(1)
        fca2 = Dense(1)

        v = share_layer(v,Dense(units=cfg['dim_a'], activation='tanh'))
        v_encode1 = share_layer(v,Dense(cfg['dim_a']))
        v_encode1 = share_layer(v_encode1,SpatialDropout1D(0.3))
        v_encode2 = share_layer(v,Dense(cfg['dim_a']))
        v_encode2 = share_layer(v_encode2,SpatialDropout1D(0.3))
        for i in range(len(q)):
            q_i = fcq(q[i])
            q_i = Dropout(0.3)(q_i)
            q_encode = fcq1(q_i)
            q_encode = RepeatVector(cfg['num_frame'] * 36)(q_encode)
            atten = Activation('tanh')(add([q_encode, v_encode1[i]]))
            atten = fca1(atten)
            atten = Flatten()(atten)

            if cfg['atten_k'] > 0:
                v_i = Lambda(top_k_ave, arguments={'k': cfg['atten_k']})([atten, v[i]])
            else:
                atten = Activation('softmax')(atten)
                atten = Reshape((1, -1))(atten)
                v_i = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten, v[i]])
                v_i = Lambda(lambda x: K.squeeze(x, axis=1))(v_i)

            u_i = add([q_i, v_i])
            q_encode = fcq2(u_i)
            q_encode = RepeatVector(cfg['num_frame'] * 36)(q_encode)
            atten = Activation('tanh')(add([q_encode, v_encode2[i]]))
            atten = fca2(atten)
            atten = Flatten()(atten)

            if cfg['atten_k'] > 0:
                v_i = Lambda(top_k_ave, arguments={'k': cfg['atten_k']})([atten, v[i]])
            else:
                atten = Activation('softmax')(atten)
                atten = Reshape((1, -1))(atten)
                v_i = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten, v[i]])
                v_i = Lambda(lambda x: K.squeeze(x, axis=1))(v_i)
            v_i = add([v_i, u_i])
            v_out.append(v_i)
    else:
        raise RuntimeError("don't have this attention")

    return q_out, v_out


def top_k_ave(x,k):
    def softmax(x, axis=1):
        ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
        return ex / K.sum(ex, axis=axis, keepdims=True)

    w = x[0]
    feature = x[1]

    top_k_w,idx = tf.nn.top_k(w,k,sorted=False)
    top_k_w = softmax(top_k_w)
    top_k_w = K.expand_dims(top_k_w,axis=1)
    idx = K.expand_dims(idx, axis=2)

    batch_size = tf.shape(feature)[0]
    i_mat = tf.transpose(tf.reshape(tf.tile(tf.range(batch_size), [k]),
                                    [k, batch_size]))
    i_mat = K.expand_dims(i_mat,axis=2)
    idx = K.concatenate([i_mat,idx],axis=2)
    feature = tf.gather_nd(feature,idx)
    output = K.batch_dot(top_k_w,feature)
    output = K.squeeze(output,axis=1)
    return output

def attr_block(q_hist,attr):

    def get_attr(q,fcq,fca,attr,attr_encode):
        q_ave = GlobalAveragePooling1D()(q)
        q_ave = fcq(q_ave)
        q_ave = Reshape((-1, cfg['dim_a']))(q_ave)  # 1  512

        merged = distance(q_ave,attr_encode,'dice')
        w = fca(merged)
        w = Flatten()(w)

        w = Activation('softmax')(w)
        w = Reshape((1,-1))(w)
        attr = Lambda(lambda x:K.batch_dot(x[0],x[1]))([w,attr])
        attr = Lambda(lambda x:K.squeeze(x,axis=1))(attr)
        return attr

    attr = Conv2D(cfg['dim_attr'], kernel_size=[1, ATTR_LEN],activation='relu')(attr)
    attr = Lambda(lambda x: K.squeeze(x, axis=2))(attr)

    q1, q2, q3, q4, q5 = q_hist

    attr_encode = Conv1D(filters=cfg['dim_a'], activation='tanh', kernel_size=1)(attr)
    fcq = Dense(cfg['dim_a'],activation='relu')
    fca = Dense(1, use_bias=False)

    attr1 = get_attr(q1,fcq, fca,attr,attr_encode)
    attr2 = get_attr(q2,fcq, fca,attr,attr_encode)
    attr3 = get_attr(q3,fcq, fca,attr,attr_encode)
    attr4 = get_attr(q4,fcq, fca,attr,attr_encode)
    attr5 = get_attr(q5,fcq, fca,attr,attr_encode)

    return attr1,attr2,attr3,attr4,attr5

def attention_model(embed_mat=None):

    def question_prepocess(q,emb):
        q = emb(q)
        q = BatchNormalization()(q)
        q = SpatialDropout1D(0.2)(q)    #涨
        return q

    num_frame = cfg['num_frame']
    frame_size = 2048
    num_ans = cfg['num_ans']
    num_o = 36
    attr_num = cfg['attr_num']


    question1 = Input(shape=(TEXT_LEN,), name='q0')
    question2 = Input(shape=(TEXT_LEN,), name='q1')
    question3 = Input(shape=(TEXT_LEN,), name='q2')
    question4 = Input(shape=(TEXT_LEN,), name='q3')
    question5 = Input(shape=(TEXT_LEN,), name='q4')
    video = Input(shape=(num_frame,num_o,frame_size), name='video')
    attrs_in = Input(shape=(attr_num,ATTR_LEN),name='attr')

    v = video
    v = BatchNormalization()(v)
    v = SpatialDropout2D(0.05)(v)


    if embed_mat is None:
        emb = Embedding(
            cfg['num_word'],
            300,
            trainable=False,
            name='embedd'
        )
    else:
        emb = Embedding(
            embed_mat.shape[0],
            embed_mat.shape[1],
            weights=[embed_mat],
            trainable=False,
            name='embedd'
        )

    attrs = TimeDistributed(emb)(attrs_in)
    attrs = BatchNormalization()(attrs)


    q1 = question_prepocess(question1,emb)
    q2 = question_prepocess(question2,emb)
    q3 = question_prepocess(question3,emb)
    q4 = question_prepocess(question4,emb)
    q5 = question_prepocess(question5,emb)
    v = temporal_attention([q1, q2, q3, q4, q5], v,use_conv=False)


    gru = Bidirectional(CuDNNGRU(cfg['dim_q'], return_sequences=True), merge_mode='sum')
    q1, q2, q3, q4, q5 = share_layer([q1, q2, q3, q4, q5],gru)
    v = temporal_attention([q1, q2, q3, q4, q5], v,use_conv=True)

    gru = Bidirectional(CuDNNGRU(cfg['dim_q'], return_sequences=True), merge_mode='sum')
    q1, q2, q3, q4, q5 = share_layer([q1, q2, q3, q4, q5], gru)
    v = temporal_attention([q1, q2, q3, q4, q5], v,use_conv=True)


    a1, a2, a3, a4, a5 = attr_block([q1, q2, q3, q4, q5], attrs)
    q_out, v_out = time_spatial_attention([q1, q2, q3, q4, q5], v)


    # -------------------------  merge ------------------------

    ansq = concatenate(share_layer(q_out,Reshape((1, -1))),axis=1)
    ansq = SpatialDropout1D(0.3)(ansq)

    ansv = concatenate(share_layer(v_out, Reshape((1, -1))),axis=1)
    ansv = SpatialDropout1D(0.2)(ansv)

    ans_attr = concatenate(share_layer([a1, a2, a3, a4, a5], Reshape((1, -1))), axis=1)
    ans_attr = SpatialDropout1D(0.5)(ans_attr)


    q_share = [
        average([q_out[1], q_out[2], q_out[3], q_out[4]]),
        average([q_out[0], q_out[2], q_out[3], q_out[4]]),
        average([q_out[1], q_out[0], q_out[3], q_out[4]]),
        average([q_out[1], q_out[2], q_out[0], q_out[4]]),
        average([q_out[1], q_out[2], q_out[3], q_out[0]]),
    ]
    ans_q_share = concatenate(share_layer(q_share, Reshape((1, -1))), axis=1)
    ans_q_share = SpatialDropout1D(0.3)(ans_q_share)
    features = [ansq,ansv,ans_q_share,ans_attr]
    merged = concatenate(features)

    # ----------------------------------  fc ----------------------------------
    merged = TimeDistributed(Dense(512,activation='relu'))(merged)
    merged = SpatialDropout1D(0.3)(merged)
    fc = TimeDistributed(Dense(num_ans,use_bias=True,activation='sigmoid'))
    output = fc(merged)
    prior_in = Input((5, num_ans), name='prior')
    output = multiply([prior_in,output])


    #binary_crossentropy
    model = Model(inputs=[question1, question2, question3, question4, question5, video,attrs_in,prior_in], outputs=output)
    if cfg['predict']:
        return model

    model.compile(loss=cfg['loss'], optimizer=Nadam(cfg['lr']), metrics=[acc])

    # model.summary()
    return model


def separate_model(embed_mat):

    def question_prepocess(q,emb):
        q = emb(q)
        q = BatchNormalization()(q)
        q = SpatialDropout1D(0.2)(q)    #涨
        return q

    num_frame = cfg['num_frame']
    frame_size = 2048
    num_ans = cfg['num_ans']
    num_o = 36
    attr_num = cfg['attr_num']


    question1 = Input(shape=(TEXT_LEN,), name='q0')
    question2 = Input(shape=(TEXT_LEN,), name='q1')
    question3 = Input(shape=(TEXT_LEN,), name='q2')
    question4 = Input(shape=(TEXT_LEN,), name='q3')
    question5 = Input(shape=(TEXT_LEN,), name='q4')
    video = Input(shape=(num_frame,num_o,frame_size), name='video')
    attrs_in = Input(shape=(attr_num,ATTR_LEN),name='attr')

    v = video
    v = BatchNormalization()(v)
    v = SpatialDropout2D(0.05)(v)

    emb = Embedding(
        embed_mat.shape[0],
        embed_mat.shape[1],
        weights=[embed_mat],
        trainable=False,
        name='embedd'
    )
    attrs = TimeDistributed(emb)(attrs_in)
    attrs = BatchNormalization()(attrs)

    q1 = question_prepocess(question1,emb)
    q2 = question_prepocess(question2,emb)
    q3 = question_prepocess(question3,emb)
    q4 = question_prepocess(question4,emb)
    q5 = question_prepocess(question5,emb)


    v = temporal_attention_single([q1, q2, q3, q4, q5],[v]*5,use_conv=False)


    gru = Bidirectional(CuDNNGRU(cfg['dim_q'], return_sequences=True), merge_mode='sum')
    q1, q2, q3, q4, q5 = share_layer([q1, q2, q3, q4, q5],gru)
    v = temporal_attention_single([q1, q2, q3, q4, q5], v,use_conv=True)


    gru = Bidirectional(CuDNNGRU(cfg['dim_q'], return_sequences=True), merge_mode='sum')
    q1, q2, q3, q4, q5 = share_layer([q1, q2, q3, q4, q5], gru)
    v = temporal_attention_single([q1, q2, q3, q4, q5], v,use_conv=True)


    a1, a2, a3, a4, a5 = attr_block([q1, q2, q3, q4, q5], attrs)
    if cfg['attention']=='mlan':
        q_out = q1, q2, q3, q4, q5
        v_out = v
    else:
        q_out, v_out = time_spatial_attention_single([q1, q2, q3, q4, q5], v)


    # -------------------------  merge ------------------------

    ansq = concatenate(share_layer(q_out, Reshape((1, -1))), axis=1)
    ansq = SpatialDropout1D(0.3)(ansq)

    ansv = concatenate(share_layer(v_out, Reshape((1, -1))), axis=1)
    ansv = SpatialDropout1D(0.2)(ansv)

    ans_attr = concatenate(share_layer([a1, a2, a3, a4, a5], Reshape((1, -1))), axis=1)
    ans_attr = SpatialDropout1D(0.5)(ans_attr)

    if cfg['q_share']:
        q_share = [
            average([q_out[1], q_out[2], q_out[3], q_out[4]]),
            average([q_out[0], q_out[2], q_out[3], q_out[4]]),
            average([q_out[1], q_out[0], q_out[3], q_out[4]]),
            average([q_out[1], q_out[2], q_out[0], q_out[4]]),
            average([q_out[1], q_out[2], q_out[3], q_out[0]]),
        ]
        ans_q_share = concatenate(share_layer(q_share, Reshape((1, -1))), axis=1)
        ans_q_share = SpatialDropout1D(0.3)(ans_q_share)
        features = [ansq, ansv, ans_q_share, ans_attr]
    else:
        features = [ansq, ansv, ans_attr]
    merged = concatenate(features)


    merged = TimeDistributed(Dense(512,activation='relu'))(merged)
    merged = SpatialDropout1D(0.3)(merged)
    fc = TimeDistributed(Dense(num_ans,use_bias=True,activation='sigmoid'))
    output = fc(merged)

    prior_in = Input((5, num_ans), name='prior')
    if cfg['use_prior']:
        output = multiply([prior_in,output])

    #binary_crossentropy
    model = Model(inputs=[question1, question2, question3, question4, question5, video,attrs_in,prior_in], outputs=output)
    model.compile(loss=cfg['loss'], optimizer=Nadam(cfg['lr']), metrics=[acc])

    return model