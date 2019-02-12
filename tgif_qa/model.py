from keras.layers import *
from keras.models import Model
from config import *
from keras.optimizers import *
from keras.regularizers import *


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


def acc(y_true, y_pred):
    y_pred = (y_pred - K.min(y_pred,axis=2,keepdims=True)/
              (K.max(y_pred,axis=2,keepdims=True)-K.min(y_pred,axis=2,keepdims=True)))
    score = K.sum(y_true*y_pred)/tf.cast(tf.shape(y_pred)[0],tf.float32)
    return score / 5

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

    def get_atten_w(q_list, v,fca):
        w_list = []
        for q in q_list:
            merged = distance(q,v,'dice')
            w = fca(merged)
            w = Flatten()(w)
            w = Activation('softmax')(w)
            w_list.append(w)
        w = average(w_list)
        w = Reshape((-1, 1, 1))(w)
        return w

    fcq = Dense(cfg['dim_a'],activation='relu')
    fca = Dense(1, use_bias=False)

    q = share_layer(q,GlobalAveragePooling1D())
    q = share_layer(q, fcq)
    q = share_layer(q, Reshape((1, -1)))

    v_encode = TimeDistributed(GlobalAveragePooling1D())(v)  # 16 *2048
    v_encode = Dense(cfg['dim_a'], activation='tanh')(v_encode)

    w = get_atten_w(q, v_encode, fca)


    if use_conv:

        v = Lambda(lambda x: x[0] * x[1])([v, w])
        v = Conv2D(filters=cfg['dim_v'], kernel_size=[1,1], padding='same')(v)
        v = BatchNormalization()(v)
        v = Activation('relu')(v)

    else:
        v = Lambda(lambda x: x[0] * x[1])([v, w])
    return v

def time_spatial_attention(q,v):

    fcaq = Dense(1)
    fcav = Dense(1)
    fcq = Conv1D(filters=cfg['dim_a'],kernel_size=1,activation='relu')
    v = Reshape((-1, cfg['dim_v']))(v)
    v_encode = Dense(units=cfg['dim_a'], activation='tanh')(v)
    v_encode = SpatialDropout1D(0.3)(v_encode)

    q_out = []
    v_out = []

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


        # v_i = Lambda(top_k_ave, arguments={'k': cfg['atten_k']})([atten_v, v])
        atten_v = Activation('softmax')(atten_v)
        atten_v = Reshape((1, -1))(atten_v)
        v_i = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten_v, v])
        v_i = Lambda(lambda x: K.squeeze(x, axis=1))(v_i)
        v_out.append(v_i)

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

    q_encode = share_layer(q,fcq)
    q_encode = share_layer(q_encode,SpatialDropout1D(0.3))

    q_out = []
    v_out = []

    for i in range(len(q)):
        atten = Lambda(lambda x: K.batch_dot(x[0], x[1]))([q_encode[i], Permute((2, 1))(v_encode[i])])  # 14 * 40


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


        # v_i = Lambda(top_k_ave, arguments={'k': cfg['atten_k']})([atten_v, v])
        atten_v = Activation('softmax')(atten_v)
        atten_v = Reshape((1, -1))(atten_v)
        v_i = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten_v, v[i]])
        v_i = Lambda(lambda x: K.squeeze(x, axis=1))(v_i)
        v_out.append(v_i)

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

    def get_atten_w(q,fcq,fca,attr,attr_encode):
        q_ave = GlobalAveragePooling1D()(q)
        q_ave = fcq(q_ave)
        q_ave = Reshape((-1, cfg['dim_a']))(q_ave)  # 1  512

        merged = distance(q_ave,attr_encode,'dice')
        w = fca(merged)
        w = Flatten()(w)
        # attr = Lambda(top_k_ave,arguments={'k':32})([w,attr])

        w = Activation('softmax')(w)
        w = Reshape((1,-1))(w)
        attr = Lambda(lambda x:K.batch_dot(x[0],x[1]))([w,attr])
        attr = Lambda(lambda x:K.squeeze(x,axis=1))(attr)
        return attr

    def softmax_mask(x_in,axis=1):
        x = x_in[0]
        mask = x_in[1]
        x = K.exp(x - K.max(x, axis=axis, keepdims=True))*mask
        return x/K.sum(x, axis=axis, keepdims=True)


    attr = Conv2D(300, kernel_size=[1, ATTR_LEN], activation='relu')(attr)
    attr = Lambda(lambda x: K.squeeze(x, axis=2))(attr)


    attr_encode = Conv1D(filters=cfg['dim_a'], activation='tanh', kernel_size=1)(attr)
    fcq = Dense(cfg['dim_a'],activation='relu')
    fca = Dense(1, use_bias=False)

    attr_result = []
    for q_i in q_hist:
        attr_result.append(get_atten_w(q_i,fcq, fca,attr,attr_encode))

    return attr_result


def attention_model(embed_mat,cfg):

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


    questions = []
    for i in range(cfg['num_q']):
        questions.append(Input(shape=(TEXT_LEN,), name='q'+str(i)))

    video = Input(shape=(num_frame,num_o,frame_size), name='video')
    attrs_in = Input(shape=(attr_num,ATTR_LEN),name='attr')


    v = video
    v = BatchNormalization()(v)

    emb = Embedding(
        embed_mat.shape[0],
        embed_mat.shape[1],
        weights=[embed_mat],
        trainable=False,
        name='embedd'
    )
    attrs = TimeDistributed(emb)(attrs_in)
    attrs = BatchNormalization()(attrs)

    q = []
    for q_i in questions:
        q.append(question_prepocess(q_i,emb))
    v = temporal_attention(q, v,use_conv=False)


    gru = Bidirectional(CuDNNGRU(cfg['dim_q'], return_sequences=True), merge_mode='sum')
    q = share_layer(q,gru)
    v = temporal_attention(q, v,use_conv=True)

    gru = Bidirectional(CuDNNGRU(cfg['dim_q'], return_sequences=True), merge_mode='sum')
    q = share_layer(q, gru)
    v = temporal_attention(q, v,use_conv=True)


    attr_out = attr_block(q, attrs)
    q_out, v_out = time_spatial_attention(q, v)


    # -------------------------  merge ------------------------

    ansq = concatenate(share_layer(q_out,Reshape((1, -1))),axis=1)
    ansq = SpatialDropout1D(0.3)(ansq)

    ansv = concatenate(share_layer(v_out, Reshape((1, -1))),axis=1)
    ansv = SpatialDropout1D(0.2)(ansv)

    ans_attr = concatenate(share_layer(attr_out,Reshape((1, -1))),axis=1)
    ans_attr = SpatialDropout1D(0.5)(ans_attr)

    if cfg['q_share']:
        q_share = [average([q_out[i] for i in range(cfg['num_q']) if i != j]) for j in range(cfg['num_q'])]
        ans_q_share = concatenate(share_layer(q_share, Reshape((1, -1))), axis=1)
        ans_q_share = SpatialDropout1D(0.3)(ans_q_share)
        merged = concatenate([ansq, ansv, ans_attr, ans_q_share])
    else:
        merged = concatenate([ansq, ansv, ans_attr])


    prior_in = Input((cfg['num_q'],num_ans),name='prior')


    merged = TimeDistributed(Dense(512,activation='relu'))(merged)
    merged = SpatialDropout1D(0.3)(merged)
    fc = TimeDistributed(Dense(num_ans,use_bias=True))
    output = fc(merged)
    def softmax_mask(x_in):
        x = x_in[0]
        mask = x_in[1]
        x = K.exp(x-K.max(x,axis=2,keepdims=True))*mask
        return x/K.sum(x,axis=2,keepdims=True)
    output = Lambda(softmax_mask)([output,prior_in])


    #binary_crossentropy
    model = Model(inputs=questions+[video,attrs_in,prior_in], outputs=output)
    model.compile(loss=cfg['loss'], optimizer=Nadam(cfg['lr']), metrics=[acc])

    # model.summary()
    return model




def separate_model(embed_mat, cfg):
    def question_prepocess(q, emb):
        q = emb(q)
        q = BatchNormalization()(q)
        q = SpatialDropout1D(0.2)(q)  # 涨
        return q

    num_frame = cfg['num_frame']
    frame_size = 2048
    num_ans = cfg['num_ans']
    num_o = 36
    attr_num = cfg['attr_num']

    questions = []
    for i in range(cfg['num_q']):
        questions.append(Input(shape=(TEXT_LEN,), name='q' + str(i)))

    video = Input(shape=(num_frame, num_o, frame_size), name='video')
    attrs_in = Input(shape=(attr_num, ATTR_LEN), name='attr')


    v = video
    v = BatchNormalization()(v)

    emb = Embedding(
        embed_mat.shape[0],
        embed_mat.shape[1],
        weights=[embed_mat],
        trainable=False,
        name='embedd'
    )
    attrs = TimeDistributed(emb)(attrs_in)
    attrs = BatchNormalization()(attrs)

    q = []
    for q_i in questions:
        q.append(question_prepocess(q_i, emb))

    v = temporal_attention_single(q,[v]*cfg['num_q'],use_conv=False)


    gru = Bidirectional(CuDNNGRU(cfg['dim_q'], return_sequences=True), merge_mode='sum')
    q = share_layer(q,gru)
    v = temporal_attention_single(q, v,use_conv=True)


    gru = Bidirectional(CuDNNGRU(cfg['dim_q'], return_sequences=True), merge_mode='sum')
    q = share_layer(q, gru)
    v = temporal_attention_single(q, v,use_conv=True)


    attr_out = attr_block(q, attrs)
    q_out, v_out = time_spatial_attention_single(q, v)

    # -------------------------  merge ------------------------

    ansq = concatenate(share_layer(q_out, Reshape((1, -1))), axis=1)
    ansq = SpatialDropout1D(0.3)(ansq)

    ansv = concatenate(share_layer(v_out, Reshape((1, -1))), axis=1)
    ansv = SpatialDropout1D(0.2)(ansv)

    ans_attr = concatenate(share_layer(attr_out, Reshape((1, -1))), axis=1)
    ans_attr = SpatialDropout1D(0.5)(ans_attr)

    if cfg['q_share']:
        q_share = [average([q_out[i] for i in range(cfg['num_q']) if i != j]) for j in range(cfg['num_q'])]
        ans_q_share = concatenate(share_layer(q_share, Reshape((1, -1))), axis=1)
        ans_q_share = SpatialDropout1D(0.3)(ans_q_share)
        merged = concatenate([ansq, ansv, ans_attr, ans_q_share])
    else:
        merged = concatenate([ansq, ansv, ans_attr])

    prior_in = Input((cfg['num_q'], num_ans), name='prior')

    merged = TimeDistributed(Dense(512, activation='relu'))(merged)
    merged = SpatialDropout1D(0.3)(merged)
    fc = TimeDistributed(Dense(num_ans, use_bias=True))
    output = fc(merged)

    def softmax_mask(x_in):
        x = x_in[0]
        mask = x_in[1]
        x = K.exp(x - K.max(x, axis=2, keepdims=True)) * mask
        return x/K.sum(x, axis=2, keepdims=True)

    output = Lambda(softmax_mask)([output, prior_in])

    # binary_crossentropy
    model = Model(inputs=questions + [video, attrs_in, prior_in], outputs=output)
    model.compile(loss=cfg['loss'], optimizer=Nadam(cfg['lr']), metrics=[acc])

    # model.summary()
    return model








































