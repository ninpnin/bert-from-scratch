import tensorflow as tf

def self_attention(x, WQ, WK, WV):

    # x ∈ ℝ^100 x V
    # WQ ∈ ℝ^20 x V
    # WK ∈ ℝ^20 x V
    # V ∈ ℝ^20 x V

    Q = tf.tensordot(x, WQ, axes=[1,0])
    K = tf.tensordot(x, WK, axes=[1,0])
    V = tf.tensordot(x, WV, axes=[1,0])
    print("Q.shape", Q.shape)
    print("K.shape", K.shape)

    log_attention = tf.tensordot(Q, K, axes=[0,0])
    log_attention = log_attention / 8
    attention = tf.nn.softmax(log_attention, axis=0)
    print(attention)

    return tf.tensordot(attention, V, axes=[0,1])

def bert_layer(x, WQ, WK, WV, A):
    # TODO
    pass

def bert():

    model, weights

if __name__ == '__main__':

    words = 3
    dim = 7
    dim_att = 5

    x = tf.random.normal((words, dim))
    WQ = tf.random.normal((dim, dim_att))
    WK = tf.random.normal((dim, dim_att))
    WV = tf.random.normal((dim, dim_att))
    

    x1 = self_attention(x, WQ, WK, WV)
    print(x1)