import tensorflow as tf
def kdfunc_CE(s, t, temperature=1, mask=None):
    if mask is not None:
        s = s *  tf.cast(mask, dtype=s.dtype)
    # s = tf.nn.softmax(tf.linalg.normalize(s, axis=1)[0] / 1, axis=1)
    t = tf.nn.softmax(tf.linalg.normalize(t, axis=1)[0] / temperature, axis=1)
    s = tf.stop_gradient(s)
    loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=s)
    loss = tf.reduce_mean(loss)
    return loss

def kdfunc_L2(s, t, temperature=1, mask=None):
    if mask is not None:
        s = s *  tf.cast(mask, dtype=s.dtype)
        # t = t * mask
        # s = mask_logits(s, mask=mask)
    s = tf.nn.softmax(tf.linalg.normalize(s, axis=1)[0] / temperature, axis=1)
    t = tf.nn.softmax(tf.linalg.normalize(t, axis=1)[0] / temperature, axis=1)
    t = tf.stop_gradient(t)
    loss = tf.nn.l2_loss(s - t)
    loss = tf.reduce_mean(loss)
    return loss
    # sloss = tf.nn.l2_loss(slogit_s - slogit_t)
    # eloss = tf.nn.l2_loss(elogit_s - elogit_t)
    # sloss = tf.compat.v1.distributions.kl_divergence(slogit_s, slogit_t)
    # eloss = tf.compat.v1.distributions.kl_divergence(elogit_s, elogit_t)