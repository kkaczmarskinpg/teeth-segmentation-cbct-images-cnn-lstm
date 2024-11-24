import tensorflow as tf

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])  
    y_pred_f = tf.reshape(y_pred, [-1])  
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

def jaccard_index(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])  
    y_pred_f = tf.reshape(y_pred, [-1])  
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)
