import tensorflow as tf

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    
    y_true_f = tf.flatten(y_true)
    y_pred_f = tf.flatten(y_pred)
    intersection = tf.dot(y_true, tf.transpose(y_pred))
    union = tf.dot(y_true,tf.transpose(y_true))+tf.dot(y_pred,tf.transpose(y_pred))
    return (2. * intersection + smooth) / (union + smooth)

def jaccard_index(y_true, y_pred, smooth=1e-6):

    y_true_f = tf.flatten(y_true)
    y_pred_f = tf.flatten(y_pred)
    intersection = tf.dot(y_true, tf.transpose(y_pred))
    union = tf.dot(y_true,tf.transpose(y_true))+tf.dot(y_pred,tf.transpose(y_pred))
    return (intersection + smooth) / (union + smooth)

