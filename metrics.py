import tensorflow as tf

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Compute the Dice Coefficient.
    Parameters:
        y_true: Ground truth binary mask.
        y_pred: Predicted binary mask.
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Dice coefficient as a float.
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)  # Flatten and cast to float32
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)  # Flatten and cast to float32
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def jaccard_index(y_true, y_pred, smooth=1e-6):
    """
    Compute the Jaccard Index (Intersection over Union).
    Parameters:
        y_true: Ground truth binary mask.
        y_pred: Predicted binary mask.
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Jaccard index as a float.
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)  # Flatten and cast to float32
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)  # Flatten and cast to float32
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)