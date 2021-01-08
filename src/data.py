import numpy as np
import tensorflow as tf


def _to_numpy(batch):
    # use this over tf.data.Data.as_numpy_iterator() for 0 copy
    return {'images': batch[0]._numpy(), 'labels': batch[1]._numpy()}


def as_numpy(ds):
    return map(_to_numpy, ds)


def get_data(seed, train_bs, val_bs, image_generator, subset=False, drop_last=True):
    train, val = tf.keras.datasets.fashion_mnist.load_data()
    (train_images, train_labels), (val_images, val_labels) = (train, val)
    train_images = train_images[..., None] / 255. - 0.5
    val_images = val_images[..., None] / 255 - 0.5
    train_labels = train_labels.astype('int32')
    val_labels = val_labels.astype('int32')

    if subset:
        train_images, train_labels = train_images[:16], train_labels[:16]
        val_images, val_labels = val_images[:16], val_labels[:16]

    assert len(val_images) % val_bs == 0  # for efficiency
    assert len(train_images) % train_bs == 0  # for efficiency

    gen_bs = 100 if not subset else 2
    train_ds = tf.data.Dataset.from_generator(
        lambda: image_generator.flow(
            x=train_images, y=train_labels, batch_size=gen_bs, shuffle=True, seed=seed),
        output_types=(tf.float32, tf.float32),
        output_shapes=([gen_bs, 28, 28, 1], [gen_bs])
        # ).unbatch(  # unbatch so we can drop remainder
        # ).batch(train_bs, drop_remainder=True
    ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (val_images, val_labels)).batch(val_bs).prefetch(buffer_size=6)

    return train_ds, val_ds


def get_random_eraser(p=0.5,
                      s_l=0.02,
                      s_h=0.4,
                      r_1=0.3,
                      r_2=1 / 0.3,
                      v_l=0,
                      v_h=255,
                      pixel_level=False):
    """Source: https://github.com/yu4u/cutout-random-erasing/blob/master/random_eraser.py"""
    def eraser(input_img):
        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape

        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            if input_img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h, w))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w] = c

        return input_img

    return eraser
