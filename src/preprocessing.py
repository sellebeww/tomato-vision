"""Identical EXIF-aware RGB/letterbox loading for training and inference."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from src.config import IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED, resolve_path

Image.MAX_IMAGE_PIXELS = 25_000_000

def load_image(value, size=IMAGE_SIZE[0]):
    """Return float32 HWC [0,1]. Accept path or binary file-like object."""
    source = resolve_path(value) if isinstance(value, (str, Path)) else value
    with Image.open(source) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Animated/multi-frame images are not supported")
        if image.width * image.height > Image.MAX_IMAGE_PIXELS:
            raise ValueError("Image exceeds 25 megapixels")
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.pad(image, (size, size), method=Image.Resampling.BILINEAR,
                             color=(127, 127, 127), centering=(0.5, 0.5))
        return np.asarray(image, dtype=np.float32) / 255.0

def load_arrays(df, size=IMAGE_SIZE[0]):
    if df.empty:
        raise ValueError("Cannot load an empty dataset")
    return (np.stack([load_image(p, size) for p in df.filepath]),
            df.label_idx.to_numpy(dtype=np.int32))

def make_augmentation(seed=RANDOM_SEED):
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(0.06, fill_mode="reflect", seed=seed + 1),
        tf.keras.layers.RandomTranslation(0.07, 0.07, fill_mode="reflect", seed=seed + 2),
        tf.keras.layers.RandomZoom((-0.12, 0.08), seed=seed + 3),
        tf.keras.layers.RandomContrast(0.12, seed=seed + 4),
        tf.keras.layers.RandomBrightness(0.08, value_range=(0, 1), seed=seed + 5),
    ], name="training_augmentation")

def array_dataset(images, labels, batch_size=BATCH_SIZE):
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((images, labels)).batch(batch_size)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return ds.with_options(options).prefetch(1)

def dataframe_to_dataset(df, batch_size=BATCH_SIZE, augment=False, shuffle=False,
                         seed=RANDOM_SEED, size=IMAGE_SIZE[0]):
    import tensorflow as tf
    images, labels = load_arrays(df, size)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if augment:
        aug = make_augmentation(seed)
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0., 1.), y))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return ds.with_options(options).prefetch(1)

def balanced_indices(labels, samples_per_class, rng):
    """Exactly balanced draws. Repeated views are not new source photographs."""
    ids = []
    for label in range(3):
        pool = np.flatnonzero(labels == label)
        if not len(pool):
            raise ValueError(f"Training class {label} missing")
        for offset in range(0, samples_per_class, len(pool)):
            ids.extend(rng.permutation(pool)[:samples_per_class-offset])
    rng.shuffle(ids)
    return np.asarray(ids)

def build_datasets(train_df, val_df, test_df, batch_size=BATCH_SIZE, seed=RANDOM_SEED):
    return (dataframe_to_dataset(train_df, batch_size, True, True, seed),
            dataframe_to_dataset(val_df, batch_size), dataframe_to_dataset(test_df, batch_size))
