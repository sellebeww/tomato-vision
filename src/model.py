"""Two CNNs initialized from scratch; no downloaded/pretrained weights."""
from src.config import ExperimentConfig, IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE

def build_model(input_shape=None, num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE, config=None):
    import tensorflow as tf
    from src.preprocessing import make_augmentation
    c=config or ExperimentConfig(learning_rate=learning_rate)
    L=tf.keras.layers
    inputs=L.Input(shape=input_shape or (c.image_size,c.image_size,3),name="rgb_0_to_1")
    x=inputs
    if c.architecture=="baseline":
        # Original project's four-block CNN topology, retrained on the fixed split.
        for i,filters in enumerate((32,64,128,256)):
            x=L.Conv2D(filters,3,padding="same",name=f"block{i}_conv")(x)
            x=L.BatchNormalization()(x)
            x=L.Activation("relu")(x)
            x=L.MaxPooling2D()(x)
        x=L.GlobalAveragePooling2D()(x)
        x=L.Dense(128,activation="relu")(x)
        x=L.Dropout(.5)(x)
    else:
        x=make_augmentation(c.seed)(x)
        x=L.ReLU(max_value=1)(x)
        regularizer=tf.keras.regularizers.l2(c.weight_decay)
        x=L.Conv2D(24,3,strides=2,padding="same",use_bias=False,kernel_regularizer=regularizer)(x)
        x=L.GroupNormalization(groups=8)(x)
        x=L.Activation("relu")(x)
        for filters in (48,96,160):
            residual=L.Conv2D(filters,1,strides=2,padding="same",use_bias=False)(x)
            x=L.SeparableConv2D(filters,3,padding="same",use_bias=False,
                                 depthwise_regularizer=regularizer,pointwise_regularizer=regularizer)(x)
            x=L.GroupNormalization(groups=8)(x)
            x=L.Activation("relu")(x)
            x=L.SeparableConv2D(filters,3,strides=2,padding="same",use_bias=False,
                                 depthwise_regularizer=regularizer,pointwise_regularizer=regularizer)(x)
            x=L.GroupNormalization(groups=8)(x)
            x=L.Add()([x,residual])
            x=L.Activation("relu")(x)
            x=L.SpatialDropout2D(c.dropout/3)(x)
        x=L.GlobalAveragePooling2D()(x)
        x=L.Dense(64,activation="relu",kernel_regularizer=regularizer)(x)
        x=L.Dropout(c.dropout)(x)
    outputs=L.Dense(num_classes,activation="softmax",name="class_probabilities")(x)
    model=tf.keras.Model(inputs,outputs,name=f"tomato_{c.architecture}_scratch")
    model.compile(optimizer=tf.keras.optimizers.Adam(c.learning_rate),
                  loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model
