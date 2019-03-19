import tensorflow as tf
import numpy as np
import random

import mal as dataset
import constants as const

def get_model(num_users, num_items, mf_dim=8, layers=[64,32,16,8],
              reg_layers=[0,0,0,0],  reg_mf=0.01):
    if not len(layers) == len(reg_layers):
        raise ValueError("Layers and reg_layers must be of equal length.")

    num_layers = len(layers)

    user_input = tf.keras.layers.Input(shape=(1, ), dtype='int32',
                                       name='user_input')
    item_input = tf.keras.layers.Input(shape=(1, ), dtype='int32',
                                       name='item_input')

    embeddings_init = 'glorot_uniform'

    mf_embedding_user = tf.keras.layers.Embedding(
        input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user',
        embeddings_initializer=embeddings_init, input_length=1,
        embeddings_regularizer=tf.keras.regularizers.l2(reg_mf)
    )
    mf_embedding_item = tf.keras.layers.Embedding(
        input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item',
        embeddings_initializer=embeddings_init, input_length=1,
        embeddings_regularizer=tf.keras.regularizers.l2(reg_mf)
    )

    mlp_embedding_user = tf.keras.layers.Embedding(
        input_dim=num_users, output_dim=layers[0]//2,
        name="mlp_embedding_user", embeddings_initializer=embeddings_init,
        input_length=1,
        embeddings_regularizer=tf.keras.regularizers.l2(reg_layers[0])
    )
    mlp_embedding_item = tf.keras.layers.Embedding(
        input_dim=num_items, output_dim=layers[0]//2,
        name="mlp_embedding_item", embeddings_initializer=embeddings_init,
        input_length=1,
        embeddings_regularizer=tf.keras.regularizers.l2(reg_layers[0])
    )

    mf_user_latent = tf.keras.layers.Flatten()(mf_embedding_user(user_input))
    mf_item_latent = tf.keras.layers.Flatten()(mf_embedding_item(item_input))
    mf_vector = tf.keras.layers.multiply([mf_user_latent, mf_item_latent])

    mlp_user_latent = tf.keras.layers.Flatten()(
        mlp_embedding_user(user_input))
    mlp_item_latent = tf.keras.layers.Flatten()(
        mlp_embedding_item(item_input))
    mlp_vector = tf.keras.layers.concatenate(
        [mlp_user_latent, mlp_item_latent])

    for i in range(1, num_layers):  
        layer = tf.keras.layers.Dense(
            layers[i],
            kernel_regularizer=tf.keras.regularizers.l2(reg_layers[i]),
            activation='relu',
            name='layer_{}'.format(i),
        )
        mlp_vector = layer(mlp_vector)

    predict_vector = tf.keras.layers.concatenate([mf_vector, mlp_vector])

    prediction = tf.keras.layers.Dense(
        1, activation='sigmoid', kernel_initializer='lecun_uniform',
        name='prediction')(predict_vector)

    model = tf.keras.models.Model([user_input, item_input], prediction)

    model.summary()

    return model

def get_train_data():
    # convert data to 0/1 labels depending on whether or not items
    # have been rated by users
    df = dataset.read_processed_data()

    user_input, item_input, labels = [], [], []

    num_users = len(df[dataset.USER_COL].unique())
    num_items = len(df[dataset.ITEM_COL].unique())

    user_rated = (-1, [])
    for _, row in df.iterrows():
        curr_user = row[dataset.USER_COL]
        # add positive instance
        user_input.append(curr_user)
        item_input.append(row[dataset.ITEM_COL])
        labels.append(1)

        # check which items are rated by current user
        # takes advantage of fact that ratings are sorted by user
        # should only execute on transition from user to user+1
        if user_rated[0] != curr_user:
            user_rated = (
                curr_user,
                df.loc[df[dataset.USER_COL] == curr_user][dataset.ITEM_COL].values
            )

        # add negative instances
        for _ in range(const.NUM_NEGS):
            k = random.randint(0, num_items - 1)
            while k in user_rated[1]:
                k = random.randint(0, num_items - 1)
            user_input.append(curr_user)
            item_input.append(k)
            labels.append(0)
    
    return user_input, num_users, item_input, num_items, labels

if __name__ == '__main__':
    # build model
    print("Getting train data...")
    user_input, num_users, item_input, num_items, labels = get_train_data()

    print("Loading model...")
    model = get_model(num_users, num_items)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        loss='binary_crossentropy',
    )

    # train
    print("Starting training...")
    hist = model.fit([np.array(user_input), np.array(item_input)],
                     np.array(labels), batch_size=128, epochs=5)

    print("Saving model...")
    model.save(const.MODEL_SAVE_PATH)
