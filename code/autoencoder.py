import pandas as pd
import h5py
from sklearn.model_selection import train_test_split

file_path = './.xlsx'
df = pd.read_excel(file_path)
train_df, test_df = train_test_split(df, test_size=0.3, stratify=df["label"])

print('train and test: ', df['label'].value_counts())

print(len(train_df))
print('train: ', train_df['label'].value_counts())

print(len(test_df))
print('test: ', test_df['label'].value_counts())

import os

labels = train_df['label']
ADs = train_df['AD']
print(labels.value_counts())

data_folder = './data/data{}_resample_filter/'
selected_filenames = []

for label, ad in zip(labels, ADs):
    if label == 0:
        data_path = data_folder.format(0)
    elif label == 1:
        data_path = data_folder.format(1)
    else:
        continue

    files = os.listdir(data_path)
    for filename in files:
        if filename.startswith(str(ad)):
            selected_filenames.append(os.path.join(data_path, filename))

print(len(selected_filenames))

org_filenames = []
data_folder = './data/data{}_resample/'
for label, ad in zip(labels, ADs):
    if label == 0:
        data_path = data_folder.format(0)
    elif label == 1:
        data_path = data_folder.format(1)
    else:
        continue

    files = os.listdir(data_path)
    for filename in files:
        if filename.startswith(str(ad)):
            org_filenames.append(os.path.join(data_path, filename))

print(len(org_filenames))

type = ['log-sigma-1-0mm-3D', 'log-sigma-2-0mm-3D', 'log-sigma-3-0mm-3D', 'log-sigma-4-0mm-3D', 'log-sigma-5-0mm-3D',
        'wavelet-HHH', 'wavelet-LHH', 'wavelet-LLH', 'wavelet-LHL', 'wavelet-HLL', 'wavelet-LLL', 'wavelet-HLH', 'wavelet-HHL',
        'exponential', 'square', 'gradient', 'squareroot', 'logarithm']

result = {}
for t in type:
    if t == 'square':
        result[t] = []
        for f in selected_filenames:
            if 'square.' in f:
                result[t].append(f)
    elif t == 'squareroot':
        result[t] = []
        for f in selected_filenames:
            if 'squareroot.' in f:
                result[t].append(f)
    else:
        result[t] = [f for f in selected_filenames if t in f]

org_result = {'original': org_filenames}

result.update(org_result)

print(len(result))

filenames_0 = []
filenames_1 = []

for filename in selected_filenames:
    if 'data0_resample_filter' in filename:
        filenames_0.append(filename)
    elif 'data1_resample_filter' in filename:
        filenames_1.append(filename)

print('label 0: ', len(filenames_0) / 18)
print('label 1: ', len(filenames_1) / 18)

import numpy as np
import random
import nibabel as nib
from skimage.transform import resize

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0],True)
    logical_devices = tf.config.list_logical_devices("GPU")

from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

batch_size = 6
epochs = 100

def data_generator(files, batch_size):
    while True:
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i+batch_size]
            batch_images = []
            for f in batch_files:
                img = nib.load(f).get_fdata()
                img = img - np.mean(img)
                img = img / np.std(img)
                img = resize(img, (128, 128, 64))
                batch_images.append(img)
            batch_images = np.array(batch_images)
            batch_images = np.expand_dims(batch_images, axis=-1)
            yield batch_images, batch_images

def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def build_modified_autoencoder():
    input_img = tf.keras.Input(shape=(None, None, None, 1))

    x = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(input_img)
    x = Dropout(0.5)(x)
    x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling3D((4, 4, 4), padding='same')(x)

    x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling3D((4, 4, 4))(x)
    x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(x)
    decoded = Conv3D(1, (3, 3, 3), activation='tanh', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.0002), loss='mse', metrics=[ssim_metric])
    return autoencoder


from sklearn.model_selection import KFold

k = 4
best_val_loss = float("inf")
best_model = None

for key in result.keys():
    print(f"Training for key: {key}")

    all_files = result[key]
    random.shuffle(all_files)
    kf = KFold(n_splits=k)
    fold = 0
    histories = []
    best_history = {}

    train_history_plot_path = train_history_plot_path_template.format(batch_size, epochs, key) #average
    history_path = history_path_template.format(key)#average

    for train_files_idx, val_files_idx in kf.split(all_files):
        fold += 1
        checkpoint_path = checkpoint_path_template.format(batch_size, epochs, key, fold)

        train_files = [all_files[idx] for idx in train_files_idx]
        val_files = [all_files[idx] for idx in val_files_idx]

        autoencoder = build_modified_autoencoder()
        train_generator = data_generator(train_files, batch_size)
        val_generator = data_generator(val_files, batch_size)
        steps_per_epoch = len(train_files) // batch_size
        validation_steps = len(val_files) // batch_size

        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)

        history = autoencoder.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=[checkpoint, earlystop]
        )

        histories.append(history.history)

        # Check if the current fold has a better validation loss
        current_val_loss = min(history.history['val_loss'])
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model = autoencoder
            best_history = history.history


    # Save the best model for the key
    best_history_path = best_history_path_template.format(key)    #best
    best_model_path = best_model_path_template.format(batch_size, epochs, key)
    best_model.save(best_model_path)
    with h5py.File(best_history_path, 'w') as hf:
        for key_his_best, value in best_history.items():
            try:
                hf.create_dataset(key_his_best, data=value, compression=True)
            except:
                print('problematic:', key_his_best, value)
    # Compute average history
    avg_history = {}

    for metric in histories[0].keys():
        metric_values = []

        for history in histories:
            metric_values.append(history[metric])

        avg_history[metric] = [sum(values) / len(values) for values in zip(*metric_values)]


    with h5py.File(history_path_template.format(key), 'w') as hf:
        for key_his, value in avg_history.items():
            try:
                hf.create_dataset(key_his, data=value, compression=True)
            except:
                print('problematic:', key_his, value)


