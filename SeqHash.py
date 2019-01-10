# -*- coding: utf-8 -*-
"""
@author: Qian Shi
"""
import os
os.environ['KERAS_BACKEND']='tensorflow'

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input,Conv1D, Lambda,MaxPooling1D,Dense, Dropout, BatchNormalization,GlobalAveragePooling1D,AveragePooling1D
from keras.optimizers import RMSprop,SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
# Check whether GPU is being or not
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

hash_bits = 48
seq_len = 100
epochs = 5
lr = 0.01
    
def main():
    input_shape = (seq_len,1)
    
    # network definition
    feat_model = createFeatModel(seq_len,hash_bits)
    
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    # because we re-use the same instance 'feat_model',
    # the weights of the network
    # will be shared across the two branches
    code_a = feat_model(input_a)
    code_b = feat_model(input_b)
    
    hash_code_dist = Lambda(hamming_distance,output_shape=hamm_dist_output_shape)([code_a, code_b])
    model_siamese_like = Model([input_a, input_b], hash_code_dist)
    
    #data split
    data_pairs = np.load("data/data_pairs.npy")
    edit_dist_pairs = np.load("data/dist_pairs.npy")
    
    pair_num = np.shape(data_pairs)[0]
    edit_dist_pairs = edit_dist_pairs.reshape((pair_num,1))
    data_pairs = data_pairs.reshape(data_pairs.shape + (1,))
    
    edit_dist_norm = edit_dist_pairs*hash_bits/seq_len #normalize edit distance
    #print (edit_dist_norm)
    
    
    X_train, X_test, y_train, y_test = train_test_split(data_pairs, edit_dist_norm, test_size=0.3, random_state=42)
    #X_train=X_train.reshape(X_train.shape + (1,))
    #X_test=X_test.reshape(X_test.shape + (1,))
    #X_train_shape = np.shape(X_train)
    #X_test_shape = np.shape(X_test)
    
    #print("Training size: ",X_train_shape, "\nTesting size: ",X_test_shape)
    # train
    rms = RMSprop()
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9)
    #model_siamese_like.compile(loss='binary_crossentropy', optimizer=sgd)
    model_siamese_like.compile(loss="mean_absolute_error", optimizer=sgd)
    
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
        
    callbacks = [LearningRateScheduler(schedule=Schedule(epochs)),
                 ModelCheckpoint("checkpoints/weights.{epoch:02d}-{val_loss:.4f}.hdf5",monitor="val_loss",verbose=1,save_best_only=True,mode="auto")]
    
    
    model_siamese_like.fit([X_train[:, 0], X_train[:, 1]], y_train,batch_size=16,epochs=epochs,callbacks=callbacks,validation_data=([X_test[:, 0], X_test[:, 1]], y_test))
    
    
    y_pred_dist = model_siamese_like.predict([data_pairs[:, 0], data_pairs[:, 1]])
    #print (y_pred_dist)
    np.savetxt('data/y_pred_dist.csv',y_pred_dist,fmt='%f', delimiter=",")
    np.savetxt('data/edit_dist_pairs.csv',edit_dist_pairs,fmt='%i', delimiter=",")
    np.savetxt('data/edit_dist_norm.csv',edit_dist_norm,fmt='%f', delimiter=",")
    
    plt.hist(edit_dist_norm, bins = 20,alpha=0.5, label='Norm Edit Dist') 
    plt.hist(y_pred_dist, bins = 20,alpha=0.5, label='Hamming Dist') 
    plt.legend(loc='upper right')
    plt.title("Histogram of Distances") 
    plt.show()
    
    uni_seq = np.unique(data_pairs[:, 0], axis=0)
    y_pred_code = feat_model.predict(uni_seq)
    np.savetxt('data/y_pred_code.csv',y_pred_code,fmt='%i', delimiter=",")

    
def createFeatModel(seq_len,hash_bits):
    print("\n--- Create 1D neural network model ---\n")
    # 1D CNN neural network
    model = Sequential()
    model.add(Conv1D(200, 5, activation='relu', input_shape=(seq_len,1) ))
    model.add(Conv1D(200, 5, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(3))
    model.add(Conv1D(100, 3, activation='relu'))
    model.add(Conv1D(100, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(hash_bits, activation='tanh'))
    hash_code = Lambda(lambda x: (K.sign(x)+1)/2)(model.output)
    feat_model = Model(model.input,hash_code)
    print(feat_model.summary())
    return feat_model

def hamming_distance(input_codes):
    code_a, code_b = input_codes
    print("hamming_distance: ", np.shape(code_a),np.shape(code_b))
    #Calculate the Hamming distance between two hash codes
    sum_square = K.sum(K.square(code_a - code_b), axis=1, keepdims=True)*1.0
    #return sum_square*1.0/hash_bits #normalization
    return sum_square

def hamm_dist_output_shape(input_shapes):
    shape_a, shape_b = input_shapes
    print("hamm_dist_output_shape: ",shape_a)
    return (shape_a[0], 1)

#y_true: edit distance
#y_pred: hamming distance
def consistency_loss(y_true,y_pred):
    return K.mean(K.square(y_true*hash_bits/seq_len-y_pred))
    
class Schedule:
    def __init__(self, epochs):
        self.epochs = epochs

    def __call__(self, epoch_idx):
        #print (epoch_idx)
        if epoch_idx < self.epochs * 0.25:
            return lr
        elif epoch_idx < self.epochs * 0.5:
            return 0.5*lr
        elif epoch_idx < self.epochs * 0.75:
            return 0.25*lr
        return 0.1*lr
    
if __name__ == '__main__':
    main()