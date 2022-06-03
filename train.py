import os
import DL_outlier_function as function
import DL_outlier_model
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import *

ref_dir = 'C:/Users/user/Desktop/Taeheon/KARI_AI_dataset' + '/Ref_patch'
sen_dir = 'C:/Users/user/Desktop/Taeheon/KARI_AI_dataset' + '/Sen_patch'
Label_dir ='C:/Users/user/Desktop/Taeheon/KARI_AI_dataset'
patch_size=256
patch_band=3

# Import reference image patch
ref_dataset = glob.glob(ref_dir+'/*.mat')	# the number of classes
ref_classes = os.listdir(ref_dir)
ref_classes.sort(key=len)
os.chdir(ref_dir)
Data_ref = np.zeros((len(ref_dataset),patch_size,patch_size,patch_band))
for i in range(len(ref_dataset)):
	file_name = ref_classes[i]
	temp = sio.loadmat(file_name)['Ref_patch_256']
	temp_data = temp.reshape(patch_size,patch_size,patch_band)
	Data_ref[i] = temp_data  # ch01

# Import sensed image patch
sen_dataset = glob.glob(sen_dir+'/*.mat')	# the number of classes
sen_classes = os.listdir(sen_dir)
sen_classes.sort(key=len)
os.chdir(sen_dir)
Data_sen = np.zeros((len(sen_dataset),patch_size,patch_size,patch_band))
for i in range(len(sen_dataset)):
	file_name = sen_classes[i]
	temp = sio.loadmat(file_name)['Sen_patch_256']
	temp_data = temp.reshape(patch_size,patch_size,patch_band)
	Data_sen[i] = temp_data  # ch01

# Import label data
os.chdir(Label_dir)
Data_Label = np.zeros(len(sen_dataset))
Label_classes = os.listdir(Label_dir)
temp = pd.read_csv(Label_classes[1], sep=',',header=None)
Data_Label=temp.values.reshape(len(sen_dataset),1)
Data_Label=np.where(Data_Label==3,1,Data_Label)
Data_Label=np.where(Data_Label==2,0,Data_Label)

# Data resize to 64*64
ref_resize,sen_resize=function.scale_resize(Data_ref,Data_sen,64)
# Optimize outliers and inliers in a 5:5 ratio
new_ref, new_sen, new_label=function.data_optimizer(ref_resize, sen_resize, Data_Label)

# Data augmentation
Data_ref_agu=function.augment(new_ref,10,64)
Data_sen_agu=function.augment(new_sen,10,64)
Data_label_agu=np.tile(new_label,(10,1))

# Split dataset(Train:Val:Test=6:2:2)
Ref_train, Ref_val, Sen_train, Sen_val, Label_train, Label_val = function.splitTrainTestSet(Data_ref_agu, Data_sen_agu, Data_label_agu, 0.4,randomState=345)
Ref_val, Ref_test, Sen_val, Sen_test, Label_val, Label_test = function.splitTrainTestSet(Ref_val, Sen_val, Label_val, 0.5,randomState=345)

# Confirm GPU spec
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

print("[INFO] building siameseSAE network...")
# Multi-GPU setting
multi_gpu=tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1"],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

# Learning rate initialization
initial_learning_rate=1e-03
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
	initial_learning_rate,
	decay_steps=250,
	decay_rate=0.95,
	staircase=True
)

# Build a deep learning model in the multi-GPU environment
with multi_gpu.scope():
	patch_size = 64
	imgA = Input(shape=(patch_size, patch_size, patch_band))
	imgB = Input(shape=(patch_size, patch_size, patch_band))
	featureExtractor = DL_outlier_model.Patchs_network((patch_size, patch_size, patch_band))
	featsA = featureExtractor(imgA)
	featsB = featureExtractor(imgB)
	L2_distance = Lambda(function.euclidean_distance)([featsA, featsB])
	outputs = Dense(1, activation="sigmoid")(L2_distance)
	model = Model(inputs=[imgA, imgB], outputs=outputs)
	model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy',
				  metrics=['accuracy'])  # compile the model

# Checkpoint setting
check_dir="C:/Users/user/Desktop/Taeheon/KARI_AI_dataset/model_output/Outlier_removal2/checkpoints/cp.ckpt"
cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=check_dir,save_weights_only=True,verbose=1)
print("[INFO] training model...")

# Model train
history = model.fit([Ref_train, Sen_train], Label_train,validation_data=([Ref_val, Sen_val], Label_val), batch_size=256,
epochs=100, verbose=1, shuffle=True)

print("[INFO] Estimating the model accuracy...")
_, acc = model.evaluate([Ref_test, Sen_test], Label_test)
print("Accuracy is = ", (acc * 100.0), "%")

print("[INFO] Estimating the model prediction...")
results=model.predict(([Ref_test, Sen_test]))

output_dir="C:/Users/user/Desktop/Taeheon/KARI_AI_dataset/model_output/Outlier_removal2"
val_loss=history.history["val_loss"]
train_loss=history.history["loss"]
val_acc=history.history["val_accuracy"]
train_acc=history.history["accuracy"]

# Save model loss & accuracy
function.save_model_loss(train_loss, val_loss, output_dir)
function.save_model_acc(train_acc, val_acc, output_dir)

# Confirm loss & accuracy in a plot
function.plot_model_acc(train_acc,val_acc)
function.plot_model_loss(train_loss,val_loss)

