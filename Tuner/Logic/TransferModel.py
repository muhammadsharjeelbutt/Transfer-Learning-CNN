from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16 # import more model as per use
from keras.models import Sequential
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class TransferModel:
    __instance__ = None

    @staticmethod
    def Instance():
        return TransferModel.__instance__

    def __init__(self):
        if TransferModel.__instance__ is None:
            TransferModel.__instance__ = self

    def create_model(self,model="vgg",imageDim=224,lr=[],opt="",act="softmax",summary=False):
        if model=="vgg":
            architecture = VGG16(input_shape=(imageDim,imageDim,3), weights='imagenet', include_top=False)
        for layer in architecture.layers:
            layer.trainable = False
        x = Flatten()(architecture.output)
        if act=="sigmoid":
            prediction = Dense(33,activation="sigmoid")(x)
        prediction = Dense(33,activation="softmax")(x)
        model = Model(inputs=architecture.input, outputs=prediction)
        if bool(lr):
            if opt=="Adam":
                opt = keras.optimizers.Adam(learning_rate=lr)
                model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
            if opt=="SGD":
                opt = keras.optimizers.SGD(learning_rate=lr)
                model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
            if opt=="RMSprop":
                opt = keras.optimizers.RMSprop(learning_rate=lr)
                model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
        if opt=="Adam":
            model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
        if opt=="SGD":
            model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])
        if opt=="RMSprop":
            model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
        if opt=="":
            model.compile(loss='categorical_crossentropy',metrics=['accuracy'])
        if summary==True:
            model.summary()
        return model
    
    def run_model(self,model,traingen,valgen,epochs_,repeat):
        model_list =[]
        for _ in range(0,repeat):
            model = self.create_model()
            history = model.fit(traingen,validation_data=valgen,epochs=epochs_,verbose=1)
            model_list.append(history)
            valgen.reset()
        return model_list

    def get_mean(self, data,repeat):
        avg_list_val = []
        for j in range(0,repeat):
            H = data[j]
            av = np.array(H.history['val_acc']) * 100
            avg_list_val.append(av)
        val = np.mean(avg_list_val,axis=0)
        return val

    def get_mean_loss(self, data,repeat):
        avg_list_val = []
        for j in range(0,repeat):
            H = data[j]
            av = np.array(H.history['val_loss']) * 100
            avg_list_val.append(av)
        val = np.mean(avg_list_val,axis=0)
        return val