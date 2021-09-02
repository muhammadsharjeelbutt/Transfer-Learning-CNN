from keras.preprocessing.image import ImageDataGenerator

class DataPrep():
    
    __instance__ = None
    
    @staticmethod
    def Instance():
        return DataPrep.__instance__
    
    def __init__(self):
        if DataPrep.__instance__ is None:
            DataPrep.__instance__ = self
    
    def load_data(self,imageDim,trainpath,valpath):
        TRAINING_DIR = trainpath
        training_datagen = ImageDataGenerator( # training data
            rescale = 1./255,
            fill_mode='nearest'
            )
        train_generator = training_datagen.flow_from_directory(
            TRAINING_DIR,
            target_size=(imageDim,imageDim),
            class_mode='categorical'
            )
        VALIDATION_DIR = valpath
        validation_datagen = ImageDataGenerator( # validation data
            rescale = 1./255,
            fill_mode='nearest'
            )
        validation_generator = validation_datagen.flow_from_directory(
            VALIDATION_DIR,
            shuffle=False,
            target_size=(imageDim,imageDim),
            class_mode='categorical'
            )
        return train_generator,validation_generator