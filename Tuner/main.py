import Logic
import os
import multiprocessing
import random as rn
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.keras import backend as K

class Process:
    t_path          = "/Users/sharjeel/Documents/FYP/FYP/Dataset complete/Tuning_Dataset/Train"
    v_path          = "/Users/sharjeel/Documents/FYP/FYP/Dataset complete/Tuning_Dataset/Test"
    fig_path        = "/Users/sharjeel/Downloads/ZCC_Server/Tuner/Logic/"
    imDim           = 224
    activation_1    = "softmax"
    activation_2    = "sigmoid"
    activation_f    = ""
    optimizer_1     = "Adam"
    optimizer_2     = "SGD"
    optimizer_3     = "RMSprop"
    optimizer_f     = ""
    lr_rates        = [5e-1, 1e0, 15e-1, 2e0, 25e-1]
    num_cores       = 1
    epoc            = 2


    def __init__(self):
        Logic.TransferModel()
        Logic.Visualize()
        Logic.DataPrep()

    def execute_activation_functions(self,mode, n_repeat, seed): 
        
        if type(seed)==int:
            seed_list = [seed]*n_repeat
        else:
            if (type(seed) in [list, tuple]) and (len(seed) >= n_repeat): 
                seed_list = seed
            else:
                raise ValueError('seed must be an integer or a list/tuple the lenght n_repeat')
            
        if mode=='gpu':
            num_GPU = 1
            num_CPU = 1
            gpu_name = tf.test.gpu_device_name()
            if (gpu_name != ''):
                gpu_message = gpu_name  
                print("Testing with GPU: {}".format(gpu_message))
            else:
                gpu_message = "ERROR <GPU NO AVAILABLE>"
                print("Testing with GPU: {}".format(gpu_message))
                return  
        else:    
            num_CPU = 1
            num_GPU = 0
            max_cores = multiprocessing.cpu_count()
            print("Testing with CPU: using {} core ({} availables)".format(self.num_cores, max_cores))

 
        for i in range(0,1):
            os.environ['PYTHONHASHSEED'] = '0'                      
            np.random.seed(seed_list[i])
            rn.seed(seed_list[i])
            tf.set_random_seed((seed_list[i]))

            session_conf = tf.ConfigProto(intra_op_parallelism_threads=self.num_cores,
                                        inter_op_parallelism_threads=self.num_cores,
                                        allow_soft_placement=True,
                                        log_device_placement=True,
                                        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})

            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            tf.compat.v1.keras.backend.get_session(sess)

            tset,vset = Logic.DataPrep().load_data(imageDim=224,trainpath=self.t_path,valpath=self.v_path)

            model = Logic.TransferModel().create_model(model="vgg",imageDim=224,lr=[],opt="",act=self.activation_1,summary=True)
            model_list = Logic.TransferModel().run_model(model=model,traingen=tset,valgen=vset,epochs_=self.epoc,repeat=n_repeat)
            v1 = Logic.TransferModel().get_mean(data=model_list,repeat=n_repeat)

            model = Logic.TransferModel().create_model(model="vgg",imageDim=224,lr=[],opt="",act=self.activation_2,summary=True)
            model_list = Logic.TransferModel().run_model(model=model,traingen=tset,valgen=vset,epochs_=self.epoc,repeat=n_repeat)
            v2 = Logic.TransferModel().get_mean(data=model_list,repeat=n_repeat)

            Logic.Visualize().plot_activationfunctions(l1=v1,l2=v2,label1=self.activation_1,label2=self.activation_2,path=self.fig_path)
            K.clear_session()

        return "ACTIVATION FUNCTIONS OVER"
    
    def execute_optimizers_functions(self,mode, n_repeat, seed): 

        
        if type(seed)==int:
            seed_list = [seed]*n_repeat
        else:
            if (type(seed) in [list, tuple]) and (len(seed) >= n_repeat): 
                seed_list = seed
            else:
                raise ValueError('seed must be an integer or a list/tuple the lenght n_repeat')
            
        if mode=='gpu':
            num_GPU = 1
            num_CPU = 1
            gpu_name = tf.test.gpu_device_name()
            if (gpu_name != ''):
                gpu_message = gpu_name  
                print("Testing with GPU: {}".format(gpu_message))
            else:
                gpu_message = "ERROR <GPU NO AVAILABLE>"
                print("Testing with GPU: {}".format(gpu_message))
                return  
        else:    
            num_CPU = 1
            num_GPU = 0
            max_cores = multiprocessing.cpu_count()
            print("Testing with CPU: using {} core ({} availables)".format(self.num_cores, max_cores))

 
        for i in range(0,1):
            os.environ['PYTHONHASHSEED'] = '0'                      
            np.random.seed(seed_list[i])
            rn.seed(seed_list[i])
            tf.set_random_seed((seed_list[i]))

            session_conf = tf.ConfigProto(intra_op_parallelism_threads=self.num_cores,
                                        inter_op_parallelism_threads=self.num_cores,
                                        allow_soft_placement=True,
                                        log_device_placement=True,
                                        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})

            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            tf.compat.v1.keras.backend.get_session(sess)

            tset,vset = Logic.DataPrep().load_data(imageDim=224,trainpath=self.t_path,valpath=self.v_path)

            model = Logic.TransferModel().create_model(model="vgg",imageDim=224,lr=[],opt=self.optimizer_1,act=self.activation_f,summary=True)
            model_list = Logic.TransferModel().run_model(model=model,traingen=tset,valgen=vset,epochs_=self.epoc,repeat=n_repeat)
            v1 = Logic.TransferModel().get_mean(data=model_list,repeat=n_repeat)

            model = Logic.TransferModel().create_model(model="vgg",imageDim=224,lr=[],opt=self.optimizer_2,act=self.activation_f,summary=True)
            model_list = Logic.TransferModel().run_model(model=model,traingen=tset,valgen=vset,epochs_=self.epoc,repeat=n_repeat)
            v2 = Logic.TransferModel().get_mean(data=model_list,repeat=n_repeat)

            model = Logic.TransferModel().create_model(model="vgg",imageDim=224,lr=[],opt=self.optimizer_3,act=self.activation_f,summary=True)
            model_list = Logic.TransferModel().run_model(model=model,traingen=tset,valgen=vset,epochs_=self.epoc,repeat=n_repeat)
            v3 = Logic.TransferModel().get_mean(data=model_list,repeat=n_repeat)

            Logic.Visualize().plot_optimizers(l1=v1,l2=v2,l3=v3,label1=self.optimizer_1,label2=self.optimizer_2,label3=self.optimizer_3,path=self.fig_path)
            K.clear_session()

        return "OPTIMIZERS OVER"
    
    def execute_learningrates_functions(self,mode, n_repeat, seed): 

        
        if type(seed)==int:
            seed_list = [seed]*n_repeat
        else:
            if (type(seed) in [list, tuple]) and (len(seed) >= n_repeat): 
                seed_list = seed
            else:
                raise ValueError('seed must be an integer or a list/tuple the lenght n_repeat')
            
        if mode=='gpu':
            num_GPU = 1
            num_CPU = 1
            gpu_name = tf.test.gpu_device_name()
            if (gpu_name != ''):
                gpu_message = gpu_name  
                print("Testing with GPU: {}".format(gpu_message))
            else:
                gpu_message = "ERROR <GPU NO AVAILABLE>"
                print("Testing with GPU: {}".format(gpu_message))
                return  
        else:    
            num_CPU = 1
            num_GPU = 0
            max_cores = multiprocessing.cpu_count()
            print("Testing with CPU: using {} core ({} availables)".format(self.num_cores, max_cores))

 
        for i in range(0,1):
            os.environ['PYTHONHASHSEED'] = '0'                      
            np.random.seed(seed_list[i])
            rn.seed(seed_list[i])
            tf.set_random_seed((seed_list[i]))

            session_conf = tf.ConfigProto(intra_op_parallelism_threads=self.num_cores,
                                        inter_op_parallelism_threads=self.num_cores,
                                        allow_soft_placement=True,
                                        log_device_placement=True,
                                        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})

            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            tf.compat.v1.keras.backend.get_session(sess)

            tset,vset = Logic.DataPrep().load_data(imageDim=224,trainpath=self.t_path,valpath=self.v_path)

            model = Logic.TransferModel().create_model(model="vgg",imageDim=224,lr=[0],opt=self.optimizer_f,act=self.activation_f,summary=True)
            model_list = Logic.TransferModel().run_model(model=model,traingen=tset,valgen=vset,epochs_=self.epoc,repeat=n_repeat)
            v1 = Logic.TransferModel().get_mean(data=model_list,repeat=n_repeat)
            v1_l = Logic.TransferModel().get_mean_loss(data=model_list,repeat=n_repeat)
            

            model = Logic.TransferModel().create_model(model="vgg",imageDim=224,lr=[1],opt=self.optimizer_f,act=self.activation_f,summary=True)
            model_list = Logic.TransferModel().run_model(model=model,traingen=tset,valgen=vset,epochs_=self.epoc,repeat=n_repeat)
            v2 = Logic.TransferModel().get_mean(data=model_list,repeat=n_repeat)
            v2_l = Logic.TransferModel().get_mean_loss(data=model_list,repeat=n_repeat)

            model = Logic.TransferModel().create_model(model="vgg",imageDim=224,lr=[2],opt=self.optimizer_f,act=self.activation_f,summary=True)
            model_list = Logic.TransferModel().run_model(model=model,traingen=tset,valgen=vset,epochs_=self.epoc,repeat=n_repeat)
            v3 = Logic.TransferModel().get_mean(data=model_list,repeat=n_repeat)
            v3_l = Logic.TransferModel().get_mean_loss(data=model_list,repeat=n_repeat)

            model = Logic.TransferModel().create_model(model="vgg",imageDim=224,lr=[3],opt=self.optimizer_f,act=self.activation_f,summary=True)
            model_list = Logic.TransferModel().run_model(model=model,traingen=tset,valgen=vset,epochs_=self.epoc,repeat=n_repeat)
            v4 = Logic.TransferModel().get_mean(data=model_list,repeat=n_repeat)
            v4_l = Logic.TransferModel().get_mean_loss(data=model_list,repeat=n_repeat)

            model = Logic.TransferModel().create_model(model="vgg",imageDim=224,lr=[4],opt=self.optimizer_f,act=self.activation_f,summary=True)
            model_list = Logic.TransferModel().run_model(model=model,traingen=tset,valgen=vset,epochs_=self.epoc,repeat=n_repeat)
            v5 = Logic.TransferModel().get_mean(data=model_list,repeat=n_repeat)
            v5_l = Logic.TransferModel().get_mean_loss(data=model_list,repeat=n_repeat)


            Logic.Visualize().plot_learningrates(l1=v1,l2=v2,l3=v3,l4=v4,l5=v5,label1=str(self.lr_rates[0]),label2=str(self.lr_rates[1]),label3=str(self.lr_rates[2]),label4=str(self.lr_rates[3]),label5=str(self.lr_rates[4]),path=self.fig_path)
            Logic.Visualize().plot_learningrates_loss(l1=v1_l,l2=v2_l,l3=v3_l,l4=v4_l,l5=v5_l,label1=str(self.lr_rates[0]),label2=str(self.lr_rates[1]),label3=str(self.lr_rates[2]),label4=str(self.lr_rates[3]),label5=str(self.lr_rates[4]),path=self.fig_path)

            K.clear_session()

        return "OPTIMIZERS OVER"


if __name__ == "__main__":
    P = Process()
    P.imDim  = 224
    P.t_path = "/Users/sharjeel/Documents/FYP/FYP/Dataset complete/Tuning_Dataset/Train"
    P.v_path = "/Users/sharjeel/Documents/FYP/FYP/Dataset complete/Tuning_Dataset/Test"
    P.fig_path = "/Users/sharjeel/Downloads/ZCC_Server/Tuner/Logic/"
    
    P.execute_activation_functions(mode='cpu',n_repeat=2,seed=1)
    #P.execute_optimizers_functions(mode='cpu',n_repeat=2,seed=1)
    #P.execute_learningrates_functions(mode='cpu',n_repeat=2,seed=1)

