import matplotlib.pyplot as plt

class Visualize:

    __instance__ = None

    @staticmethod
    def Instance():
        return Visualize.__instance__

    def __init__(self):
        if Visualize.__instance__ is None:
            Visualize.__instance__ = self

    def plot_activationfunctions(self,l1,l2,label1="",label2="",path=""):
        #b: blue, r: red, g: green, M: magenta, y: yellow
        epochs = range(len(l1))
        fig = plt.figure(figsize=(12, 8))
        plt.plot(epochs, l1, 'black', label=label1)
        plt.plot(epochs, l2, 'b', label=label2)
        plt.xticks(color='w')
        plt.title('Activation Functions')
        plt.ylabel("Accuracy")
        plt.xlabel("")
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        plt.legend(loc=0)
        fig.savefig(path+"activation_functions aggregate"+'.jpeg', dpi = 700)

    def plot_optimizers(self,l1,l2,l3,label1="",label2="",label3="",path=""):
            #b: blue, r: red, g: green, M: magenta, y: yellow
            epochs = range(len(l1))
            fig = plt.figure(figsize=(12, 8))
            plt.plot(epochs, l1, 'black', label=label1)
            plt.plot(epochs, l2, 'b', label=label2)
            plt.plot(epochs, l3, 'r', label=label3)
            plt.xticks(color='w')
            plt.title('Optimizers')
            plt.ylabel("Accuracy")
            plt.xlabel("")
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            plt.legend(loc=0)
            fig.savefig(path+"optimizers aggregate"+'.jpeg', dpi = 700)

    def plot_learningrates(self,l1,l2,l3,l4,l5,label1="",label2="",label3="",label4="",label5="",path=""):
            #b: blue, r: red, g: green, M: magenta, y: yellow
            epochs = range(len(l1))
            fig = plt.figure(figsize=(12, 8))
            plt.plot(epochs, l1, 'black', label=label1)
            plt.plot(epochs, l2, 'b', label=label2)
            plt.plot(epochs, l3, 'r', label=label3)
            plt.plot(epochs, l4, 'g', label=label4)
            plt.plot(epochs, l5, 'y', label=label5)
            plt.xticks(color='w')
            plt.title('Learning Rates')
            plt.ylabel("Accuracy")
            plt.xlabel("")
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            plt.legend(loc=0)
            fig.savefig(path+"learningrates aggregate"+'.jpeg', dpi = 700)
    
    def plot_learningrates_loss(self,l1,l2,l3,l4,l5,label1="",label2="",label3="",label4="",label5="",path=""):
            #b: blue, r: red, g: green, M: magenta, y: yellow
            epochs = range(len(l1))
            fig = plt.figure(figsize=(12, 8))
            plt.plot(epochs, l1, 'black', label=label1)
            plt.plot(epochs, l2, 'b', label=label2)
            plt.plot(epochs, l3, 'r', label=label3)
            plt.plot(epochs, l4, 'g', label=label4)
            plt.plot(epochs, l5, 'y', label=label5)
            plt.xticks(color='w')
            plt.title('Learning Rates')
            plt.ylabel("Loss")
            plt.xlabel("")
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            plt.legend(loc=0)
            fig.savefig(path+"learningrates aggregate"+'.jpeg', dpi = 700)