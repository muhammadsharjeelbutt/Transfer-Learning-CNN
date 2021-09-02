from .TransferModel import TransferModel
from .DataPrep import DataPrep
from .Visualize import Visualize


def GetTransferModel():
    return TransferModel.Instance()

def GetDataPrep():
    return DataPrep.Instance()

def GetVisualize():
    return Visualize.Instance()