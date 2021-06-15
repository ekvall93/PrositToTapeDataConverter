import lmdb
from keras.utils import HDF5Matrix
import h5py
import pickle as pkl
import numpy as np
import shutil
from tqdm import tqdm
import os 
from typing import Union, Tuple, Dict
from pathlib import Path



class hdf5Loader:
    """ Load hdf5 file """
    @staticmethod
    def from_hdf5(path: str, n_samples:Union[int,None] = None)->HDF5Matrix:
        # Get a list of the keys for the datasets
        with h5py.File(path, 'r') as f:
            dataset_list = list(f.keys())
        # Assemble into a dictionary
        data = dict()
        for dataset in dataset_list:
            data[dataset] = HDF5Matrix(path, dataset, start=0, end=n_samples, normalizer=None)
        return data

class SequenceConverter:
    """ Convert Prosit integer peptide sequence to Tape peptide sequence """
    @staticmethod
    def intToPeptide(sequence:np.array)->str:
        #Set M(ox) = X i.e., IUPAC unlown aminoacid
        ALPHABET = {
        "A": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "K": 9,
        "L": 10,
        "M": 11,
        "N": 12,
        "P": 13,
        "Q": 14,
        "R": 15,
        "S": 16,
        "T": 17,
        "V": 18,
        "W": 19,
        "Y": 20,
        "X": 21,
        }
        ALPHABET_S = {integer: char for char, integer in ALPHABET.items()}
        return "".join([ALPHABET_S[int(i)] for i in sequence if int(i)!=0])

class BatchLoader:
    """ Load hdf5 in batches for speed-up"""
    def __init__(self):
        pass
    @staticmethod
    def getChunks(n: int, batch_size: int)->int:
        """ Get number of batches """
        return (n - 1) // batch_size + 1

    def getBatchIxs(self, n: int, batch_size: int)->Tuple[int,int]:
        """ Get start and end index for batch """
        chunks = self.getChunks(n, batch_size)
        for i in tqdm(range(chunks)):
            start, end = i*batch_size, (i+1)*batch_size
            if end >= n:
                end = n
            yield start, end

class PathHandler:
    """ Handled dir for data """
    def deleteDir(self, rt_path: str)->None:
        """ Delete dir """
        if self.isDir(rt_path):
            try:
                shutil.rmtree(rt_path)
            except Exception as e:
                print(e)

    @staticmethod
    def createLMDBdir(rt_path:str, n_data_points: int)->None:
        """ Creat LMDB dir """
        env = lmdb.open(str(rt_path), map_size=int(1e12))
        with env.begin(write=True) as txn:
            key = b"num_examples"
            value = n_data_points
            txn.put(key, pkl.dumps(value))
        env.close()

    @staticmethod
    def isDir(path: str)->bool:
        """ Check if dir exists """
        return os.path.isdir(path) 

    @staticmethod
    def isFile(path: str)->bool:
        """ Check if file exists """
        return Path(path).is_file()
    
    @staticmethod
    def createDir(path: str)->None:
        """ Create dir """
        os.mkdir(path)
    
class SaveLMDB(PathHandler):
    """ Save datapoint to LMDB file """
    def __init__(self):
        PathHandler.__init__(self)

    @staticmethod
    def save(rt_path: str, data: dict, key: str)->None:
        """ Save data point """
        env = lmdb.open(str(rt_path), map_size=int(1e12))
        with env.begin(write=True) as txn:
            key = str(key)
            txn.put(key.encode("ascii"), pkl.dumps(data))
        env.close()