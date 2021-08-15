from utils import PrepareTapeData, PathHandler, hdf5Loader
import pickle as pkl
from keras.utils import HDF5Matrix
from typing import TypeVar, Type, Union, Dict

T = TypeVar('T', bound='TapeOutputToPrositHDF5')


class TapeOutputToPrositHDF5(PrepareTapeData, PathHandler, hdf5Loader):
    """Convert Tape result into Prsoit HDF5"""
    def __init__(self, prosit_hdf5_data: HDF5Matrix, tape_result: dict)->None:
        PrepareTapeData.__init__(self, prosit_hdf5_data, tape_result)
        PathHandler.__init__(self)
        hdf5Loader.__init__(self)
        
    @classmethod
    def fromPath(cls: Type[T], prositHDF5path: str, tapeOutPath: str)->T:
        """Init from TapeOutputToPrositHDF5"""
        if not PathHandler.isFile(prositHDF5path):
                raise RuntimeError(f"Path {prositHDF5path} to prosit hdf5-file don't exist.")
        if not PathHandler.isFile(tapeOutPath):
                raise RuntimeError(f"Path {tapeOutPath} to tape output-file don't exist.")

        prositHDF5data = cls.from_hdf5(prositHDF5path)
        tapeOutData = pkl.load(open(tapeOutPath, "rb"))

        return cls(prositHDF5data, tapeOutData)

    def convert(self, out_path: str)->None:
        """Convert Tape result into Prosit hdf5-file"""
        tape_data_hdf5 = self.createTapeHDF5Dict()
        self.deleteFile(out_path)
        self.to_hdf5(tape_data_hdf5, out_path)
        print("HDF5 file has been successfully saved at {}".format(out_path))