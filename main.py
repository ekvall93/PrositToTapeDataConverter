from DataConverter import DataConverter
from utils import PathHandler
if __name__ == "__main__":
    dataTypes = ["cid", "hcd"]
    dataSplits = ["ho", "train", "valid"]

    for datatype in dataTypes:
        for split in dataSplits:
            path = f"./hdf5/{datatype}/prediction_{datatype}_{split}.hdf5"

            if not PathHandler.isFile(path):
                raise RuntimeError(f"Path {path} don't exists. Download it with 'download_prosit_hdf5.sh'")

            print(f"Start to convert hdf5 to lmdb for datatype {datatype}, split {split}. \n")
            print(f"Saved at {path}")

            DC = DataConverter.convertFromPath(split, datatype, path, "./LMDB")
            DC.convert()


