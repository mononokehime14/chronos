from os import name
import os
import unittest
import sys


from pandas.errors import EmptyDataError
print(os.getcwd())
sys.path.append("..")
print(sys.path)
from preprocessing.preprocessor import LoadData, MissingColumnError

class TestLoadData(unittest.TestCase):

    def test_empty_path(self):
        data_path = ''
        dscol = "Time"
        ycols = ["SA1900282"]
        data_load = LoadData(data_path=data_path, dscol=dscol, ycols=ycols)
        with self.assertRaises(ValueError) as context:
            _= data_load.fit_transform()
        self.assertTrue("The data path is empty, cannot load data." in str(context.exception))

    
    def test_empty_data(self):
        data_path = 'for_tests/test_empty_data.csv'
        dscol = 'Time'
        ycols = ["SA1900282"]
        data_load = LoadData(data_path=data_path, dscol=dscol, ycols=ycols)
        self.assertRaises((ValueError, EmptyDataError), data_load.fit_transform)
        # with self.assertRaises((ValueError, EmptyDataError)) as cm:
        #     _ = data_load.fit_transform()
    
    def test_missing_dscol(self):
        data_path = 'for_tests/test_missing_dscol.csv'
        dscol = 'Time'
        ycols = ["SA1900282"]
        data_load = LoadData(data_path=data_path, dscol=dscol, ycols=ycols)
        
        self.assertRaises(MissingColumnError, data_load.fit_transform)
    
    def test_missing_ycol(self):
        data_path = 'for_tests/test_missing_ycol.csv'
        dscol = 'Time'
        ycols = ["SA1900282"]
        params = {}
        data_load = LoadData(data_path=data_path, dscol=dscol, ycols=ycols)
        
        self.assertRaises(MissingColumnError, data_load.fit_transform)
            
    def test_no_weekends(self):
        data_path = 'for_tests/test_no_weekends.csv'
        dscol = 'Time'
        ycols = ["SA1900282"]
        data_load = LoadData(data_path=data_path, dscol=dscol, ycols=ycols)
        params = {}
        dfs, _ = data_load.fit_transform(**params)
        self.assertTrue(dfs[1].empty)


if __name__ == '__main__':
    unittest.main()