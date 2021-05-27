from os import name
import os
import unittest
import sys
import pandas as pd


sys.path.append("..")
from preprocessing.preprocessor import FindFrequency, DataError, ParamsError

class TestFindFreq(unittest.TestCase):
    
    def test_empty_dfs(self):
        dfs = []
        params = {}
        params['ycols'] = "SA1900282"
        params['dscol'] = 'Time'
        params['weekdays'] = {}
        params['weekends'] = {}
        params['orient'] = True
        find_freq = FindFrequency()
        self.assertRaises(DataError, find_freq.fit_transform, dfs, **params)
    
    def test_params_missing_ycols(self):
        dfs = [pd.DataFrame(), pd.DataFrame()]
        params = {}
        params['dscol'] = 'Time'
        params['weekdays'] = {}
        params['weekends'] = {}
        params['orient'] = True
        find_freq = FindFrequency()
        self.assertRaises(ParamsError, find_freq.fit_transform, dfs, **params)
    
    def test_params_missing_dscol(self):
        dfs = [pd.DataFrame(), pd.DataFrame()]
        params = {}
        params['ycols'] = "SA1900282"
        params['weekdays'] = {}
        params['weekends'] = {}
        params['orient'] = True
        find_freq = FindFrequency()
        self.assertRaises(ParamsError, find_freq.fit_transform, dfs, **params)

if __name__ == '__main__':
    unittest.main()