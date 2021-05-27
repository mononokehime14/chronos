from logging import NOTSET
from os import name
import os
import unittest
import sys
import pandas as pd


sys.path.append("..")
from preprocessing.preprocessor import FillGap, DataError, ParamsError

class TestAlignData(unittest.TestCase):
    
    def test_empty_dfs(self):
        dfs = []
        params = {}
        params['ycols'] = "SA1900282"
        params['dscol'] = 'Time'
        params['orient'] = True
        _ = FillGap()
        self.assertRaises(DataError, _.fit_transform, dfs, **params)
    
    def test_params_missing_ycols(self):
        dfs = [pd.DataFrame(), pd.DataFrame()]
        params = {}
        params['dscol'] = 'Time'
        params['orient'] = True
        _ = FillGap()
        self.assertRaises(ParamsError, _.fit_transform, dfs, **params)
    
    def test_params_missing_dscol(self):
        dfs = [pd.DataFrame(), pd.DataFrame()]
        params = {}
        params['ycols'] = "SA1900282"
        params['orient'] = True
        _ = FillGap()
        self.assertRaises(ParamsError, _.fit_transform, dfs, **params)

    def test_empty_params(self):
        dfs = [pd.DataFrame(), pd.DataFrame()]
        params = {}
        _ = FillGap()
        self.assertRaises(ParamsError, _.fit_transform, dfs, **params)

    def test_params_missing_minfreq(self):
        dfs = [pd.DataFrame(), pd.DataFrame()]
        params = {}
        params['ycols'] = "SA1900282"
        params['dscol'] = 'Time'
        params['weekdays'] = {'colfreqs':{"SA1900282":2}}
        params['orient'] = True
        _ = FillGap()
        self.assertRaises(ParamsError, _.fit_transform, dfs, **params)

    def test_params_missing_colfreqs(self):
        dfs = [pd.DataFrame(), pd.DataFrame()]
        params = {}
        params['ycols'] = "SA1900282"
        params['dscol'] = 'Time'
        params['weekdays'] = {'minfreq':2}
        params['orient'] = True
        _ = FillGap()
        self.assertRaises(ParamsError, _.fit_transform, dfs, **params)

    def test_params_missing_period(self):
        dfs = [pd.DataFrame(), pd.DataFrame()]
        params = {}
        params['ycols'] = "SA1900282"
        params['dscol'] = 'Time'
        params['weekdays'] = {'minfreq':2, 'colfreqs':{"SA1900282":2}}
        params['orient'] = True
        _ = FillGap()
        self.assertRaises(ParamsError, _.fit_transform, dfs, **params)

    def test_params_missing_zeropoint(self):
        dfs = [pd.DataFrame(), pd.DataFrame()]
        params = {}
        params['ycols'] = "SA1900282"
        params['dscol'] = 'Time'
        params['weekdays'] = {'minfreq':2, 'colfreqs':{"SA1900282":2}, 'period':2}
        params['orient'] = True
        _ = FillGap()
        self.assertRaises(ParamsError, _.fit_transform, dfs, **params) 


if __name__ == '__main__':
    unittest.main()