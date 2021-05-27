from logging import NOTSET
from os import name
import os
import unittest
import sys
import pandas as pd


sys.path.append("..")
from preprocessing.preprocessor import LoadData, FindFrequency, PeriodDetect, DataError, ParamsError, NoPeriodError

class TestFindFreq(unittest.TestCase):
    def test_not_multiple(self):
        _ = {'Time':[0,1],'SA1900282':[0,1]}
        dfs = [pd.DataFrame(_), pd.DataFrame(_)]
        params = {}
        params['ycols'] = "SA1900282"
        params['dscol'] = 'Time'
        params['weekdays'] = {'minfreq':pd.Timedelta('0 days 00:30:00'), 'colfreqs':{"_":pd.Timedelta('0 days 00:14:00')}}
        params['weekends'] = {'minfreq':pd.Timedelta('0 days 00:30:00'), 'colfreqs':{"_":pd.Timedelta('0 days 00:14:00')}}
        params['orient'] = True
        peiod_detect = PeriodDetect()
        self.assertRaises(ValueError, peiod_detect.fit_transform, dfs, **params)
    
    def test_no_period(self):
        dscol = 'Time'
        ycols = ['SA1900282']
        _ld = LoadData('for_tests/test_no_period.csv', dscol, ycols)
        params = {}
        dfs, params = _ld.fit_transform(**params)
        dfs, params = FindFrequency().fit_transform(dfs, **params)
        print(params)
        peiod_detect = PeriodDetect()
        self.assertRaises(NoPeriodError, peiod_detect.fit_transform, dfs, **params)
    
    def test_empty_dfs(self):
        dfs = []
        params = {}
        params['ycols'] = "SA1900282"
        params['dscol'] = 'Time'
        params['weekdays'] = {}
        params['weekends'] = {}
        params['orient'] = True
        peiod_detect = PeriodDetect()
        self.assertRaises(DataError, peiod_detect.fit_transform, dfs, **params)
    
    def test_params_missing_ycols(self):
        dfs = [pd.DataFrame(), pd.DataFrame()]
        params = {}
        params['dscol'] = 'Time'
        params['weekdays'] = {}
        params['weekends'] = {}
        params['orient'] = True
        peiod_detect = PeriodDetect()
        self.assertRaises(ParamsError, peiod_detect.fit_transform, dfs, **params)
    
    def test_params_missing_dscol(self):
        dfs = [pd.DataFrame(), pd.DataFrame()]
        params = {}
        params['ycols'] = "SA1900282"
        params['weekdays'] = {}
        params['weekends'] = {}
        params['orient'] = True
        peiod_detect = PeriodDetect()
        self.assertRaises(ParamsError, peiod_detect.fit_transform, dfs, **params)

    def test_empty_params(self):
        dfs = [pd.DataFrame(), pd.DataFrame()]
        params = {}
        peiod_detect = PeriodDetect()
        self.assertRaises(ParamsError, peiod_detect.fit_transform, dfs, **params)

    def test_params_missing_minfreq(self):
        dfs = [pd.DataFrame(), pd.DataFrame()]
        params = {}
        params['ycols'] = "SA1900282"
        params['dscol'] = 'Time'
        params['weekdays'] = {'colfreqs':{"SA1900282":2}}
        params['weekends'] = {}
        params['orient'] = True
        peiod_detect = PeriodDetect()
        self.assertRaises(ParamsError, peiod_detect.fit_transform, dfs, **params)

    def test_params_missing_colfreqs(self):
        dfs = [pd.DataFrame(), pd.DataFrame()]
        params = {}
        params['ycols'] = "SA1900282"
        params['dscol'] = 'Time'
        params['weekdays'] = {'minfreq':2}
        params['weekends'] = {}
        params['orient'] = True
        peiod_detect = PeriodDetect()
        self.assertRaises(ParamsError, peiod_detect.fit_transform, dfs, **params)

if __name__ == '__main__':
    unittest.main()