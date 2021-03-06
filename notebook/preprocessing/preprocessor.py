import logging
import math
import time
import numpy as np
from numpy.lib.function_base import median
import pandas as pd
import statsmodels.tsa.stattools as sts
from scipy.signal import argrelextrema
import glob
import zipfile

from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from sklearn.model_selection import train_test_split
from tqdm.notebook import tnrange, tqdm

"""This file contains all pre-processing modules needed for training and testing
Explainations of each module can be found from Chronos+ handover.ipynb or [ANDY]Chronos+ handover.ipynb
Here, I just turn all precessing functions into callable classes, enable them to pass down data and other information to next module in pipeline.
"""

#general class for all
class Adam:
    """Base class for all preprocessing transformers"""
    def __init__(self):
        self.name = 'adam'
    
    def _validate_params(self, name, dfs, **params):
        if len(dfs) == 0:
            raise DataError(f'Dataframes are empty! Error occur at {name} step.')
        
        if len(dfs) == 1:
            if dfs[0].empty:
                raise DataError(f'Dataframe is empty! Error occur at {name} step.')

        if not params:
            raise ParamsError(f'Empty params! Error occur at {name} step.')
        if 'ycols' not in params:
            raise ParamsError(f'Lack ycols value. Please ensure you called LoadData. Current step: {name}')
        if 'dscol' not in params:
            raise ParamsError(f'Lack dscol value. Please ensure you called LoadData. Current step: {name}')
        
        # if ('weekdays' not in params) | ('weekends' not in params):
        #     raise ParamsError(f'Lack weekdays and weekends in params. Please ensure you called LoadData. Current step: {name}')
        pattern_list = []
        if 'weekdays' in params:
            pattern_list.append('weekdays')
        if 'weekends' in params:
            pattern_list.append('weekends')

        for _ in pattern_list:
            if 'minfreq' not in params[_]:
                raise ParamsError(f'Lack minimal frequency data. Please ensure you called FindFrequency. Current step: {name}')
            if 'colfreqs' not in params[_]:
                raise ParamsError(f'Lack column frequencies data. Please ensure you called FindFrequency.Current step: {name}')
            if name in ['drop_extrema', 'normalizer', 'fill_gap','generate_input', 'left_pad']:
                if 'period' not in params[_]:
                    raise ParamsError(f'Lack column period data. Please ensure you called PeriodDetect.Current step: {name}')
            if name in ['normalizer','fill_gap']:
                if 'zeropoint' not in params[_]:
                    raise ParamsError(f'Lack column zeropoint data. Please ensure you called AlignData.Current step: {name}')
            if name == 'generate_input':
                if 'modnorms' not in params[_]:
                    raise ParamsError(f'Lack column modnorms data. Please ensure you called Normalizer.Current step: {name}')
            if name == 'left_pad':
                if 'profile' not in params[_]:
                    raise ParamsError(f'Lack median profile data. Testing pipeline needs median profile data which is gained after training step FillGap.Current step: {name}')

class ParamsError(Exception):
    def __init__(self, message, *args, **kwargs):
        self.message = message
        Exception.__init__(self, *args, **kwargs)

class DataError(Exception):
    pass

class MissingColumnError(Exception):
    def __init__(self, col, *args, **kwargs):
        self.column = col
        Exception.__init__(self, *args, **kwargs)

class SparseColumnError(Exception):
    pass

class NoPeriodError(Exception):
    pass

class MisalignedColumnsError(Exception):
    pass


class LoadData(Adam):
    def __init__(self, data_path, dscol, ycols, predefined_params=None):
        super().__init__()
        self.data_path = data_path
        self.name = 'load_data'
        self.dscol = dscol
        self.ycols = ycols
        self.predefined_params = predefined_params
    
    def _validate_params(self, **params):
        if len(self.data_path) == 0:
            raise ValueError("The data path is empty, cannot load data.")
        if params:
            raise ValueError("The params is not empty. LoadData should be the first step!")

    def fit_transform(self, **params):
        self._validate_params()
        _data_path = self.data_path

        if not self.predefined_params:
            params['ycols'] = self.ycols
            params['dscol'] = self.dscol
            params['orient'] = True
        else:
            params = self.predefined_params
            params['orient'] = False

        if _data_path.split(".")[1] == 'csv':
            return self.ingest(_data_path, self.dscol, self.ycols), params
        elif _data_path.split(".")[1] == 'zip':
            return self.ingest_from_zip(_data_path, self.dscol, self.ycols), params

    def ingest(self, data_path, dscol, ycols):
        try:
            df = pd.read_csv(data_path)
        except:
            raise ValueError("cannot load data from csv file.")

        if dscol not in df:
            raise MissingColumnError(dscol)

        if not ycols:
            raise ValueError(f"ycols is empty: {ycols}")

        for ycol in ycols:
            if ycol not in df:
                raise MissingColumnError(ycol)

        # if the date column is not parseable as a date, then we should discard the row
        df[dscol] = pd.to_datetime(df[dscol], errors="coerce")
        df = df[~df[dscol].isnull()]

        # parse value column as float. no exceptions
        for ycol in ycols:
            df[ycol] = pd.to_numeric(df[ycol], errors='coerce') 

        df_weekdays, df_weekends = self.separate_weekends(df, ycols, dscol)
        df_weekdays = df_weekdays.set_index(dscol).sort_index()[ycols]
        df_weekends = df_weekends.set_index(dscol).sort_index()[ycols]
        return [df_weekdays, df_weekends]
    
    def ingest_from_zip(self, filename, dscol, ycols):
        try:
            for zip_file in glob.glob(filename):
                zf = zipfile.ZipFile(zip_file)
                dfs = []
                dfs = [pd.read_csv(zf.open(f)) for f in zf.namelist()]
                df = pd.concat(dfs,ignore_index=True)
        except:
            raise ValueError("cannot load data from zip file.")
            
        if dscol not in df:
            raise MissingColumnError(dscol)

        if not ycols:
            raise ValueError(f"ycols is empty: {ycols}")

        for ycol in ycols:
            if ycol not in df:
                raise MissingColumnError(ycol)

        # if the date column is not parseable as a date, then we should discard the row
        df[dscol] = pd.to_datetime(df[dscol], errors="coerce")
        df = df[~df[dscol].isnull()]

        # parse value column as float. no exceptions
        for ycol in ycols:
            df[ycol] = pd.to_numeric(df[ycol], errors='coerce')  
        df_weekdays, df_weekends = self.separate_weekends(df, ycols, dscol)
        df_weekdays = df_weekdays.set_index(dscol).sort_index()[ycols]
        df_weekends = df_weekends.set_index(dscol).sort_index()[ycols]

        return [df_weekdays, df_weekends]
    
    def separate_weekends(self, df, ycols, dscol):
        list_weekday = []
        list_weekends = []
        for idx, row in df.iterrows():
            weekno = row[dscol].weekday()      
            if weekno < 5:
                list_weekday.append(row.tolist())
            else:
                list_weekends.append(row.tolist())
        _ycols = ycols.copy()
        _ycols.insert(0,dscol)
        df_weekday = pd.DataFrame(list_weekday, columns = _ycols)
        df_weekends = pd.DataFrame(list_weekends, columns = _ycols)
        return df_weekday, df_weekends


#find frequency part
def remove_duplicate_rows(df):
    """returns df with duplicates removed 
    """    
    idx_name = df.index.name  # [ANDY] this is the same as dscol
    return df.reset_index().drop_duplicates().set_index(idx_name)

def detect_ds_frequency(df, dscol):
    """detects the frequency of datetime in df.
    
    1) subtract all the datetime in ds by its next datetime to retrieve the deltas
    
    2) count the number of occurrences for each delta
    
    Parameters
    ----------
    df : pandas.DataFrame 
        The dataframe to detect the datetime frequency, needs to have a sorted datetime index
    Returns
    -------
    freq : pd.Series of size 1 whose index is the modal timedelta and value is the number of times it appeared if there is one mode
    and it appeared > 50% of the time
        else pd.Series of max size 10 sorted by ascending order of modal timedeltas
    """
    deltas = df.reset_index()[dscol].diff()  # [ANDY] deltas = df.reset_index()[dscol].diff()
    threshold = len(deltas)/2
    delta_counts = deltas.value_counts()
    modal_delta = delta_counts.index[0]  # [ANDY] modal_delta = delta_counts.index[0]
    mdcount = delta_counts.iloc[0]  # mdcount = delta_counts.iloc[0]

    if mdcount > threshold:
        return delta_counts.head(1)

    dc = [item for item in delta_counts.iteritems()]
    # sort by ascending order of timedelta
    dc.sort(key=lambda e:e[0])
    # sort by descending order of occurrences
    dc.sort(key=lambda e:e[1], reverse=True)
    dc = dc[:50]

    return pd.Series(data=[e[1] for e in dc], index=[e[0] for e in dc])

#check for sparse part
def check_sparse_cols(df, colfreqs):
    totalduration = df.index[-1] - df.index[0]
    for y, freq in colfreqs.items():
        numpoints = len(df[[y]].dropna())
        # every column must cover 75% of the duration
        expected = totalduration // freq * 0.75
        if numpoints < expected:
            raise SparseColumnError(f"column {y} has frequency {freq} and {numpoints} points, but duration of dataset is {totalduration}")

    
class FindFrequency(Adam):
    def __init__(self):
        super().__init__()
        self.name = 'find_frequency'
    
    def _validate_params(self, dfs, **params):
        if len(dfs) == 0:
            raise DataError('Dataframes are empty! Error occur at Find Frequency part.')
        if 'ycols' not in params:
            raise ParamsError(f'Lack ycols value. Please ensure you called LoadData. Current step: {self.name}')
        if 'dscol' not in params:
            raise ParamsError(f'Lack dscol value. Please ensure you called LoadData. Current step: {self.name}')
    
    def fit_transform(self, dfs, **params):
        self._validate_params(dfs, **params)
        for i in tqdm(range(len(dfs)), desc = 'Finding frequency'):
            df = dfs[i]
            current_pattern = 'weekdays'
            if i:
                current_pattern = 'weekends'
                
            if df.empty:
                continue

            if params['orient']:
                params[current_pattern] = {}

            df = remove_duplicate_rows(df)

            dupe_ds = list(df[df.index.duplicated()].index.drop_duplicates())
            # we also reject if we have duplicate datetimes that are not duplicate rows
            if len(dupe_ds)!=0:
                raise Exception("rows with duplicate datetimes detected")

            colfreqs = {}
            for y in params['ycols']:
                currdf = df[[y]].dropna()
                freqs = detect_ds_frequency(currdf, params['dscol'])
                if len(freqs)!=1:
                    # we reject if we have no modal sampling frequency
                    raise Exception("more than one sampling frequency was detected")

                colfreqs[y] = freqs.index[0]

            #combine sparse column detection here with frequency detction
            #check_sparse_cols(df,colfreqs)

            minfreq = min(colfreqs.values())
            params[current_pattern]['minfreq'] = minfreq
            params[current_pattern]['colfreqs'] = colfreqs

            if df.empty:
                raise DataError(f'Data of {current_pattern} is lost in {self.name} session.')
            dfs[i] = df

        return dfs, params



#detect period part
class PeriodDetect(Adam):
    def __init__(self):
        super().__init__()
        self.name = 'period_detect'
    
    def fit_transform(self, dfs, **params):
        self._validate_params(self.name, dfs, **params)
        for i in tqdm(range(len(dfs)), desc = 'Detecting Period'):
            df = dfs[i]
            current_pattern = 'weekdays'
            if i:
                current_pattern = 'weekends'
            if df.empty:
                continue
            _colfreqs = params[current_pattern]['colfreqs']
            _minfreq = params[current_pattern]['minfreq']

            notime = pd.Timedelta(0)
            for f in _colfreqs.values():
                if f % _minfreq != notime:
                    raise ValueError("detected sampling frequency that is not a multiple of the minimum sampling frequency")

            # there is no point calculating autocorrelation for lags greater than n/2
            nlags = len(df)//2  # [ANDY] assuming df covers at least two periods ? Why?
            acfs = np.zeros((nlags+1,))
            for y in _colfreqs:
                yvals = df[y].dropna()
                yacfs = sts.acf(yvals, nlags=len(yvals)//2)
                step_size = _colfreqs[y]//_minfreq
                yacfs_idx = range(0, len(acfs), step_size)
                # we simplify by cropping because if len(df) % len(yvals) != step_size we get tedious off-by-one errors
                yacfs = yacfs[:len(yacfs_idx)]
                # if y has missing values, yacfs can be shorter than yacfs_idx -_-
                yacfs_idx = yacfs_idx[:len(yacfs)]
                acfs[yacfs_idx] += yacfs

            max_corr_points = argrelextrema(acfs, np.greater, order=max(len(df)//1000, 2))  # [ANDY] why max(len(df)//1000, 2))?
            max_corr_points = max_corr_points[0]
            max_corr_points = max_corr_points[acfs[max_corr_points]>0.2]  # [ANDY] why 0.2?
            max_corr_points = np.insert(max_corr_points, 0, 0, axis=0)
            max_cor_diff = []
            for point_1 in max_corr_points:
                for point_2 in max_corr_points:
                    if point_1==point_2: continue
                    max_cor_diff.append(abs(point_1-point_2))
            max_cor_diff = np.array(max_cor_diff)
            unique_vals, counts = np.unique(max_cor_diff, return_counts=True)
            adjust_counts = []
            for idx in range(len(unique_vals)):
                adjust_counts.append(np.sum(counts[np.where(unique_vals%unique_vals[idx]==0)]))

            if len(adjust_counts) > 0:
                if (np.max(adjust_counts)>5):  # [ANDY] why 5?
                    params[current_pattern]['period'] = unique_vals[np.argmax(adjust_counts)]
                    continue
            raise NoPeriodError("no period detected")

        return dfs, params

#align data
class AlignData(Adam):
    def __init__(self):
        super().__init__()
        self.name = 'align_data'

    def fit_transform(self, dfs, **params):
        """aligns data according to sampling frequencies"""
        self._validate_params(self.name, dfs, **params)
        for i in tqdm(range(len(dfs)), desc = 'Aligning data'):
            df = dfs[i]
            current_pattern = 'weekdays'
            if i:
                current_pattern = 'weekends'
            if df.empty:
                continue

            _colfreqs = params[current_pattern]['colfreqs']
            if ('anchors' in params[current_pattern]) & (not params['orient']):
                _anchors = params[current_pattern]['anchors']
            else:
                _anchors = None
            
            if ('zeropoint' in params[current_pattern]) & (not params['orient']):
                _zeropoint = params[current_pattern]['zeropoint']
            else:
                _zeropoint = None
            
            aligned_df = {}
            for y, freq in _colfreqs.items():
                col_df = df[y].dropna()
                index_diff = col_df.index[1:]-col_df.index[:-1]
                sampling_groups = []
                cur_group = [col_df.index[0]]
                for idx in range(len(index_diff)):
                    if (index_diff[idx]==freq):
                        cur_group.append(col_df.index[idx+1])
                    else:
                        if (len(cur_group)>0):
                            sampling_groups.append(cur_group)
                        cur_group = [col_df.index[idx+1]]

                if (len(cur_group)>0):
                    sampling_groups.append(cur_group)
                    
                merged_sampling_group = []
                while len(sampling_groups)>0:
                    cur_group = sampling_groups.pop(0)
                    merged_idx = []
                    for gidx, group in enumerate(sampling_groups):
                        if ((group[0]-cur_group[-1])%freq==pd.Timedelta(0)):  # [ANDY] (group[0]-cur_group[-1])%freq == pd.Timedelta(0)?
                            cur_group.extend(group)
                            merged_idx.append(gidx)
                
                    for m_idx in sorted(merged_idx, reverse=True):
                        del sampling_groups[m_idx]
                    
                    merged_sampling_group.append(cur_group)
                    
                merged_group_lens = [len(group) for group in merged_sampling_group]
                dominant_group_idx = np.argmax(merged_group_lens)
                init_group = merged_sampling_group[dominant_group_idx]
                
                n_samplings_before = round((init_group[0]-df.index[0])/freq)+1
                n_samplings_after = round((df.index[-1] - init_group[0])/freq)  # [ANDY] should +1?
                index_before = pd.date_range(end=init_group[0], periods = n_samplings_before, freq = freq, closed = 'left')
                index_after = pd.date_range(start=init_group[0], periods = n_samplings_after, freq = freq)
                aligned_index = index_before.append(index_after)
                
                # need to understand the mechanism of reindex on how it fill the NA value
                aligned_col_df = col_df.reindex(index=aligned_index, method='nearest', limit=1)
                aligned_df[y] = aligned_col_df
            
            aligned_df = pd.DataFrame(aligned_df)

            if _anchors is None:
                _anchors = {y:aligned_df[[y]].dropna().index[0] for y in _colfreqs}
                #NOTE: the zeropoint value is necessary for building the time_index column, which fill_gaps does
                _zeropoint = aligned_df.index[0]
                params[current_pattern]['anchors'] = _anchors
                params[current_pattern]['zeropoint'] = _zeropoint
            #if anchors is passed in, do a sanity check on alignment
            else:
                notime = pd.Timedelta(seconds=0)
                for y, yfreq in _colfreqs.items():
                    thisorigin = aligned_df[[y]].dropna().index[0]
                    if (thisorigin - _anchors[y]) % yfreq != notime:
                        raise MisalignedColumnsError(f"after aligning the data, column {y}'s zero point {thisorigin} is not aligned with the recorded anchor point {_anchors[y]} with sampling frequency {yfreq}")


            if aligned_df.empty:
                raise DataError(f'Data of {current_pattern} is lost in {self.name} session.')
            dfs[i] = aligned_df
        return dfs, params

#drop extrema
class DropExtrema(Adam):
    def __init__(self):
        super().__init__()
        self.name = 'drop_extrema'

    def fit_transform(self, dfs, **params):  # [ANDY] confirmed ts has been aligned, no need to fill ts gaps in this func again
        """drop values beyond 2 standard deviations from the mean"""
        # calculate the points where the gap is more than n_max_fill*interval
        self._validate_params(self.name, dfs, **params)
        for i in tqdm(range(len(dfs)), desc = 'Dropping Extrema'):
            df = dfs[i]
            current_pattern = 'weekdays'
            if i:
                current_pattern = 'weekends'
            if df.empty:
                continue
            _minfreq = params[current_pattern]['minfreq']
            _period = params[current_pattern]['period'] 
            _ycols = params['ycols']

            time_between_points = df.index.to_series().diff()
            gaps = time_between_points[(time_between_points>=2*_minfreq)]

            index_all = df.index
            for gidx, gap in gaps.iteritems():
                gap_length = math.ceil(gap/_minfreq)
                padding = pd.date_range(end = gidx, periods = gap_length, freq = _minfreq, closed = 'left')
                index_all = index_all.append(padding)

            index_all = index_all.sort_values()
            df = df.reindex(index_all)  # [ANDY] df is already aligned in last step, why there are still ts gaps to be filled?

            time_index = [i for i in range(0, _period)]
            data_time_index = np.array([i % _period for i in range(0, len(df))])
            data_time_index_series = pd.Series(index=index_all, data=data_time_index)

            n_std = 2

            for y in _ycols:
                for period_idx in time_index:
                    lbound = df[y][data_time_index_series==period_idx].mean()-n_std*df[y][data_time_index_series==period_idx].std()
                    ubound = df[y][data_time_index_series==period_idx].mean()+n_std*df[y][data_time_index_series==period_idx].std()
                
                    sub_data = df[y][data_time_index_series==period_idx]
                    if (len(sub_data[(sub_data<lbound) | (sub_data>ubound)].index)>0):
                        df.loc[sub_data[(sub_data<lbound) | (sub_data>ubound)].index, y] = None
            if df.empty:
                raise DataError(f'Data of {current_pattern} is lost in {self.name} session.')
            dfs[i] = df
        return dfs, params

#normalisation part
class Normalizer(Adam):
    def __init__(self):
        super().__init__()
        self.name = 'normalizer'

    def fit_transform(self, dfs, **params):
        self._validate_params(self.name, dfs, **params)
        for i in tqdm(range(len(dfs)), desc = 'Normalising'):
            df = dfs[i]
            current_pattern = 'weekdays'
            if i:
                current_pattern = 'weekends'
            if df.empty:
                continue
            if params['orient']:
                modnorms = {y:( df[y].min(), df[y].max() ) for y in params['ycols']}
                params[current_pattern]['modnorms'] = modnorms
            df['time_index'] = [((t-params[current_pattern]['zeropoint'])//params[current_pattern]['minfreq'])%params[current_pattern]['period'] for t in df.index]  # [ANDY] period is matched to minfreq?
            if df.empty:
                raise DataError(f'Data of {current_pattern} is lost in {self.name} session.')
            dfs[i] = df
        return dfs, params

#get median profiles
# def get_median_profiles(df, ycols, period):  # [ANDY] get median profiles after dropping extreme values?
#     time_index = list(range(period))

#     median_profiles = pd.DataFrame(index=time_index)

#     for y in ycols:
#         median_profiles[y] = [df[df['time_index']==t][y].median() for t in time_index]

#     return median_profiles, time_index

def get_median_profiles(df, ycols, period):  # [ANDY] get median profiles after dropping extreme values?
    time_index = list(range(period))

    median_profiles = pd.DataFrame(index=time_index)

    for y in ycols:
        median_profiles[y] = [df[df['time_index']==t][y].median() for t in time_index]

    return median_profiles, time_index

#fill gap part
class FillGap(Adam):
    def __init__(self):
        super().__init__()
        self.name = 'fill_gap'
    
    def fit_transform(self, dfs, **params):
        self._validate_params(self.name, dfs, **params)
        for i in tqdm(range(len(dfs)), desc = 'Filling gap'):
            df = dfs[i]
            current_pattern = 'weekdays'
            if i:
                current_pattern = 'weekends'
            if df.empty:
                continue
            _colfreqs = params[current_pattern]['colfreqs']
            _minfreq =params[current_pattern]['minfreq']
            _period = params[current_pattern]['period']
            _median_profiles, _time_index = get_median_profiles(df, params['ycols'], _period)
            _zeropoint = params[current_pattern]['zeropoint']
            params[current_pattern]['profile'] = _median_profiles
            params[current_pattern]['time_index'] = _time_index
            # df = df.copy()
            firstts = df.index[0]
            lastts = df.index[-1]
            for y, yfreq in _colfreqs.items():
                yfactor = yfreq//_minfreq
                yperiod = _period//yfactor
                ydf = df[[y]].dropna()  # [ANDY] df already aligned.. if don't dropna, df[[y]] is already alighed with correct ts, just need to fill value gaps instead of ts gaps
                ydf, gaps = self.gap_reindex(ydf, yfreq, firstts, lastts)  # [ANDY] why still need to fill index(ts) gaps?
                ydf['time_index'] = [((t-_zeropoint)//_minfreq)%_period for t in ydf.index]
                ydf = self.fill_gaps_col(ydf, y, gaps, yfreq, yperiod, _median_profiles)
                #newindex = ydf.index.union(df.index).sort_values()  # [ANDY] redundant, index is already aligned
                #df = df.reindex(newindex)  # [ANDY] redundant, index is already aligned
                df[y] = ydf[y]

            #df['time_index'] = [((t-_zeropoint)//_minfreq)%_period for t in df.index]
            if df.empty:
                raise DataError(f'Data of {current_pattern} is lost in {self.name} session.')
            dfs[i] = df
        return dfs, params

    def gap_reindex(self, df, freq, firstts, lastts):
        # calculate the points where the gap is more than n_max_fill*freq
        # these points don't want to interpolate
        time_between_points = df.index.to_series().diff()

        # if first timestamp for this column is not equal to first timestamp for all columns, we need to fill
        yfirstts = df.index[0]
        if yfirstts!=firstts:
            time_between_points.at[yfirstts] = yfirstts - firstts + freq

        # if last timestamp for this column is not equal to last timestamp for all columns, we need to fill
        ylastts = df.index[-1]
        if ylastts!=lastts:
            time_between_points.at[lastts+freq] = lastts - ylastts + freq

        gaps = time_between_points[(time_between_points>=2*freq)]

        index_all = df.index
        padded_indices = pd.date_range(end = min(index_all), periods = 0, freq = freq, closed = 'left')  # [ANDY] why this?
        # create index positions for gaps
        for gidx, gap in gaps.iteritems():
            gap_length = math.ceil(gap/freq)
            padding = pd.date_range(end = gidx, periods = gap_length, freq = freq, closed = 'left')
            padded_indices = padded_indices.union(padding)  # [ANDY] why this?
            index_all = index_all.append(padding)

        index_all = index_all.sort_values()

        df = df.reindex(index_all)
        return df, gaps

    def fill_gaps_col(self, df, y, gaps, freq, period, median_profiles):
        # interpolate large gaps using median values
        max_miss_length = period//10
        for gidx, gap in gaps.iteritems():
            if (gap>max_miss_length*freq):
                gap_length = math.ceil(gap/freq)
                n_periods = 2
                while (n_periods<=gap_length):
                    pre_index = pd.date_range(end = gidx, periods = n_periods, freq = freq, closed = 'left')
                    pre_index = pre_index[0]

                    df.loc[pre_index, y] = median_profiles.loc[df.loc[pre_index, "time_index"], y]

                    n_periods = n_periods+1

        # small gaps interpolated linearly
        return df.interpolate(method='linear', limit_direction="both")

def normdf(df, modnorms):
    normed = df.copy()
    for y, ( ymin, ymax ) in modnorms.items():
        mmdiff = ymax - ymin
        normed[y] = (normed[y]-ymin)/mmdiff
    return normed

#generate model input
class GenerateInput(Adam):
    def __init__(self, times=None, offset=None):
        super().__init__()
        self.name = 'generate_input'
        self.offset = offset
        self.times = times
    
    def generate_model_input(self, df, current_pattern, params):
        period = params[current_pattern]['period']
        modnorms = params[current_pattern]['modnorms']
        ycols = params['ycols']
        if params['orient']:
            median_values = None
        else:
            median_values = params[current_pattern]['profile']
        t = params[current_pattern]['time_index'][-1]
        if self.times is None:
            fullperiods = df.iloc[period-1:]  # [ANDY] why? should be period-1
            times = fullperiods[fullperiods['time_index']==t].index
        else:
            times = self.times
        offset = period-1 if self.offset is None else self.offset
        window_size = 10
        rows = []
        for t in times:
            idx = df.index.get_loc(t)+1
            rowdf = df.iloc[idx-period:idx].copy()
            if median_values is not None and window_size<period:
                replaceidx = rowdf.iloc[:len(rowdf)-window_size].index
                rowdf.loc[replaceidx, ycols] = median_values.loc[rowdf.loc[replaceidx, "time_index"], ycols].to_numpy()
            # normalize training/scoring data
            rowdf = normdf(rowdf, modnorms)
            startpos = period-1-offset
            rowdf = rowdf.iloc[startpos:].append(rowdf.iloc[:startpos])  # [ANDY] why?
            row = np.concatenate([rowdf[y].dropna() for y in ycols])
            rows.append(row)

        collens = set(len(r) for r in rows)
        if len(collens)!=1:
            logging.error(f"detected row lengths: {collens}")
            raise MisalignedColumnsError("unable to create input data because dimensions don't match")

        numcols = collens.pop()
        df = np.concatenate(rows).reshape((-1, numcols))
        if not df.size:
            raise DataError(f'Data of {current_pattern} is lost in {self.name} session.')
        return df

    def fit_transform(self, dfs, **params):
        """
        generate model input for windows ending at timestamps in the times arguments

        Assumptions:
            1. times all have the same time index relative to period
            2. sampling intervals for each column are regular
        """
        self._validate_params(self.name, dfs, **params)
        for i in tqdm(range(len(dfs)), desc = 'Generating Input'):
            df = dfs[i]
            current_pattern = 'weekdays'
            if i:
                current_pattern = 'weekends'
            if df.empty:
                continue        
            dfs[i] = self.generate_model_input(df, current_pattern, params)
        return dfs, params


class LeftPad(Adam):
    def __init__(self):
        super().__init__()
        self.name = 'left_pad'
    
    def fit_transform(self, dfs, **params):
        """leftpads dataframe by period-1 points with median values for each column"""
        self._validate_params(self.name, dfs, **params)
        for i in tqdm(range(len(dfs))):
            df = dfs[i]
            current_pattern = 'weekdays'
            if i:
                current_pattern = 'weekends'
            df = df.copy()
            minfreq = params[current_pattern]['minfreq']
            period = params[current_pattern]['period']
            ycols = params['ycols']
            profile = params[current_pattern]['profile']
            firstindex = df['time_index'][0]
            leftpad_index = [i%period for i in range(firstindex - (period - 1), firstindex)]
            dat = {y:[profile.loc[i][y] for i in leftpad_index] for y in ycols}
            dat['time_index'] = leftpad_index
            leftdf = pd.DataFrame(
                data = dat,
                index = pd.date_range(end=df.index[0], periods=period, freq=minfreq, closed='left'))
            df = leftdf.append(df)
            if df.empty:
                raise DataError(f'Data of {current_pattern} is lost in {self.name} session.')
            dfs[i] = df

        return dfs, params