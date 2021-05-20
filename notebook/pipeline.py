import os
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable

print(os.getcwd())
from preprocessing.preprocessor import LeftPad, LoadData, FindFrequency, PeriodDetect, AlignData, DropExtrema, Normalizer, FillGap, GenerateInput
from preprocessing.preprocessor import ParamsError

class Pipeline():
    def __init__(self, ae_list=None, params_dict=None):
        self.ae_list = ae_list
        self.params_dict = params_dict

    def _validate_tasklist(self, tasks):
        if len(tasks) == 0:
            raise ValueError("Empty progress list provided. Chronos+ will slack = =.")
        
        task_names = [_.name for _ in tasks]
        if task_names[0] != 'load_data':
            raise ValueError("Lack LoadData transformer which is necessary at the first of tasks list.")

        if ('benchmark' in task_names) & ('train_ae' in task_names):
            raise ValueError("Cannot have both benchmark testing and training simultaneously")
        else:
            if ('benchmark' in task_names) & (task_names[-1] != 'benchmark'):
                raise ValueError("Benchmark testing must be the last step of the pipeline, please~")
            
            if ('train_ae' in task_names) & (task_names[-1] != 'train_ae'):
                raise ValueError("Training must be the last step of the pipeline, please~")
        
        return (task_names[-1] == 'benchmark') | (task_names[-1] == 'train_ae')
    
    def _validate_params(self, params):
        if 'ycols' not in params:
            raise ParamsError(f'Lack ycols value in params.')
        check_list = ['time_index', 'period', 'modnorms', 'profile']
        for p in ['weekdays', 'weekends']:
            for _ in check_list:
                if _ not in params[p]:
                    raise ParamsError(f'Lack {_} value in params[{p}].')


    def fit_transform(self, tasks):
        last_step_flag = self._validate_tasklist(tasks)
        if (not last_step_flag) | (tasks[-1].name != 'train_ae'):
            raise ValueError(f"fit transform function needs the last step to be train_ae estimator.")

        dfs = pd.DataFrame()
        params = {}
        for i in range(len(tasks) - 1):
            _transformer = tasks[i]
            if _transformer is None:
                print(f"Chronos+ skiped task {i} because transformer is None stype.")
                continue
            
            if i == 0:
                try:
                    dfs, params = _transformer.fit_transform(**params)
                except:
                    raise ValueError('Error at loading data step.')
            else:
                try:
                    dfs, params = _transformer.fit_transform(dfs, **params)
                except:
                    raise ValueError(f'error at step {i + 1}')
        
        _last = tasks[-1]
        if _last is None:
            raise ValueError(f"Chrono+ want to proceed with training AE but the transformer is empty. (Why is this error possible?T_T) ")
        try:
            ae_list = _last.fit_transform(dfs, **params)
        except:
            raise ValueError(f"Error at training AE")
        return ae_list, params
    
    def transform(self, tasks):
        last_step_flag = self._validate_tasklist(tasks)
        if last_step_flag:
            raise ValueError(f"fit transform function needs the last step to be train_ae estimator.")

        dfs = pd.DataFrame()
        params = {}
        for i in range(len(tasks)):
            _transformer = tasks[i]
            if _transformer is None:
                print(f"Chronos+ skiped task {i} because transformer is None stype.")
                continue
            
            if i == 0:
                try:
                    dfs, params = _transformer.fit_transform(**params)
                except:
                    raise ValueError('Error at loading data step.')
            else:
                try:
                    dfs, params = _transformer.fit_transform(dfs, **params)
                except:
                    raise ValueError(f'error at step {i + 1}')
        return dfs, params

    def benchmark_transform(self, tasks):
        last_step_flag = self._validate_tasklist(tasks)
        if (not last_step_flag) | (tasks[-1].name != 'benchmark'):
            raise ValueError(f"fit transform function needs the last step to be benchmark estimator.")

        dfs = pd.DataFrame()
        params = {}
        for i in range(len(tasks) - 1):
            _transformer = tasks[i]
            if _transformer is None:
                print(f"Chronos+ skiped task {i} because transformer is None stype.")
                continue
            
            if i == 0:
                try:
                    dfs, params = _transformer.fit_transform(**params)
                except:
                    raise ValueError('Error at loading data step.')
            else:
                try:
                    dfs, params = _transformer.fit_transform(dfs, **params)
                except:
                    raise ValueError(f'error at step {i + 1}')
        
        _last = tasks[-1]
        if _last is None:
            raise ValueError(f"Chrono+ want to proceed with benchmark testing but the transformer is empty. (Why is this error possible?T_T) ")
        try:
            loss_memory, val_loss_memory = _last.fit_transform(dfs, **params)
        except:
            raise ValueError(f"Error at benchmark testing")
        return loss_memory, val_loss_memory 
    
    def prediction(self, test_filename, ae_list, params):
        self._validate_params(params)
        for_test_input = [
            LoadData(test_filename, params['dscol'], params['ycols'], params),
            FindFrequency(),
            AlignData(),
            Normalizer(),
            LeftPad(),
            FillGap(),
        ]
        test_inputs, _ = self.transform(for_test_input)
        for_test_times = [
            LoadData(test_filename, params['dscol'], params['ycols'], params),
            FindFrequency(),
            AlignData(),
            Normalizer(),
        ]
        test_times, _ = self.transform(for_test_times)
        assert len(test_inputs) == len(test_times) == len(ae_list)
        for i in range(len(test_inputs)):
            if i:
                current_pattern = 'weekends'
            else:
                current_pattern = 'weekdays'
            nona_test_times = test_times[i].dropna(how="all")
            for t in params[current_pattern]['time_index']:
                times = nona_test_times[nona_test_times ['time_index']==t].index
                generate_model_input = GenerateInput(times=times, offset=t)
                signals = generate_model_input.generate_model_input(test_inputs[i], current_pattern, **params)
                _signals = signals.reshape([signals.shape[0], -1])

                _signals = torch.from_numpy(signals)
                signals = Variable(_signals.float())
                code,preds = ae_list[i](signals)
                errors = np.sum(np.square((signals-preds).detach().numpy()), axis=1)
                test_times[i].loc[times, 'score'] = errors
        return test_times


    
        
