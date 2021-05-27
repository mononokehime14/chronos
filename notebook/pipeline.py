import os
import numpy as np
import pandas as pd
import json


import torch
from torch.autograd import Variable


from preprocessing.preprocessor import LeftPad, LoadData, FindFrequency, PeriodDetect, AlignData, DropExtrema, Normalizer, FillGap, GenerateInput
from preprocessing.preprocessor import ParamsError


def load_params(filename):
    with open(filename, 'r') as file:
        params = json.load(file)
    return params

class Pipeline():
    def __init__(self, ae_list=None, params_location=None):
        self.ae_list = ae_list
        self.params_location = params_location


    def _validate_tasklist(self, tasks):
        """This function test given tasklist, see whether the last one is estimator

        Args:
            tasks (list): task list

        Raises:
            ValueError: if list is empty
            ValueError: if first transform is not load data
            ValueError: if both benchmark and training estimator is given
            ValueError: if benchmark or training estimator is not the last task

        Returns:
            Boolean: whether there is benchmark or training task in the tasks list
        """
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
        pattern_list = []
        if 'weekdays' in params:
            pattern_list.append('weekdays')
        if 'weekends' in params:
            pattern_list.append('weekends')

        for p in pattern_list:
            for _ in check_list:
                if _ not in params[p]:
                    raise ParamsError(f'Lack {_} value in params[{p}].')


    def fit_transform(self, tasks):
        """This function generate training dataset and then train AE models using training dataset.

        Args:
            tasks (list): tasks list

        Raises:
            ValueError: if there is no training task in task list or training step is not the last one
            ValueError: Any transformer is None type
            ValueError: Any error occur in pre-processing modules
            ValueError: The last estimator (training step) is None type
            ValueError: Error in training process
        """
        last_step_flag = self._validate_tasklist(tasks)
        if (not last_step_flag) | (tasks[-1].name != 'train_ae'):
            raise ValueError(f"fit transform function needs the last step to be train_ae estimator.")

        dfs = []
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
                    raise ValueError(f'error at step {_transformer.name}')

        _last = tasks[-1]
        if _last is None:
            raise ValueError(f"Chrono+ want to proceed with training AE but the transformer is empty. (Why is this error possible?T_T) ")
        try:
            self.ae_list, self.params_location = _last.fit_transform(dfs, **params)
        except:
            raise ValueError(f"Error at training AE")
    
    def transform(self, tasks):
        """This function simply generates training dataset using tasks list.
        It goes through preprocessing modules.

        Args:
            tasks (list): tasks list

        Raises:
            ValueError: if there is estimator like training task or benchmark testing task in tasks list
            ValueError: Any transformer is None type
            ValueError: Any error occur in any pre processing transformer

        Returns:
            list of dataframes: training set for weekdays and weekends
            params: information like period, anchors that are needed in training, benchmark testing and prediction
        """
        last_step_flag = self._validate_tasklist(tasks)
        if last_step_flag:
            raise ValueError(f"fit transform function needs the last step to be train_ae estimator.")

        dfs = []
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
        """This function generate training input and then goes through benchmark testing:
        It will try different combinations of hyperparameters and record loss and validation loss so that
        we can decide what hyperparameter set is the best suitable one.

        Args:
            tasks (list): tasks list

        Raises:
            ValueError: if last estimator is not benchmark testing
            ValueError: any transformer is none type
            ValueError: the benchmark testing estimator is none type

        Returns:
            loss_memory: recorded training loss of each combination
            val_loss_memory: recorded validation loss of each combination
        """
        last_step_flag = self._validate_tasklist(tasks)
        if (not last_step_flag) | (tasks[-1].name != 'benchmark'):
            raise ValueError(f"fit transform function needs the last step to be benchmark estimator.")

        dfs = []
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
    
    def prediction(self, test_filename):
        """This function predicts on testing dataset using training models (saved after fit_transform)
         and other information (generated after fit_transform)

        Args:
            test_filename (string): file path to test data

        Raises:
            ValueError: if there is no models and params saved (if we have not fit_transform to train before)

        Returns:
            test_times: Loss values (MSE between real data and prediction) in testing data's time scale
        """
        if (self.params_location is None) | (self.ae_list is None):
            raise ValueError(f"Prediction needs params and AE trained by training step.")
        #params = load_params(self.params_location) 
        #params = self.params_location
        self._validate_params(self.params_location)
        for_test_input = [
            LoadData(test_filename, self.params_location['dscol'], self.params_location['ycols'], predefined_params = self.params_location),
            FindFrequency(),
            AlignData(),
            Normalizer(),
            LeftPad(),
            FillGap(),
        ]
        test_inputs, _ = self.transform(for_test_input)
        for_test_times = [
            LoadData(test_filename, self.params_location['dscol'], self.params_location['ycols'], predefined_params = self.params_location),
            FindFrequency(),
            AlignData(),
            Normalizer(),
        ]
        test_times, _ = self.transform(for_test_times)
        assert len(test_inputs) == len(test_times) == len(self.ae_list)
        for i in range(len(test_inputs)):
            if i:
                current_pattern = 'weekends'
            else:
                current_pattern = 'weekdays'
            nona_test_times = test_times[i].dropna(how="all")
            _ae = torch.load(self.ae_list[i])
            _test_results = test_times[i]
            _ae.eval()
            for t in self.params_location[current_pattern]['time_index']:
                times = nona_test_times[nona_test_times ['time_index']==t].index
                generate_model_input = GenerateInput(times=times, offset=t)
                signals = generate_model_input.generate_model_input(test_inputs[i], current_pattern, self.params_location)
                _signals = signals.reshape([signals.shape[0], -1])

                _signals = torch.from_numpy(signals)
                signals = Variable(_signals.float())
                code,preds = _ae(signals)
                errors = np.sum(np.square((signals-preds).detach().numpy()), axis=1)
                _test_results.loc[times, 'score'] = errors
            test_times[i] = _test_results
        return test_times


    
        
