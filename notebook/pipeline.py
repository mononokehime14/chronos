import numpy as np
import pandas as pd

class Pipeline():
    def __init__(self, tasks):
        self.tasks = tasks
    
    def fit_transform(self):
        _tasks = self.tasks

        if len(_tasks) == 0:
            raise ValueError("Empty progress list provided. Chronos+ will slack = =.")

        if _tasks[0].name != 'load_data':
            raise ValueError("Lack LoadData transformer which is necessary at the first of tasks list.")
        
        df = pd.DataFrame()
        params = {}
        for i in range(len(_tasks)):
            _transformer = _tasks[i]
            if _transformer is None:
                print(f"Chronos+ skiped task {i} because transformer is None stype.")
                continue
            
            if i == 0:
                try:
                    df = _transformer.fit_transform()
                except Exception as e:
                    raise ValueError(e)
            else:
                try:
                    df, params = _transformer.fit_transform(df, **params)
                except Exception as e:
                    print(f'error at step {i + 1}')
                    raise ValueError(e)
        
        return df, params
    
        
