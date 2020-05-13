import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from datetime import date, datetime
from sklearn.ensemble import RandomForestRegressor
from fastai.tabular import *

class InputData():
    def __init__(self, train_data_path, test_data_path, split_ratio=0.2):
        self._train_data_path = train_data_path
        self._test_data_path = test_data_path
        self._split_ratio = split_ratio
        np.random.seed(42)
        self._train_data = None
        self._eval_data = None
        self._test_data = None

    def _add_features(self, data):
        today = datetime(2018, 4, 15)

        data['birth_date'] = pd.to_datetime(data['birth_date'])
        data['age'] = (today - data['birth_date']).apply(lambda x: x.days) / 365.
        data['BMI'] = 10000. * data['weight_kg'] / (data['height_cm'] ** 2)
        data['is_gk'] = data['gk'] > 0

        positions = ['rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb', 'gk']

        data['best_pos'] = data[positions].max(axis=1)
        data['best_pos'] = data[positions].max(axis=1)

        return data

    def _split_data(self, data):
        if self._split_ratio <= 0:
            return data, None
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * self._split_ratio)

        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return data.iloc[train_indices], data.iloc[test_indices]

    def _load_train_data(self):
        df = pd.read_csv(self._train_data_path)
        data = self._add_features(df)
        self._train_data, self._eval_data = self._split_data(data)

    def _load_test_data(self):
        df = pd.read_csv(self._test_data_path)
        self._test_data = self._add_features(df)

    def train_data(self, is_gk):
        if self._train_data is not None:
            return self._train_data[self._train_data['is_gk'] == is_gk]
        self._load_train_data()

        return self._train_data[self._train_data['is_gk'] == is_gk]

    def eval_data(self, is_gk):
        if self._eval_data is not None:
            return self._eval_data[self._eval_data['is_gk'] == is_gk]

        self._load_train_data()
        return self._eval_data[self._eval_data['is_gk'] == is_gk]

    def test_data(self, is_gk):
        if self._test_data is not None:
            return self._test_data[self._test_data['is_gk'] == is_gk]
        self._load_test_data()
        return self._test_data[self._test_data['is_gk'] == is_gk]

    def all_test_data(self):
        if self._test_data is not None:
            return self._test_data
        self._load_test_data()
        return self._test_data

    def has_eval(self):
        return self._eval_data is not None


class Model():
    def __init__(self, model_builder, input_data: InputData, is_gk, features=None):
        self.model = model_builder()
        self.features = features
        self.input_data = input_data
        self.is_gk = is_gk

    def fit(self):
        train_data = self.input_data.train_data(self.is_gk)
        self.model.fit(train_data[self.features], train_data['y'])

    def eval(self):
        if self.input_data.has_eval():
            eval_data = self.input_data.eval_data(self.is_gk)
        else:
            eval_data = self.input_data.eval_data(self.is_gk)
        if eval_data is None:
            return None, None
        return self.model.predict(eval_data[self.features]), eval_data

    def test(self):
        test_data = self.input_data.test_data(self.is_gk)
        if test_data is None:
            return None, None
        return self.model.predict(test_data[self.features]), test_data


class SplitGKModel():
    def __init__(self, model_builder, input_data: InputData, gk_features=None, not_gk_features=None):
        self.gk_model = Model(model_builder, input_data, True, gk_features)
        self.not_gk_model = Model(model_builder, input_data, False, not_gk_features)
        self.input_data = input_data

    def fit(self):
        self.gk_model.fit()
        self.not_gk_model.fit()

    def evaluate(self):
        prediction1, eval_data1 = self.gk_model.eval()
        prediction2, eval_data2 = self.not_gk_model.eval()
        sum = np.abs(eval_data1['y'] - prediction1).sum() + np.abs(eval_data2['y'] - prediction2).sum()
        return sum / (len(eval_data1['y']) + len(eval_data2['y'])), prediction1, prediction2

    def test(self):
        prediction1, eval_data1 = self.gk_model.test()
        prediction2, eval_data2 = self.not_gk_model.test()
        return prediction1, prediction2


class Trainer():
    def __init__(self, model_builder, gk_features=None, not_gk_features=None, eval_ratio=0.2):
        self.train_file_path = '../dataset/fifa2018/train.csv'
        self.test_file_path = '../dataset/fifa2018/test.csv'
        self.submit_file_path = '../dataset/fifa2018/sample_submit.csv'
        self.prediction_file_path = '../dataset/fifa2018/prediction.csv'
        if not_gk_features is None:
            not_gk_features = gk_features

        self.input_data = InputData(self.train_file_path, self.test_file_path, eval_ratio)
        self.model = SplitGKModel(model_builder, self.input_data, gk_features, not_gk_features)

    def train(self):
        self.model.fit()
        result, gk_preds, not_gk_preds = self.model.evaluate()
        return result, gk_preds, not_gk_preds

    def train_and_test(self):
        self.model.fit()
        gk_preds, not_gk_preds = self.model.test()
        test = self.input_data.all_test_data()
        submit = pd.read_csv(self.submit_file_path)
        test.loc[test['is_gk'] == True, 'pred'] = gk_preds
        test.loc[test['is_gk'] == False, 'pred'] = not_gk_preds

        submit['y'] = np.array(test['pred'])
        submit.to_csv(self.prediction_file_path, index=False)


class RunnerBase():

    def _build(self):
        return None

    def _trainer(self, gk_features, not_gk_features=None, eval_ratio=0.2):
        return Trainer(self._build, gk_features, not_gk_features, eval_ratio)

    def train(self, gk_features, not_gk_features=None):
        trainer = self._trainer(gk_features, not_gk_features)
        print(trainer.train()[0])

    def train_and_test(self, gk_features, not_gk_features=None):
        self._trainer(gk_features, not_gk_features, 0).train_and_test()


class XGBoostRunner(RunnerBase):
    def __init__(self, max_depth=8):
        self.max_depth = max_depth

    def _build(self):
        return xgb.XGBRegressor(max_depth=self.max_depth, learning_rate=0.1, n_estimators=160, silent=False,
                                objective='reg:gamma')

def train1():
    XGBoostRunner().train(['height_cm', 'weight_kg', 'potential', 'BMI', 'pac',
                           'phy', 'international_reputation', 'age', 'best_pos'])
