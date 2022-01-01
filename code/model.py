import os
import joblib
from abc import ABCMeta, abstractmethod


class AbstractModel(metaclass=ABCMeta):
    def __init__(self, name_fold):
        self.name_fold = name_fold
        self.model = None


    @abstractmethod
    def train(self, X_tr, y_tr, X_va, y_va, params, train_params):
        pass


    @abstractmethod
    def predict(self, X):
        pass


    @abstractmethod
    def get_score(self):
        pass


    def save_model(self):
        path = f"../model/model/{self.name_fold}.model"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path, compress=True)


    def load_model(self):
        path = f"../model/model/{self.name_fold}.model"
        self.model = joblib.load(path)