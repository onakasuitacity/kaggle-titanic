import numpy as np
from sklearn.model_selection import StratifiedKFold

class Runner:
    def __init__(self, name, model_cls, cv=True):
        self.name = name
        self.n_splits = 5
        self.models = []
        for fold in range(1 + cv * (self.n_splits - 1)):
            self.models.append(model_cls(f"{name}_{fold}"))


    def train(self, X, y, params, train_params={}):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=0)
        for model, (tr_idx, va_idx) in zip(self.models, skf.split(X, y)):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
            model.train(X_tr, y_tr, X_va, y_va, params, train_params)


    def predict(self, X):
        if len(self.models) > 1:
            return np.mean([model.predict(X) for model in self.models], axis=0)
        else:
            return self.models[0].predict(X)


    def get_score(self):
        if len(self.models) > 1:
            return np.mean([model.get_score() for model in self.models])
        else:
            return self.models[0].get_score()


    def save_model(self):
        any(model.save_model() for model in self.models)


    def load_model(self):
        any(model.load_model() for model in self.models)


    @property
    def model(self):
        return self.models[0]