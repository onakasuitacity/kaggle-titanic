import pandas as pd
import xgboost as xgb
from model import AbstractModel

class ModelXGB(AbstractModel):
    def train(self, X_tr, y_tr, X_va, y_va, params, train_params):
        dtrain = xgb.DMatrix(X_tr, y_tr)
        dvalid = xgb.DMatrix(X_va, y_va)
        self.model = xgb.train(
            params,
            dtrain,
            evals=[(dtrain, "train"), (dvalid, "eval")],
            **train_params
        )
        return self


    def predict(self, X):
        return self.model.predict(xgb.DMatrix(X), ntree_limit=self.model.best_ntree_limit)

    
    def get_score(self):
        return self.model.best_score