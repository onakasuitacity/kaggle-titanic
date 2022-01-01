import pandas as pd
import lightgbm as lgb
from model import AbstractModel

class ModelLGB(AbstractModel):
    def train(self, X_tr, y_tr, X_va, y_va, params, train_params):
        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_val = lgb.Dataset(X_va, y_va, reference=lgb_train)
        self.model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            **train_params
        )
        return self


    def predict(self, X):
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    
    def get_score(self):
        return list(self.model.best_score["valid_1"].values())[0]


    @property
    def feature_importance_(self):
        return pd.Series(
            self.model.feature_importance(importance_type="gain"),
            self.model.feature_name()
        ).sort_values(ascending=False)