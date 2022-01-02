from model import AbstractModel
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential


class ModelNN(AbstractModel):
    def train(self, X_tr, y_tr, X_va, y_va, params, train_params):
        units_list = params["units_list"]
        dropout = params["dropout"]
        num_classes = params["num_classes"]
        
        model = Sequential()
        model.add(Dense(units_list[0], kernel_initializer="lecun_normal", activation="selu", input_dim=X_tr.shape[1]))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        for units in units_list[1:]:
            model.add(Dense(units, kernel_initializer="lecun_normal", activation="selu"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        model.add(Dense(num_classes, activation="softmax"))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model = model
        self.result = self.model.fit(
            X_tr, y_tr, validation_data=(X_va, y_va),
            **train_params,
        )
        return self


    def predict(self, X):
        return self.model.predict(X)

    
    def get_score(self):
        return min(self.result.history["val_loss"])