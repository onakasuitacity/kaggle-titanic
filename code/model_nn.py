from model import AbstractModel
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.layers.advanced_activations import PReLU


class ModelNN(AbstractModel):
    def train(self, X_tr, y_tr, X_va, y_va, params, train_params):
        layers = params["layers"]
        dropout = params["dropout"]
        units = params["units"]
        
        model = Sequential()
        model.add(Dense(units, activation=PReLU(), input_dim=X_tr.shape[1]))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        for _ in range(layers - 2):
            model.add(Dense(units, activation=PReLU()))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics="acc")
        self.model = model
        self.result = self.model.fit(
            X_tr, y_tr, validation_data=(X_va, y_va),
            **train_params,
        )
        return self


    def predict(self, X):
        return self.model.predict(X).reshape(-1,)

    
    def get_score(self):
        return min(self.result.history["val_loss"])