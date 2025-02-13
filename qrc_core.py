import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

class QRC:
    """
    Base class for Quantum Reservoir Computing.
    """
    def __init__(self, n_sites, readouts=None, scaler=None):
        """
        Initializes the QRC.

        Args:
            n_sites (int): Number of qubits/spins in the reservoir.
            readouts (list): List of readout operators.
            scaler (sklearn.preprocessing): Scaler to use for data normalization. If None, MinMaxScaler is used
        """
        self.n_sites = n_sites
        self.readouts = readouts
        if self.readouts is None:
            self.readouts = self.generate_readouts(n_sites)
        self.scaler = scaler if scaler is not None else MinMaxScaler(feature_range=(-1, 1))


    def generate_readouts(self, n_sites):
        """
        Generates a default set of readout operators.

        Override this method in subclasses to provide different readout strategies.
        """
        # Default: Measure each qubit in the Z basis
        readouts = [f"Z{i}" for i in range(n_sites)]
        return readouts

    def encode(self, data, *args, **kwargs):
        """
        Encodes classical data into a quantum state (Bloqade state).

        Override this method in subclasses.
        """
        raise NotImplementedError("Encoding method must be implemented in a subclass.")

    def apply_dynamics(self, encoded_state, *args, **kwargs):
        """
        Applies the dynamics (time evolution) to the quantum state.

        Override this method in subclasses.
        """
        raise NotImplementedError("Dynamics method must be implemented in a subclass.")

    def measure(self, evolved_state, *args, **kwargs):
        """
        Performs measurements on the evolved quantum state to extract features.

        Override this method in subclasses.
        """
        raise NotImplementedError("Measurement method must be implemented in a subclass.")

    def predict(self, train_data, train_labels, test_data):
        """
        Trains a classical machine learning model and makes predictions.
        """
        model = RandomForestRegressor(n_estimators=50, random_state=402)  # Example model
        model.fit(train_data, train_labels)
        predictions = model.predict(test_data)
        return predictions

    def train_test_split(self,X,y,test_size,random_state):

      """
      Splits the dataset into training and testing sets.

      Args:
        X: input data
        y: output data
        test_size: size of test dataset
        random_state: random state for consistent splitting
      """
      # First, split into 60% train and 40% temp (test + val)
      X_train, X_temp, y_train, y_temp = train_test_split(
          X, y, test_size=test_size, shuffle=True, random_state=random_state
      )
      # Next, split the remaining 40% equally into test (20%) and validation (20%)
      X_test, X_val, y_test, y_temp = train_test_split(
          X_temp, y_temp, test_size=0.5, shuffle=True, random_state=random_state
      )
      return X_train, X_val, X_test, y_train, y_val, y_test

    def normalize(self, X, y):
      """
      Normalizes dataset using MixMaxScaler
      """
      X_normalized = self.scaler.fit_transform(X)

      y = y.reshape(-1, 1)  # Reshape y to a 2D array for normalization

      scaler_y = MinMaxScaler(feature_range=(0, 1))

      y_normalized = scaler_y.fit_transform(y).flatten()

      return X_normalized, y_normalized

    def evaluate(self, predictions, true_values):
        """
        Evaluates the performance of the model.
        """
        mse = mean_squared_error(true_values, predictions)
        print(f"Mean Squared Error: {mse}")
        return mse

    def run(self, X, y, test_size=0.4, random_state=402):
        """
        Executes the complete QRC workflow.

        Args:
            X: Input data (classical).
            y: Output data (classical labels).

        Returns:
            Mean Squared Error on the test set.
        """
        # Normalization
        X_normalized, y_normalized = self.normalize(X,y)
        # Split Data
        X_train, X_val, X_test, y_train, y_val, y_test = self.train_test_split(X_normalized, y_normalized, test_size, random_state)

        # Encode
        encoded_train = self.encode(X_train)
        encoded_test = self.encode(X_test)
        # Evolve
        evolved_train = self.apply_dynamics(encoded_train)
        evolved_test = self.apply_dynamics(encoded_test)

        # Measure
        train_features = self.measure(evolved_train)
        test_features = self.measure(evolved_test)

        # Predict
        predictions = self.predict(train_features, y_train, test_features)

        # Evaluate
        mse = self.evaluate(predictions, y_test)
        return mse