import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import glob

file_paths = [
    "data/Goldman Sachs_stock_data.csv",
    "data/Johnson and Johnson_stock_data.csv",
    "data/JP Morgan Chase and Co_stock_data.csv",
    "data/Nike_stock_data.csv",
    "data/Pfizer_stock_data.csv"
]

def process_stock_data(file_path):
    df = pd.read_csv(file_path, skiprows=2)
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df['Ticker'] = file_path.split('/')[-1].split('_')[0]

    df['H-L'] = df['High'] - df['Low']
    df['O-C'] = df['Open'] - df['Close']
    df['7D_MA'] = df['Close'].rolling(window=7).mean()
    df['14D_MA'] = df['Close'].rolling(window=14).mean()
    df['21D_MA'] = df['Close'].rolling(window=21).mean()
    df['7D_STD'] = df['Close'].rolling(window=7).std()

    df.dropna(inplace=True)

    output_file = file_path.replace(".csv", "_processed.csv")
    df.to_csv(output_file, index=False)
    print(f"Processed file saved: {output_file}")

    return df


dataframes = [process_stock_data(file) for file in file_paths]


class MyGBMRegressor:


    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.trees_ = []
        self.F0_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        # initial
        self.F0_ = np.mean(y)
        current_pred = np.full_like(y, self.F0_, dtype=float)
        # iteration
        for i in range(self.n_estimators):
            residual = y - current_pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X, residual)
            pred_residual = tree.predict(X)
            current_pred += self.learning_rate * pred_residual
            self.trees_.append(tree)

    def predict(self, X):
        X = np.array(X)
        y_pred = np.full((X.shape[0],), self.F0_, dtype=float)
        for tree in self.trees_:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

file_list = [
    "Goldman Sachs_stock_data_processed.csv",
    "Johnson and Johnson_stock_data_processed.csv",
    "JP Morgan Chase and Co_stock_data_processed.csv",
    "Nike_stock_data_processed.csv",
    "Pfizer_stock_data_processed.csv"
]

stocks = ['Goldman Sachs ', 'JNJ ', 'JP Morgan ', 'Nike ', 'Pfizer ']
method = '[GBM]'

for idx, file in enumerate(file_list):
    df = pd.read_csv("data/" + file)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df.dropna(inplace=True)

    df["Close_next"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    features = ["H-L", "O-C", "7D_MA", "14D_MA", "21D_MA", "7D_STD"]
    target = "Close_next"
    X = df[features]
    y = df[target]

    init_train_size = int(len(df) * 0.8)

    rolling_predictions = []
    rolling_actual = []
    rolling_dates = []

    final_model = None

    for i in range(init_train_size, len(df) - 1):
        X_train_current = X.iloc[:i]
        y_train_current = y.iloc[:i]
        X_test_current = X.iloc[i:i + 1]
        y_test_current = y.iloc[i]
        date_current = df["Date"].iloc[i + 1]

        model = MyGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
        model.fit(X_train_current, y_train_current)
        final_model = model

        pred = model.predict(X_test_current)[0]
        rolling_predictions.append(pred)
        rolling_actual.append(y_test_current)
        rolling_dates.append(date_current)

    # save model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{stocks[idx].strip().replace(' ', '_')}_FinalGBM.pkl")
    joblib.dump(final_model, model_path)
    print(f"[saved] {model_path}")

    rolling_predictions = np.array(rolling_predictions)
    rolling_actual = np.array(rolling_actual)
    rolling_dates = pd.Series(rolling_dates)

    test_rmse = np.sqrt(mean_squared_error(rolling_actual, rolling_predictions))
    test_mape = mean_absolute_percentage_error(rolling_actual, rolling_predictions) * 100
    test_mbe = (rolling_actual - rolling_predictions).mean()

    stock_name = file.replace("_stock_data_processed.csv", "")
    print(f"===== {stock_name} =====")
    print("The test set rolls the prediction results:")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAPE: {test_mape:.4f}%")
    print(f"MBE: {test_mbe:.4f}")

    plot_data = pd.DataFrame({
        'Date': rolling_dates,
        'Original Closing Price': rolling_actual,
        'Predicted Closing Price': rolling_predictions
    })

    plt.figure(figsize=(10, 6))
    plt.plot(plot_data['Date'], plot_data['Original Closing Price'], 's-', label='Original Closing Price',
             color='blue', linewidth=1, markersize=3)
    plt.plot(plot_data['Date'], plot_data['Predicted Closing Price'], 'o-', label='Predicted Closing Price',
             color='red', linewidth=1, markersize=3)

    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Original vs Predicted Closing Prices')
    plt.legend()

    dates_to_plot = plot_data['Date'][::125].tolist()
    last_date = plot_data['Date'].iloc[-1]
    if last_date not in dates_to_plot:
        dates_to_plot.append(last_date)
    plt.xticks(dates_to_plot, rotation=0)

    plt.text(0.95, 0.05, stocks[idx] + method,
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=plt.gca().transAxes,
             fontsize=14)

    output_filename = './pictures/' + stocks[idx] + '_RollingGBM.png'
    plt.savefig(output_filename)
    plt.close()
    print(f"pictures saved as '{output_filename}'\n")

