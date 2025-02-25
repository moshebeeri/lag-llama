import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
from huggingface_hub import hf_hub_download
from lag_llama.model.module import LagLlamaModel
from gluonts.torch.distributions import StudentTOutput
from gluonts.time_feature import time_features_from_frequency_str, get_lags_for_frequency
from gluonts.torch.util import lagged_sequence_values
import matplotlib.pyplot as plt
import subprocess
import os
import time
from torch.distributions import Normal
import seaborn as sns
import math
from gluonts.time_feature.holiday import SPECIAL_DATE_FEATURES
import torch.nn.functional as F
import traceback
from pathlib import Path
import matplotlib.dates as mdates
import csv
import argparse

# Monkey-patch torch.distributions.StudentT to define a custom expand method
from torch.distributions.studentT import StudentT as TorchStudentT

def studentt_expand(self, batch_shape, _instance=None):
    new_loc = self.loc.expand(batch_shape)
    new_scale = self.scale.expand(batch_shape)
    return self.__class__(self.df, new_loc, new_scale, validate_args=self._validate_args)

TorchStudentT.expand = studentt_expand

# End of monkey patch

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Top 10 Tech Companies (by market cap as of 2024) plus NASDAQ-100 index
TECH_TICKERS = [
    'NDX',   # NASDAQ-100 Index (without the ^ symbol)
    'AAPL',  # Apple
    'MSFT',  # Microsoft
    'GOOGL', # Alphabet
    'AMZN',  # Amazon
    'NVDA',  # NVIDIA
    'META',  # Meta
    'TSM',   # Taiwan Semiconductor
    'AVGO',  # Broadcom
    'ASML',  # ASML Holding
    'AMD'    # AMD
]

PREDICTION_LENGTH = 32

# Constants for prediction horizons
PREDICTION_HORIZONS = {
    '2d': 2,
    '1w': 5,  # business week
    '1m': 21  # business month
}

def create_time_features(idx):
    """Create time features for a given index."""
    # Create a simple time feature vector
    # This is a placeholder - you can expand this with more sophisticated time features
    time_feat = torch.zeros(6, dtype=torch.float32)
    time_feat[0] = math.sin(2 * math.pi * idx / 24)  # Hour of day
    time_feat[1] = math.cos(2 * math.pi * idx / 24)
    time_feat[2] = math.sin(2 * math.pi * idx / 7)   # Day of week
    time_feat[3] = math.cos(2 * math.pi * idx / 7)
    time_feat[4] = math.sin(2 * math.pi * idx / 365) # Day of year
    time_feat[5] = math.cos(2 * math.pi * idx / 365)
    return time_feat

def get_historical_data(tickers=TECH_TICKERS, days=3000, production_mode=False):
    """Fetch historical data for given tickers with local caching."""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    cache_file = "data/stock_data_cache.csv"
    
    # Use UTC timezone for consistency
    end_date = pd.Timestamp.now(tz='UTC').normalize()
    start_date = end_date - pd.Timedelta(days=days)
    
    # Try to load cached data
    cached_data = None
    if os.path.exists(cache_file):
        try:
            cached_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            cached_data.index = pd.to_datetime(cached_data.index, utc=True)  # Convert to UTC
            print("Found cached data")
            
            # Check if we need to update the cache
            if len(cached_data) > 0:
                last_date = cached_data.index[-1]  # Already in UTC
                days_since_update = (end_date - last_date).days
                
                # Production mode: only fetch if there's a new trading day
                if production_mode:
                    if days_since_update < 1 or not is_trading_day(last_date + pd.Timedelta(days=1)):
                        print("Using cached data (no new trading day)")
                        return cached_data
                    print("New trading day detected, fetching latest data...")
                # Development mode: use cache if less than 1 day old
                else:
                    if days_since_update < 1:
                        print("Using cached data (less than 1 day old)")
                        return cached_data
                
                # Update start date to only fetch new data
                start_date = last_date + pd.Timedelta(days=1)
                if start_date >= end_date:
                    print("Cache is up to date")
                    return cached_data
                print(f"Fetching {days_since_update} days of new data")
        except Exception as e:
            print(f"Error reading cache: {e}")
            cached_data = None
    
    # Fetch data for each ticker
    data = {}
    for ticker in tickers:
        try:
            # Add ^ prefix for indices
            yf_ticker = f"^{ticker}" if ticker == 'NDX' else ticker
            stock = yf.Ticker(yf_ticker)
            hist = stock.history(start=start_date, end=end_date)
            # Convert timezone-aware dates to UTC
            hist.index = hist.index.tz_convert('UTC')
            data[ticker] = hist['Close']  # Store with original ticker (without ^)
            print(f"Fetched {len(hist)} days of data for {ticker}")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    # Convert to DataFrame
    new_data = pd.DataFrame(data)
    
    # If we have cached data, combine it with new data
    if cached_data is not None:
        # Combine old and new data
        combined_data = pd.concat([cached_data, new_data])
        # Remove duplicates keeping the latest value
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        # Sort by date
        combined_data = combined_data.sort_index()
        new_data = combined_data
    
    # Save to cache
    new_data.to_csv(cache_file)
    print(f"Saved data to cache: {cache_file}")
    
    return new_data

def validate_dimensions(past_target, past_observed_values, past_time_feat, future_time_feat=None):
    """Validate dimensions of input tensors."""
    # Basic dimension checks
    assert len(past_target.shape) == 3, f"past_target should be 3D [batch, seq_len, feature], got shape {past_target.shape}"
    assert len(past_observed_values.shape) == 3, f"past_observed_values should be 3D [batch, seq_len, feature], got shape {past_observed_values.shape}"
    assert len(past_time_feat.shape) == 3, f"past_time_feat should be 3D [batch, seq_len, time_features], got shape {past_time_feat.shape}"
    
    # Check matching dimensions
    assert past_target.shape[0] == past_observed_values.shape[0], f"Batch sizes don't match: past_target {past_target.shape[0]} vs past_observed_values {past_observed_values.shape[0]}"
    assert past_target.shape[1] == past_observed_values.shape[1], f"Sequence lengths don't match: past_target {past_target.shape[1]} vs past_observed_values {past_observed_values.shape[1]}"
    assert past_target.shape[1] == past_time_feat.shape[1], f"Time feature sequence length doesn't match: past_target {past_target.shape[1]} vs past_time_feat {past_time_feat.shape[1]}"
    
    if future_time_feat is not None:
        assert len(future_time_feat.shape) == 3, f"future_time_feat should be 3D [batch, seq_len, time_features], got shape {future_time_feat.shape}"
        assert future_time_feat.shape[0] == past_target.shape[0], f"Future time features batch size doesn't match: future_time_feat {future_time_feat.shape[0]} vs past_target {past_target.shape[0]}"
        assert future_time_feat.shape[2] == past_time_feat.shape[2], f"Future time features dimension doesn't match: future_time_feat {future_time_feat.shape[2]} vs past_time_feat {past_time_feat.shape[2]}"
        assert future_time_feat.shape[1] == past_time_feat.shape[1], f"Future time features sequence length doesn't match: future_time_feat {future_time_feat.shape[1]} vs past_time_feat {past_time_feat.shape[1]}"

def prepare_data_for_lagllama(data, dates, model, horizon_days=2, total_required=752):
    """
    Prepare data for Lag-Llama model using returns instead of raw prices.
    
    Args:
        data: Historical stock data (pd.Series)
        dates: Index of dates
        model: LagLlama model instance
        horizon_days: Number of days to forecast
        total_required: Minimum number of data points needed
    """
    # Ensure we have enough data
    if len(data) < total_required + 1:  # Need extra point for first return
        raise ValueError(f"Not enough data points. Need {total_required + 1}, but got {len(data)}")
    
    # Calculate returns
    returns = np.diff(data) / data[:-1]
    
    # Get the required number of returns
    returns = returns[-(total_required):]
    dates = dates[-(total_required):]
    
    # Create future dates for the specific horizon
    future_dates = pd.date_range(
        start=dates[-1] + pd.Timedelta(days=1),
        periods=horizon_days,
        freq='B'
    )
    
    # Create time features
    def create_time_features(dates):
        features = []
        # Hour of day
        hours = dates.hour
        features.append(np.sin(2 * np.pi * hours / 24))
        features.append(np.cos(2 * np.pi * hours / 24))
        
        # Day of week
        days = dates.dayofweek
        features.append(np.sin(2 * np.pi * days / 7))
        features.append(np.cos(2 * np.pi * days / 7))
        
        # Day of year
        day_of_year = dates.dayofyear - 1
        features.append(np.sin(2 * np.pi * day_of_year / 365.25))
        features.append(np.cos(2 * np.pi * day_of_year / 365.25))
        
        return np.stack(features, axis=1)
    
    # Create past and future time features
    past_time_feat = create_time_features(dates)
    future_time_feat = create_time_features(future_dates)
    
    # Convert to tensors and add batch and feature dimensions
    past_target = torch.tensor(returns, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
    past_time_feat = torch.tensor(past_time_feat, dtype=torch.float32).unsqueeze(0)     # [1, seq_len, 6]
    
    # Create future time features with same sequence length as past
    padded_future_time_feat = np.zeros((total_required, 6))  # Initialize with zeros
    padded_future_time_feat[:horizon_days] = future_time_feat  # Fill first horizon_days with actual features
    future_time_feat = torch.tensor(padded_future_time_feat, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, 6]
    
    # Create past observed values tensor (all 1s since we're using clean data)
    past_observed_values = torch.ones_like(past_target)
    
    # Move tensors to the correct device
    past_target = past_target.to(DEVICE)
    past_time_feat = past_time_feat.to(DEVICE)
    future_time_feat = future_time_feat.to(DEVICE)
    past_observed_values = past_observed_values.to(DEVICE)
    
    # Validate tensor dimensions
    validate_dimensions(past_target, past_observed_values, past_time_feat, future_time_feat)
    
    return past_target, past_observed_values, past_time_feat, future_time_feat

def load_lagllama_model(context_length=128):
    """Load the Lag-Llama model."""
    global lag_indices  # Make lag_indices available globally
    
    # Get lags for all frequencies
    lag_indices = []
    for freq in ['Q', 'M', 'W', 'D', 'H', 'T', 'S']:
        curr_lags = get_lags_for_frequency(freq_str=freq, num_default_lags=1)
        lag_indices.extend(curr_lags)
    lag_indices = sorted(list(set(lag_indices)))
    lag_indices = [lag - 1 for lag in lag_indices]  # Convert to 0-based indexing
    
    # Ensure we use exactly 77 lags to match the model's feature size (77 + 1 + 6 + 6 + 2 = 92)
    lag_indices = lag_indices[:77]
    
    print(f"Total unique lags (0-based): {lag_indices}")
    print(f"Number of lags: {len(lag_indices)}")
    
    # Model parameters
    model = LagLlamaModel(
        context_length=context_length,
        max_context_length=2048,
        scaling="mean",
        input_size=1,
        n_layer=8,
        n_embd_per_head=16,
        n_head=9,
        lags_seq=lag_indices,
        distr_output=StudentTOutput(),
        num_parallel_samples=100,
        time_feat=True,
        dropout=0.0,
        feature_size=92,  # Match the checkpoint's feature size
        rope_scaling={"type": "linear", "factor": max(1.0, context_length / 32)}  # Enable RoPE scaling
    ).to(DEVICE)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    model_path = "models/lag-llama.ckpt"
    
    # Download the model if it doesn't exist
    if not os.path.exists(model_path):
        print("Downloading model weights...")
        subprocess.run(["huggingface-cli", "download", "time-series-foundation-models/Lag-Llama", "lag-llama.ckpt", "--local-dir", "models"])
    
    print(f"Loading model from {model_path}")
    state_dict = torch.load(model_path, map_location=DEVICE)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # Create a new state dict with the correct keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    
    # Load the state dict
    model.load_state_dict(new_state_dict)
    return model

def load_prediction_history():
    """Load historical predictions from CSV file."""
    history_file = "data/prediction_history.csv"
    if not os.path.exists(history_file):
        return pd.DataFrame()
    return pd.read_csv(history_file, parse_dates=['prediction_date', 'target_date'])

def save_prediction(ticker: str, prediction_date: datetime, target_date: datetime, 
                   predicted_value: float, horizon: str, actual_value: float = None):
    """Save prediction to CSV file with tracking information."""
    history_file = "data/prediction_history.csv"
    os.makedirs("data", exist_ok=True)
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(history_file):
        with open(history_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ticker', 'prediction_date', 'target_date', 'horizon',
                           'predicted_value', 'actual_value', 'accuracy_pct'])
    
    # Append new prediction
    with open(history_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            ticker,
            prediction_date.strftime('%Y-%m-%d'),
            target_date.strftime('%Y-%m-%d'),
            horizon,
            predicted_value,
            actual_value if actual_value is not None else '',
            ((actual_value - predicted_value) / predicted_value * 100) if actual_value is not None else ''
        ])

def update_prediction_accuracy():
    """Update accuracy of past predictions using current market data."""
    df = load_prediction_history()
    if df.empty:
        return
    
    # Get current market data for validation
    current_data = get_historical_data(tickers=TECH_TICKERS)
    
    # Update predictions where we now have actual values
    updated_rows = []
    for _, row in df.iterrows():
        if pd.isna(row['actual_value']) and row['target_date'] <= pd.Timestamp.now():
            ticker_data = current_data[row['ticker']]
            target_date = pd.Timestamp(row['target_date'])
            if target_date in ticker_data.index:
                actual_value = ticker_data[target_date]
                accuracy_pct = (actual_value - row['predicted_value']) / row['predicted_value'] * 100
                row['actual_value'] = actual_value
                row['accuracy_pct'] = accuracy_pct
        updated_rows.append(row)
    
    # Save updated predictions
    pd.DataFrame(updated_rows).to_csv("data/prediction_history.csv", index=False)

def forecast_stock(model, data, dates):
    """
    Generate forecasts for a stock using the LagLlama model.
    
    Args:
        model: The trained LagLlama model
        data: Historical stock data (pd.Series)
        dates: Forecast dates
    
    Returns:
        dict: Dictionary containing forecast information
    """
    # Get the last historical value for converting returns back to prices
    last_value = float(data.iloc[-1])
    print(f"Last historical value: {last_value:.2f}")
    
    # Calculate historical volatility and momentum using returns
    recent_returns = np.diff(data[-30:].values) / data[-30:-1].values
    daily_std = np.std(recent_returns)
    momentum = np.mean(recent_returns[-5:])  # Use last 5 days for momentum
    avg_return = np.mean(recent_returns)     # Average return over 30 days
    print(f"Historical 30-day volatility: {daily_std*100:.2f}%")
    print(f"5-day momentum: {momentum*100:.2f}%")
    print(f"30-day avg return: {avg_return*100:.2f}%")
    
    forecasts = {}
    current_date = pd.Timestamp.now()
    
    # Generate forecasts for each horizon
    for horizon_name, horizon_days in PREDICTION_HORIZONS.items():
        print(f"\nGenerating {horizon_name} forecast...")
        
        # Prepare data specifically for this horizon
        past_target, past_observed_values, past_time_feat, future_time_feat = prepare_data_for_lagllama(
            data, dates, model, horizon_days=horizon_days
        )
        
        with torch.no_grad():
            # Reset model's cache for each new horizon
            model.reset_cache()
            
            # Generate predictions
            transformer_input, loc, scale = model.prepare_input(
                past_target=past_target,
                past_time_feat=past_time_feat,
                future_time_feat=future_time_feat,
                past_observed_values=past_observed_values
            )
            
            # Forward pass through the model
            x = model.transformer.wte(transformer_input)
            for block in model.transformer.h:
                x = block(x, use_kv_cache=False)
            x = model.transformer.ln_f(x)
            params = model.param_proj(x)
            
            # Create distribution from parameters
            distr = model.distr_output.distribution(params, loc, scale)
            
            # Get predicted returns
            predicted_returns = distr.mean.cpu().numpy().squeeze()
            
            # Convert returns to price forecasts
            forecast_values = np.zeros(horizon_days)
            prev_value = last_value
            
            for i in range(horizon_days):
                # Get the predicted return and apply volatility-based adjustment
                predicted_return = predicted_returns[i]
                
                # Apply volatility-based scaling to the predicted return
                volatility_scale = np.clip(daily_std / 0.02, 0.5, 2.0)  # Scale based on historical volatility
                predicted_return = predicted_return / volatility_scale
                
                # Calculate the price prediction
                forecast_values[i] = prev_value * (1 + predicted_return)
                prev_value = forecast_values[i]
                
                # Save prediction to history
                target_date = current_date + pd.Timedelta(days=i+1)
                save_prediction(
                    ticker=data.name,
                    prediction_date=current_date,
                    target_date=target_date,
                    predicted_value=forecast_values[i],
                    horizon=horizon_name
                )
            
            forecasts[horizon_name] = forecast_values
            print(f"{horizon_name} Forecast values:")
            for i, value in enumerate(forecast_values):
                print(f"Day {i+1}: {value:.2f} ({(value/last_value - 1)*100:+.2f}%)")
    
    return {
        'last_historical_value': last_value,
        'forecasts': forecasts,
        'volatility': daily_std,
        'momentum': momentum,
        'avg_return': avg_return
    }

def test_lagllama_dimensions(model, lag_indices):
    """
    Test Lag-Llama model with different sequence lengths using simulated data.
    """
    print("\nTesting Lag-Llama dimensions with simulated data...")
    print(f"Total unique lags (0-based): {lag_indices}")
    print(f"Number of lags: {len(lag_indices)}")
    
    # Model configuration
    max_lag = max(lag_indices) + 1
    context_length = 32  # Fixed context length
    total_required = context_length + max_lag
    
    print("\nModel configuration:")
    print(f"Maximum lag: {max_lag}")
    print(f"Context length: {context_length}")
    print(f"Total required points (context + max_lag): {total_required}")
    
    # Test with different sequence lengths
    for n_points in [1200, 1500, 2000]:
        print(f"\nTesting with {n_points} points...")
        print(f"Available points in synthetic data: {n_points}")
        print(f"Lag indices: {lag_indices}")
        
        try:
            # Generate synthetic data
            dates = pd.date_range(start='2020-01-01', periods=n_points, freq='D')
            data = pd.Series(
                np.random.randn(n_points).cumsum(),
                index=dates,
                name='Close'
            )
            
            # Prepare data ensuring we have enough points for the maximum lag
            past_target, past_time_feat, future_time_feat = prepare_data_for_lagllama(
                data, dates, model, total_required
            )
            
            print("Input shapes:")
            print(f"- past_target: {past_target.shape}")
            print(f"- past_time_feat: {past_time_feat.shape}")
            print(f"- future_time_feat: {future_time_feat.shape}")
            
            # Test model forward pass
            print("Testing model forward pass...")
            with torch.no_grad():
                params, loc, scale = model(
                    past_target=past_target,
                    past_time_feat=past_time_feat,
                    future_time_feat=future_time_feat
                )
                print("Model forward pass successful")
                print(f"- params shape: {[p.shape for p in params]}")
                print(f"- loc shape: {loc.shape}")
                print(f"- scale shape: {scale.shape}")
                
                # Create distribution
                print("Creating distribution...")
                distr = model.distr_output.distribution(params, loc, scale)
                print("Distribution created successfully")
                
                # Get mean predictions
                print("Getting mean predictions...")
                forecast_means = distr.mean.cpu().numpy().squeeze()
                print(f"Mean predictions shape: {forecast_means.shape}")
                print(f"First few predictions: {forecast_means[:5]}")
            
            print(f"Test successful with {n_points} points")
            
        except Exception as e:
            print(f"Failed with {n_points} points: {str(e)}")
            traceback.print_exc()

def save_forecasts(ticker, forecasts):
    """Save forecasts to a CSV file."""
    # Convert forecasts to a numpy array and flatten to 1D
    forecast_values = forecasts[0].cpu().numpy().flatten()
    
    # Create dates for the forecast period
    forecast_dates = pd.date_range(
        start=pd.Timestamp.now(),
        periods=len(forecast_values),
        freq='S'
    )
    
    # Create DataFrame with forecasts
    forecast_df = pd.DataFrame({
        'ticker': ticker,
        'date': forecast_dates,
        'forecast': forecast_values
    })
    
    # Save to CSV
    output_file = f'forecasts_{ticker}.csv'
    forecast_df.to_csv(output_file, index=False)
    print(f"Saved forecasts for {ticker} to {output_file}")

def is_trading_day(date):
    """Check if a given date is a trading day (weekday and not a holiday)."""
    return bool(len(yf.download('^GSPC', start=date, end=date + pd.Timedelta(days=1), progress=False)))

def check_new_trading_day():
    """Check if there's a new trading day's data available."""
    cache_file = "data/stock_data_cache.csv"
    
    if not os.path.exists(cache_file):
        return True
    
    cached_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    cached_data.index = pd.to_datetime(cached_data.index, utc=True)
    
    last_cached_date = cached_data.index[-1]
    current_date = pd.Timestamp.now(tz='UTC').normalize()
    
    # If it's the same calendar day, no new data
    if last_cached_date.date() == current_date.date():
        return False
    
    # Check if there's been a trading day since last cache
    return is_trading_day(last_cached_date + pd.Timedelta(days=1))

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock price forecasting using Lag-Llama model')
    parser.add_argument('--production', action='store_true', 
                       help='Run in production mode (only generate predictions for new trading days)')
    args = parser.parse_args()
    
    # In production mode, first check if there's new data
    if args.production:
        print("Running in production mode...")
        if not check_new_trading_day():
            print("No additional trading day data available. Exiting.")
            return
        print("New trading day detected. Generating predictions...")
    else:
        print("Running in development mode...")
    
    # Update accuracy of past predictions only in production mode
    if args.production:
        update_prediction_accuracy()
    
    # Load the model with larger context length
    print("\nLoading model...")
    model = load_lagllama_model(context_length=256)
    model.eval()
    
    # Get stock data
    print("\nFetching stock data...")
    stocks = TECH_TICKERS
    
    # Fetch all stock data at once with production mode parameter
    data = get_historical_data(tickers=stocks, production_mode=args.production)
    
    # In production mode, verify we actually got new data
    if args.production:
        if data.empty:
            print("No data available for prediction. Exiting.")
            return
        
        # Verify the data is actually from the latest trading day
        latest_data_date = data.index[-1]
        if latest_data_date.date() < pd.Timestamp.now(tz='UTC').normalize().date():
            print(f"Latest data ({latest_data_date.date()}) is not from current trading day. Exiting.")
            return
    
    # Create a list to store forecast results
    forecast_results = []
    
    for stock in stocks:
        print(f"\nProcessing {stock}:")
        stock_data = data[stock]
        stock_data.name = stock  # Set series name for reference
        dates = data.index
        
        try:
            forecast_data = forecast_stock(model, stock_data, dates)
            
            # Store results for each horizon
            for horizon, values in forecast_data['forecasts'].items():
                forecast_results.append({
                    'Ticker': stock,
                    'Last Value': forecast_data['last_historical_value'],
                    'Horizon': horizon,
                    'Values': values,
                    'Percent Changes': (values - forecast_data['last_historical_value']) / forecast_data['last_historical_value'] * 100
                })
                
                # In production mode, save predictions to history
                if args.production:
                    current_date = pd.Timestamp.now(tz='UTC').normalize()
                    for i, value in enumerate(values):
                        target_date = current_date + pd.Timedelta(days=i+1)
                        save_prediction(
                            ticker=stock,
                            prediction_date=current_date,
                            target_date=target_date,
                            predicted_value=value,
                            horizon=horizon
                        )
                        
        except Exception as e:
            print(f"Error forecasting {stock}: {str(e)}")
            traceback.print_exc()
            if args.production:
                # In production mode, log errors more prominently
                print(f"PRODUCTION ERROR: Failed to generate forecast for {stock}")
    
    # Display results in a simplified table
    if forecast_results:
        print("\n" + "="*120)
        print("Stock Price Forecasts")
        print("="*120)
        
        for horizon in PREDICTION_HORIZONS.keys():
            print(f"\n{horizon} Horizon Forecasts:")
            print("-"*120)
            print(f"{'Ticker':<8} {'Last Price':<12} {'Final Price':<12} {'Max Change %':<12} {'Min Change %':<12} {'Avg Change %':<12}")
            print("-"*120)
            
            for result in forecast_results:
                if result['Horizon'] == horizon:
                    print(f"{result['Ticker']:<8} "
                          f"${result['Last Value']:<11.2f} "
                          f"${result['Values'][-1]:<11.2f} "
                          f"{max(result['Percent Changes']):>11.2f}% "
                          f"{min(result['Percent Changes']):>11.2f}% "
                          f"{np.mean(result['Percent Changes']):>11.2f}%")
        print("="*120)
        
        # Add summary table for average predictions
        print("\nSummary of Average Predictions:")
        print("-"*100)
        print(f"{'Ticker':<8} {'Next Day %':<12} {'2d Avg %':<12} {'1w Avg %':<12} {'1m Avg %':<12}")
        print("-"*100)
        
        for stock in stocks:
            stock_results = {horizon: [] for horizon in PREDICTION_HORIZONS.keys()}
            next_day_change = None
            for result in forecast_results:
                if result['Ticker'] == stock:
                    stock_results[result['Horizon']] = np.mean(result['Percent Changes'])
                    if result['Horizon'] == '2d':  # Get next day prediction from 2d forecast
                        next_day_change = result['Percent Changes'][0]
            
            print(f"{stock:<8} "
                  f"{next_day_change:>11.2f}% "
                  f"{stock_results['2d']:>11.2f}% "
                  f"{stock_results['1w']:>11.2f}% "
                  f"{stock_results['1m']:>11.2f}%")
        print("-"*100)

        if args.production:
            print("\nProduction run completed successfully. All predictions have been saved.")

if __name__ == '__main__':
    main() 