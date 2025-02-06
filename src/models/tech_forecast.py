import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
from huggingface_hub import hf_hub_download
from lag_llama import LagLlamaForProbabilisticForecasting
import matplotlib.pyplot as plt

# Top 10 Tech Companies (by market cap as of 2024)
TECH_TICKERS = [
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

def get_historical_data(tickers, days=90):
    """Fetch historical data for given tickers."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            data[ticker] = hist['Close']
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    return pd.DataFrame(data)

def prepare_data_for_lagllama(series, context_length=32):
    """Prepare data in format suitable for Lag-Llama."""
    # Normalize the data
    mean = series.mean()
    std = series.std()
    normalized = (series - mean) / std
    
    # Convert to tensor and ensure length matches context_length
    data = torch.tensor(normalized.values[-context_length:], dtype=torch.float32).unsqueeze(-1)
    return data, mean, std

def load_lagllama_model():
    """Load the Lag-Llama model."""
    # Download model from HuggingFace
    model_path = hf_hub_download(
        repo_id="time-series-foundation-models/Lag-Llama",
        filename="model.pt"
    )
    
    # Load model
    model = LagLlamaForProbabilisticForecasting.from_pretrained(model_path)
    model.eval()
    return model

def forecast_stock(model, data, forecast_days=[1,2,3,5,8], context_length=32):
    """Generate forecasts for specified days ahead."""
    forecasts = {}
    
    for days in forecast_days:
        with torch.no_grad():
            # Generate prediction
            prediction = model.forecast(
                data,
                prediction_length=days,
                context_length=context_length
            )
        
        forecasts[days] = prediction
    
    return forecasts

def main():
    # 1. Get historical data
    print("Fetching historical data...")
    historical_data = get_historical_data(TECH_TICKERS)
    
    # 2. Load model
    print("Loading Lag-Llama model...")
    model = load_lagllama_model()
    
    # 3. Generate forecasts for each stock
    results = {}
    for ticker in TECH_TICKERS:
        print(f"\nProcessing {ticker}...")
        series = historical_data[ticker]
        data, mean, std = prepare_data_for_lagllama(series)
        
        # Generate forecasts
        forecasts = forecast_stock(model, data)
        
        # Store results
        results[ticker] = {
            'data': series,
            'forecasts': forecasts,
            'stats': {'mean': mean, 'std': std}
        }
    
    # 4. Calculate correlations
    correlations = historical_data.corr()
    print("\nStock Correlations:")
    print(correlations)
    
    # 5. Plot results
    plt.figure(figsize=(15, 10))
    plt.imshow(correlations, cmap='coolwarm')
    plt.colorbar()
    plt.xticks(range(len(TECH_TICKERS)), TECH_TICKERS, rotation=45)
    plt.yticks(range(len(TECH_TICKERS)), TECH_TICKERS)
    plt.title("Tech Stock Correlations")
    plt.tight_layout()
    plt.savefig('tech_correlations.png')
    
    # Save forecasts
    forecast_results = pd.DataFrame()
    for ticker in TECH_TICKERS:
        for days in [1,2,3,5,8]:
            forecast = results[ticker]['forecasts'][days]
            mean_forecast = forecast.mean.numpy() * results[ticker]['stats']['std'] + results[ticker]['stats']['mean']
            forecast_results[f"{ticker}_{days}d"] = mean_forecast.flatten()
    
    forecast_results.to_csv('tech_forecasts.csv')
    print("\nResults saved to 'tech_forecasts.csv' and 'tech_correlations.png'")

if __name__ == "__main__":
    main() 