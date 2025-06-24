#!/usr/bin/env python
# coding: utf-8

# # Hybrid Black-Litterman Portfolio Optimization
# ## ARMA-GARCH and SVR Approach for Quantitative View Generation
# 
# This notebook implements the hybrid approach from "A Hybrid Approach for Generating Investor Views in Black-Litterman Model" by Yebei-Rong.
# The methodology replaces subjective views with a systematic pipeline: ARMA-GARCH → SVR → Black-Litterman



# ### 1. Import Libraries and Setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Portfolio optimization
import yfinance as yf
from pypfopt import BlackLittermanModel, risk_models, expected_returns
from pypfopt.black_litterman import market_implied_prior_returns, market_implied_risk_aversion
from pypfopt.efficient_frontier import EfficientFrontier

# Technical indicators
import talib

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")



# ### 2. Data Collection and Preprocessing

class DataProcessor:
    """
    Handles data collection, cleaning, and technical indicator calculation
    """
    
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.prices = None
        self.returns = None
        self.indicators = {}
        
    def fetch_data(self):
        """Fetch historical price data"""
        print(f"Fetching data for {self.tickers} from {self.start_date} to {self.end_date}")
        self.prices = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Close']
        
        # Handle single ticker case
        if len(self.tickers) == 1:
            self.prices = self.prices.to_frame(self.tickers[0])
            
        # Calculate returns
        self.returns = self.prices.pct_change().dropna()
        
        # Fetch VIX for market sentiment
        self.vix = yf.download('^VIX', start=self.start_date, end=self.end_date)['Close']
        
        print(f"Data shape: {self.prices.shape}")
        print(f"Date range: {self.prices.index[0]} to {self.prices.index[-1]}")
        
    def calculate_technical_indicators(self):
        """
        Calculate the 8 technical indicators used in the hybrid approach:
        ATR, ADXR, EMA, MACD, SMA, Hurst Exponent, RSI, VIX
        """
        print("\nCalculating technical indicators...")
        
        for ticker in self.tickers:
            print(f"  Processing {ticker}...")
            
            # Get OHLC data for indicators that need it
            ticker_data = yf.download(ticker, start=self.start_date, end=self.end_date)
            high = ticker_data['High']
            low = ticker_data['Low']
            close = ticker_data['Close']

            #High low close for numpy arrays
            high_num = high.to_numpy().squeeze()
            low_num = low.to_numpy().squeeze()
            close_num = close.to_numpy().squeeze()
            
            # Create indicator DataFrame
            indicators_df = pd.DataFrame(index=close.index)
            
            # 1. ATR (Average True Range) - 14 days
            indicators_df['ATR'] = talib.ATR(high_num, low_num, close_num, timeperiod=14)
            
            # 2. ADXR (Average Directional Movement Index Rating) - 14 days
            indicators_df['ADXR'] = talib.ADXR(high_num, low_num, close_num, timeperiod=14)
            
            # 3. EMA (Exponential Moving Average) - 20 days
            indicators_df['EMA'] = talib.EMA(close_num, timeperiod=20)
            
            # 4. MACD (Moving Average Convergence Divergence)
            macd, macdsignal, macdhist = talib.MACD(close_num, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators_df['MACD'] = macd
            
            # 5. SMA (Simple Moving Average) - 20 days
            indicators_df['SMA'] = talib.SMA(close_num, timeperiod=20)
            
            # 6. Hurst Exponent (simplified version)
            indicators_df['Hurst'] = self._calculate_hurst_exponent(close, window=100) #ovo cu odjebat jer nez koji je to penis
            
            # 7. RSI (Relative Strength Index) - 14 days
            indicators_df['RSI'] = talib.RSI(close_num, timeperiod=14)
            
            # 8. VIX (market sentiment proxy) - aligned to ticker dates
            indicators_df['VIX'] = self.vix.reindex(indicators_df.index, method='ffill')
            
            # Drop NaN values and store
            self.indicators[ticker] = indicators_df.dropna()
            
        print("Technical indicators calculated successfully")
        
    def _calculate_hurst_exponent(self, series, window=100):
        """
        Calculate rolling Hurst exponent as a measure of trending/mean-reverting behavior
        """
        def hurst(ts):
            """Calculate Hurst exponent for a time series"""
            lags = range(2, min(len(ts)//2, 20))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        # Rolling calculation
        hurst_values = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            hurst_values.iloc[i] = hurst(series.iloc[i-window:i].values)
            
        return hurst_values


# ### 3. ARMA-GARCH Model Implementation


class ARMAGARCHForecaster:
    """
    Implements ARMA-GARCH forecasting for technical indicators
    """
    
    def __init__(self, arma_order=(1,0,1), garch_order=(1,1)):
        self.arma_order = arma_order
        self.garch_order = garch_order
        self.models = {}
        self.forecasts = {}
        
    def check_stationarity(self, series, indicator_name, verbose=True):
        """
        Check stationarity using Augmented Dickey-Fuller test
        Returns stationary series (differenced if necessary)
        """
        # Remove any NaN or infinite values
        clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_series) < 20:
            if verbose:
                print(f"  Warning: Not enough data for {indicator_name} after cleaning")
            return clean_series, False
            
        result = adfuller(clean_series)
        if verbose:
            print(f"\n{indicator_name} - ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}")
        
        if result[1] > 0.05:
            if verbose:
                print(f"  Series is non-stationary. Applying differencing...")
            # Use log differences for positive series like VIX
            indicator_base = indicator_name.split('_')[-1]
            if indicator_base in ['VIX', 'ATR', 'ADXR']:
                # Ensure all values are positive before log
                if (clean_series > 0).all():
                    diff_series = np.log(clean_series).diff().dropna()
                else:
                    # Use regular differencing if log is not applicable
                    diff_series = clean_series.diff().dropna()
            else:
                diff_series = clean_series.diff().dropna()
            
            # Remove any NaN or infinite values after differencing
            diff_series = diff_series.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(diff_series) < 20:
                if verbose:
                    print(f"  Warning: Not enough data after differencing")
                return clean_series, False
            
            # Check again
            result_diff = adfuller(diff_series)
            if verbose:
                print(f"  After differencing - ADF Statistic: {result_diff[0]:.4f}, p-value: {result_diff[1]:.4f}")
            return diff_series, True
        else:
            if verbose:
                print(f"  Series is stationary")
            return clean_series, False
            
    def fit_arma_garch(self, series, indicator_name, verbose=True):
        """
        Fit ARMA-GARCH model to a single indicator series
        """
        # Check and ensure stationarity
        stationary_series, was_differenced = self.check_stationarity(series, indicator_name, verbose)
        
        # Check if series has enough variance
        if np.var(stationary_series) < 1e-8:
            if verbose:
                print(f"  Warning: Series has very low variance. Using last value forecast.")
            self.models[indicator_name] = {
                'model': None,
                'scaler': None,
                'was_differenced': False,
                'last_value': series.iloc[-1],
                'last_diff': None
            }
            return
        
        # Scale the series to help with convergence
        scaler = MinMaxScaler(feature_range=(0.1, 1))
        scaled_series = scaler.fit_transform(stationary_series.values.reshape(-1, 1)).flatten()
        scaled_series = pd.Series(scaled_series, index=stationary_series.index)
        
        # Fit ARMA-GARCH model
        model = arch_model(
            scaled_series, 
            vol='Garch',
            p=self.garch_order[0], 
            q=self.garch_order[1],
            mean='ARX',
            lags=self.arma_order[0]
        )
        
        try:
            fitted_model = model.fit(disp='off', show_warning=False)
            
            # Store model information
            self.models[indicator_name] = {
                'model': fitted_model,
                'scaler': scaler,
                'was_differenced': was_differenced,
                'last_value': series.iloc[-1],
                'last_diff': stationary_series.iloc[-1] if was_differenced else None
            }
            
            if verbose:
                print(f"\n{indicator_name} - Model fitted successfully")
                print(f"  Log-likelihood: {fitted_model.loglikelihood:.2f}")
                print(f"  Convergence: {fitted_model.convergence_flag == 0}")
            
        except Exception as e:
            if verbose:
                print(f"\nError fitting {indicator_name}: {str(e)}")
                print(f"  Series length: {len(scaled_series)}")
                print(f"  Series variance: {np.var(scaled_series):.6f}")
            # Use simple model as fallback
            self.models[indicator_name] = None
            
    def forecast_indicators(self, horizon=1):
        """
        Generate forecasts for all fitted indicators
        """
        self.forecasts = {}
        
        for indicator_name, model_info in self.models.items():
            if model_info is None or (isinstance(model_info, dict) and model_info.get('model') is None):
                # Use last value as forecast for failed models
                if isinstance(model_info, dict):
                    self.forecasts[indicator_name] = model_info.get('last_value', 0)
                else:
                    self.forecasts[indicator_name] = 0
                continue
                
            model = model_info['model']
            scaler = model_info['scaler']
            
            # Generate forecast
            try:
                forecast_result = model.forecast(horizon=horizon)
                
                # Extract the forecast value - handle different return types
                if hasattr(forecast_result, 'mean'):
                    forecast_mean = forecast_result.mean
                    
                    # Handle different data structures
                    if isinstance(forecast_mean, pd.DataFrame):
                        # Take the last row, first column
                        forecast_scaled = float(forecast_mean.iloc[-1, 0])
                    elif isinstance(forecast_mean, pd.Series):
                        # Take the last value
                        forecast_scaled = float(forecast_mean.iloc[-1])
                    elif isinstance(forecast_mean, np.ndarray):
                        # Take the last value
                        if forecast_mean.ndim > 1:
                            forecast_scaled = float(forecast_mean[-1, 0])
                        else:
                            forecast_scaled = float(forecast_mean[-1])
                    else:
                        # Already a scalar
                        forecast_scaled = float(forecast_mean)
                else:
                    # Fallback
                    forecast_scaled = 0.0
                    
                # Inverse transform - reshape to 2D array for sklearn
                forecast_value = scaler.inverse_transform(np.array([[forecast_scaled]]))[0][0]
                
                # If series was differenced, integrate back
                if model_info['was_differenced']:
                    indicator_base = indicator_name.split('_')[-1]  # Get indicator type
                    if indicator_base in ['VIX', 'ATR', 'ADXR']:
                        # For log-differenced series
                        forecast_value = model_info['last_value'] * np.exp(forecast_value)
                    else:
                        forecast_value = model_info['last_value'] + forecast_value
                        
                self.forecasts[indicator_name] = forecast_value
                
            except Exception as e:
                # Only show warning for unexpected errors
                if "cannot convert the series" not in str(e):
                    print(f"  Warning: Forecast failed for {indicator_name}: {str(e)}")
                # Use last known value as forecast
                self.forecasts[indicator_name] = model_info.get('last_value', 0)
            
        return self.forecasts

# ### 4. Support Vector Regression Implementation

class SVRViewGenerator:
    """
    Translates technical indicator forecasts to return predictions using SVR
    """
    
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.best_params = {}
        
    def prepare_training_data(self, indicators_df, returns, lookback=10):
        """
        Prepare features (indicators) and targets (future returns) for SVR training
        """
        # Align indicators and returns
        common_dates = indicators_df.index.intersection(returns.index)
        
        # Create feature matrix (indicators at time t)
        X = indicators_df.loc[common_dates].values[:-lookback]
        
        # Create target (future return over holding period)
        # Use log returns for better properties
        y = []
        for i in range(len(X)):
            # Get returns for next 'lookback' days
            future_returns = returns.iloc[i+1:i+1+lookback]
            if len(future_returns) == lookback:
                # Calculate cumulative return over holding period
                cumulative_return = (1 + future_returns).prod() - 1
                y.append(cumulative_return)
            
        return X[:len(y)], np.array(y)
        
    def train_svr_model(self, ticker, indicators_df, returns, optimize_params=True):
        """
        Train SVR model for a specific asset
        """
        if optimize_params:
            print(f"\nTraining SVR for {ticker}...")
        
        # Prepare data
        X, y = self.prepare_training_data(indicators_df, returns)
        
        # Check if we have enough data
        if len(X) < 100:
            print(f"  Warning: Not enough training data for {ticker} ({len(X)} samples)")
            return
        
        # Scale features to (-1, 1) range
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        
        # Remove outliers in target variable (returns > 3 std)
        y_mean, y_std = np.mean(y), np.std(y)
        mask = np.abs(y - y_mean) <= 3 * y_std
        X_scaled = X_scaled[mask]
        y = y[mask]
        
        if optimize_params:
            # Try different kernels and parameters
            param_grid = [
                {
                    'kernel': ['rbf'],
                    'C': [0.1, 1.0, 10.0],
                    'epsilon': [0.001, 0.01, 0.1],
                    'gamma': ['scale', 'auto']
                },
                {
                    'kernel': ['linear'],
                    'C': [0.1, 1.0, 10.0],
                    'epsilon': [0.001, 0.01, 0.1]
                }
            ]
            
            svr = SVR()
            tscv = TimeSeriesSplit(n_splits=3)
            
            grid_search = GridSearchCV(
                svr, param_grid, 
                cv=tscv, 
                scoring='r2',  # Use R² instead of MSE
                n_jobs=-1
            )
            
            grid_search.fit(X_scaled, y)
            best_model = grid_search.best_estimator_

            self.best_params[ticker] = grid_search.best_params_
            
            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  Best CV R²: {grid_search.best_score_:.4f}")
            
            # If R² is too low, try a simpler model
            if grid_search.best_score_ < 0.01:
                print(f"  Warning: Poor model fit. Using linear regression fallback.")
                from sklearn.linear_model import Ridge
                best_model = Ridge(alpha=1.0, random_state=42)
                best_model.fit(X_scaled, y)
            
        else:
            # Use default parameters
            best_model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
            best_model.fit(X_scaled, y)
            
        # Calculate in-sample R²
        y_pred = best_model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        
        if optimize_params or r2 < 0:
            print(f"  In-sample R² for {ticker}: {r2:.4f}")
        
        # Store model and scaler
        self.models[ticker] = best_model
        self.scalers[ticker] = scaler
        
        # Calculate feature importance only on first iteration
        if optimize_params:
            self._calculate_feature_importance(ticker, X_scaled, y)
    
    	
    def get_optimal_parameters(self):
        """
        Return the optimal parameters found for each ticker	
        """
	
        return self.best_params
	
    

    def display_optimal_parameters(self):
        """
        Display the optimal SVR parameters found for each ticker
        """
        print("\n" + "="*60)
        print("OPTIMAL SVR PARAMETERS")
        print("="*60)
        
        for ticker, params in self.best_params.items():
            print(f"\n{ticker}:")
            for param, value in params.items():
                print(f"  {param}: {value}")
                
        # Create a summary of most common parameters

        all_kernels = [p.get('kernel', 'rbf') for p in self.best_params.values()]
        all_C = [p.get('C', 1.0) for p in self.best_params.values()]
        all_epsilon = [p.get('epsilon', 0.1) for p in self.best_params.values()]
        
        print("\nMost common parameters across all tickers:")
        print(f"  Kernel: {max(set(all_kernels), key=all_kernels.count)}")
        print(f"  C (median): {np.median(all_C):.3f}")
        print(f"  Epsilon (median): {np.median(all_epsilon):.3f}")
	
        print("\nNote: You can use these parameters to skip grid search in future runs.")
	
        print("Set optimize_params=False and initialize SVRViewGenerator with these values.")
	
        return self.best_params
        
    def _calculate_feature_importance(self, ticker, X, y):
        """
        Calculate feature importance using a simple permutation approach
        """
        model = self.models[ticker]
        baseline_score = r2_score(y, model.predict(X))
        
        importances = []
        feature_names = ['ATR', 'ADXR', 'EMA', 'MACD', 'SMA', 'Hurst', 'RSI', 'VIX']
        
        for i in range(X.shape[1]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_score = r2_score(y, model.predict(X_permuted))
            importance = baseline_score - permuted_score
            importances.append(max(0, importance))
            
        self.feature_importance[ticker] = dict(zip(feature_names, importances))
        
    def generate_views(self, indicator_forecasts):
        """
        Generate return views from indicator forecasts
        """
        views = {}
        uncertainties = {}
        issues = []
        
        for ticker in self.models.keys():
            if ticker not in indicator_forecasts:
                issues.append(f"No forecasts for {ticker}")
                continue
                
            forecast_values = indicator_forecasts[ticker]
            
            # Prepare forecast array
            X_forecast = np.array(list(forecast_values.values())).reshape(1, -1)
            
            # Check if we have the right number of features
            if X_forecast.shape[1] != 8:  # We expect 8 indicators
                issues.append(f"{ticker}: {X_forecast.shape[1]} indicators instead of 8")
                continue
            
            X_forecast_scaled = self.scalers[ticker].transform(X_forecast)
            
            # Generate return prediction
            return_prediction = self.models[ticker].predict(X_forecast_scaled)[0]
            
            # Calculate uncertainty based on support vectors distance
            # Simplified approach: use fixed uncertainty based on training performance
            # In practice, you might want to use prediction intervals
            uncertainty = 0  # 2% base uncertainty #TODO Im going with 0
            
            views[ticker] = return_prediction
            uncertainties[ticker] = uncertainty
        
        if issues:
            print(f"  View generation issues: {len(issues)}")

        print(views)
            
        return views, uncertainties

# ### 5. Black-Litterman Model Implementation

class HybridBlackLitterman:
    """
    Black-Litterman model with hybrid view generation using PyPortfolioOpt
    """
    
    def __init__(self, risk_aversion=2.5, tau=0.025):
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.cov_matrix = None
        self.market_prior = None
        self.bl_model = None
        self.posterior_returns = None
        self.optimal_weights = None
        
    def optimize_portfolio(self, returns_data, views, uncertainties, prices_data, holding_period=90, market_caps=None):
        """
        Optimize portfolio using PyPortfolioOpt's Black-Litterman implementation
        """
        tickers = list(returns_data.columns)

        print(prices_data.shape)
        print(prices_data)

        #print(market_caps.shape)
        
        # Calculate covariance matrix
        self.cov_matrix = risk_models.CovarianceShrinkage(prices_data).ledoit_wolf() #TODO mijenjao
        
        # Calculate market-implied returns (equilibrium returns)
        # Using equal weights as market cap proxy
        market_caps = pd.Series([100000000.0] * len(tickers), index=tickers) #TODO OVO SAM ZABORAVIO AAAAA (vidjetcemo dal mi radi)
        # moram market capove kak spada racunat... nez dal ovo ista ima smisla...

        print(market_caps)
        self.market_prior = market_implied_prior_returns(
            market_caps=market_caps,
            risk_aversion=self.risk_aversion,
            cov_matrix=self.cov_matrix,
            risk_free_rate=0.04 #TODO mijenjao
        )
        
        # CRITICAL FIX: Use ticker names as keys, not numeric indices
        absolute_views = {}
        
        for ticker, view_return in views.items():
            if ticker in tickers:  # Only include views for tickers in our universe
                #absolute_views[ticker] = view_return * 252  # Annualize
                # Views are cumulative returns over holding period
                # Annualize them properly for Black-Litterman
                annualized_view = ((1 + view_return) ** (252 / holding_period)) - 1
                
                # Sanity check - cap at reasonable annual returns
                annualized_view = np.clip(annualized_view, -0.50, 1.00)  # -50% to +100% annual
                
                absolute_views[ticker] = annualized_view
        
        if not absolute_views:
            print("  Warning: No matching views found!")
            return np.array([1/len(tickers)] * len(tickers))
        
        # Convert uncertainties to a single confidence value
        avg_confidence = np.mean([1.0 / (1.0 + uncertainties[ticker]) 
                                  for ticker in absolute_views.keys()])
        
        print("Printing model parameters:")
        print("Cov matrix\n", self.cov_matrix)
        print()
        print("Market prior\n", self.market_prior)
        print()
        print("Views\n", absolute_views)
        print()
        print(avg_confidence)
        print()
        print(self.tau)        
        # Create Black-Litterman model with ticker-keyed views

        #absolute_views = {'AAPL': np.float64(0.8), 'MSFT': np.float64(0.5)}
        self.bl_model = BlackLittermanModel(
            cov_matrix=self.cov_matrix,
            pi=self.market_prior,
            absolute_views=absolute_views,  # This must be {ticker: view} dict
            omega = "idzorek",  # Use Idzorek's method for uncertainty
            view_confidences = np.array([1.0] * len(absolute_views)),  # Uniform confidence for each view (100%)
            tau=self.tau
        )
        
        # Get posterior returns
        self.posterior_returns = self.bl_model.bl_returns()
        posterior_cov = self.bl_model.bl_cov()

        print(f"  Posterior returns: {self.posterior_returns}")
        print(f"  Posterior covariance matrix:\n{posterior_cov}")

        print(f"  Posterior returns: {self.posterior_returns.shape}")
        print(f"  Posterior covariance matrix:\n{posterior_cov.shape}")
        
        # Optimize portfolio using Efficient Frontier
        ef = EfficientFrontier(self.posterior_returns, posterior_cov)
        
        # Add constraintsm #TODO, MIJENJAO
        #ef.add_constraint(lambda w: w >= 0)  # Long-only
        #ef.add_constraint(lambda w: w <= 0.45)  # Max 35% per asset ##TODO mijenjao
        
        # Maximize Sharpe ratio
        # try: TODO mijenjao PUNO MIJANJAO
        ef.max_sharpe()
        self.optimal_weights = ef.clean_weights()
        #self.optimal_weights = self.bl_model.weights
        
        # Convert to numpy array in ticker order
        return np.array([self.optimal_weights[ticker] for ticker in tickers])
            
        # except Exception as e:
        #     print(f"Optimization failed: {e}")
        #     # Return equal weights as fallback
        #     return np.array([1/len(tickers)] * len(tickers))

# ### 6. Backtesting Framework

class HybridBacktester:
    """
    Backtesting framework for the hybrid Black-Litterman strategy
    """
    
    def __init__(self, data_processor, holding_period=10):
        self.data_processor = data_processor
        self.holding_period = holding_period
        self.portfolio_values = []
        self.weights_history = []
        self.returns_history = []
        self.rebalance_dates = []
        
        # Benchmark tracking
        self.benchmark_values = []
        self.benchmark_returns = []
        self.benchmark_weights = None
    
    def plot_portfolio_weights_detailed(self):
        """
        Create a detailed visualization of portfolio weights evolution
        """
        # Extract weights data properly
        weights_data = []
        dates = []
        for w in self.weights_history:
            weights_data.append(w['weights'])
            dates.append(w['date'])
        
        # Create DataFrame with proper structure
        weights_df = pd.DataFrame(weights_data, index=dates)
        
        # Ensure all tickers are present as columns
        for ticker in self.data_processor.tickers:
            if ticker not in weights_df.columns:
                print(f"Warning: {ticker} not found in weights DataFrame")
                weights_df[ticker] = 0
        
        # Create subplots for each asset
        n_assets = len(self.data_processor.tickers)
        fig, axes = plt.subplots(n_assets, 1, figsize=(12, 2.5 * n_assets), sharex=True)
        
        # If only one asset, make it a list
        if n_assets == 1:
            axes = [axes]
        
        # Plot each asset's weight evolution
        for i, (ticker, ax) in enumerate(zip(self.data_processor.tickers, axes)):
            # BL weights
            ax.plot(weights_df.index, weights_df[ticker], 
                   label=f'{ticker} (BL)', linewidth=2.5, marker='o', markersize=6)
            
            # Benchmark weight (constant line)
            benchmark_weight = self.benchmark_weights[i]
            ax.axhline(y=benchmark_weight, color='red', linestyle='--', 
                      label=f'{ticker} (Benchmark: {benchmark_weight:.1%})', linewidth=2)
            
            # Formatting
            ax.set_ylabel(f'{ticker} Weight', fontsize=10)
            ax.set_ylim(0, 0.20)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            
            # Highlight rebalancing dates
            for date in self.rebalance_dates:
                ax.axvline(x=date, color='gray', alpha=0.3, linestyle=':')
        
        # Overall title and x-label
        fig.suptitle('Individual Asset Weight Evolution (BL vs Benchmark)', 
                    fontsize=16, fontweight='bold', y=0.995)
        axes[-1].set_xlabel('Date', fontsize=12)
        
        # Rotate x-axis labels
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Also create a heatmap of weights
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Prepare data for heatmap
        weights_matrix = weights_df.T.values
        
        # Create heatmap
        im = ax.imshow(weights_matrix, aspect='auto', cmap='YlOrRd', 
                      interpolation='nearest', vmin=0, vmax=0.15)
        
        # Set ticks
        ax.set_xticks(np.arange(len(weights_df.index)))
        ax.set_yticks(np.arange(len(self.data_processor.tickers)))
        
        # Set labels
        ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in weights_df.index], 
                          rotation=45, ha='right')
        ax.set_yticklabels(self.data_processor.tickers)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Weight', fontsize=12)
        
        # Add title
        ax.set_title('Portfolio Weights Heatmap Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Rebalancing Date', fontsize=12)
        ax.set_ylabel('Asset', fontsize=12)
        
        # Add text annotations for weights
        for i in range(len(self.data_processor.tickers)):
            for j in range(len(weights_df.index)):
                text = ax.text(j, i, f'{weights_matrix[i, j]:.1%}',
                             ha='center', va='center', color='black' if weights_matrix[i, j] < 0.075 else 'white',
                             fontsize=8)
        
        plt.tight_layout()
        plt.show()    
        
        print("  Fetching market caps...")
        market_caps = {}
        
        for ticker in self.data_processor.tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                market_cap = info.get('marketCap', 0)
                if market_cap == 0:
                    # Fallback to shares * price
                    shares = info.get('sharesOutstanding', 0)
                    price = self.data_processor.prices[ticker].iloc[-1]
                    market_cap = shares * price if shares > 0 else 0
                market_caps[ticker] = market_cap
            except:
                # If we can't get market cap, use equal weight
                market_caps[ticker] = 1.0
                
        # Convert to weights
        total_cap = sum(market_caps.values())
        if total_cap > 0:
            weights = {ticker: cap/total_cap for ticker, cap in market_caps.items()}
        else:
            # Fallback to equal weights
            n = len(self.data_processor.tickers)
            weights = {ticker: 1/n for ticker in self.data_processor.tickers}
            
        # Display weights
        print("  Market cap weights:")
        for ticker, weight in weights.items():
            print(f"    {ticker}: {weight:.2%}")
            
        return np.array([weights[ticker] for ticker in self.data_processor.tickers])
    
    def calculate_market_cap_weights(self):
        """
        Calculate market cap weights for benchmark
        Using current market caps from yfinance
        """
        print("  Fetching market caps...")
        market_caps = {}
        
        for ticker in self.data_processor.tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                market_cap = info.get('marketCap', 0)
                if market_cap == 0:
                    # Fallback to shares * price
                    shares = info.get('sharesOutstanding', 0)
                    price = self.data_processor.prices[ticker].iloc[-1]
                    market_cap = shares * price if shares > 0 else 0
                market_caps[ticker] = market_cap
            except:
                # If we can't get market cap, use equal weight
                market_caps[ticker] = 1.0
                
        # Convert to weights
        total_cap = sum(market_caps.values())
        if total_cap > 0:
            weights = {ticker: cap/total_cap for ticker, cap in market_caps.items()}
        else:
            # Fallback to equal weights
            n = len(self.data_processor.tickers)
            weights = {ticker: 1/n for ticker in self.data_processor.tickers}
            
        # Display weights
        print("  Market cap weights:")
        for ticker, weight in weights.items():
            print(f"    {ticker}: {weight:.2%}")
            
        return np.array([weights[ticker] for ticker in self.data_processor.tickers])
        
    def run_backtest(self, train_start, train_end, test_start, test_end):
        """
        Run the complete backtest with rolling window
        """
        print(f"\nRunning backtest from {test_start} to {test_end}")
        print(f"Holding period: {self.holding_period} days")
        
        # Calculate benchmark weights (market cap weighted)
        print("\nCalculating market cap weights for benchmark...")
        self.benchmark_weights = self.calculate_market_cap_weights()
        
        # Initialize models
        arma_garch = ARMAGARCHForecaster()
        svr_generator = SVRViewGenerator()
        bl_model = HybridBlackLitterman()

        # Store svr_generator for parameter access
        self.svr_generator = svr_generator
        
        # Get test period dates
        test_dates = self.data_processor.returns.loc[test_start:test_end].index
        
        # Initialize portfolio values
        portfolio_value = 100  # Start with $100
        benchmark_value = 100
        self.portfolio_values = [portfolio_value]
        self.benchmark_values = [benchmark_value]
        
        # Rebalancing loop
        for i in range(0, len(test_dates) - self.holding_period, self.holding_period):
            current_date = test_dates[i]
            next_rebalance = test_dates[min(i + self.holding_period, len(test_dates) - 1)]
            
            print(f"\nRebalancing on {current_date.strftime('%Y-%m-%d')}")
            
            # Update training window (expanding or rolling)
            train_data_end = current_date - timedelta(days=1)
            
            # Step 1: Train ARMA-GARCH models for each indicator
            indicator_forecasts = {}
            fitting_issues = 0
            
            for ticker in self.data_processor.tickers:
                ticker_forecasts = {}
                indicators_df = self.data_processor.indicators[ticker].loc[:train_data_end]
                
                for indicator_name in indicators_df.columns:
                    # Suppress verbose output for individual indicator fitting
                    arma_garch.fit_arma_garch(
                        indicators_df[indicator_name], 
                        f"{ticker}_{indicator_name}",
                        verbose=False  # Add verbose parameter
                    )
                    
                # Forecast indicators
                forecasts = arma_garch.forecast_indicators(horizon=self.holding_period)
                
                # Extract forecasts for this ticker
                for key, value in forecasts.items():
                    if key.startswith(f"{ticker}_"):
                        indicator = key.replace(f"{ticker}_", "")
                        ticker_forecasts[indicator] = value
                        
                indicator_forecasts[ticker] = ticker_forecasts
                
            # Count fitting issues
            for model_info in arma_garch.models.values():
                if model_info is None or (isinstance(model_info, dict) and model_info.get('model') is None):
                    fitting_issues += 1
                    
            print(f"  ARMA-GARCH models fitted ({fitting_issues} issues out of {len(arma_garch.models)})")
            print(f"  Indicator forecasts prepared for: {list(indicator_forecasts.keys())}")
                
            # Step 2: Train SVR models and generate views
            if i == 0:
                print("\n  Training SVR models...")
            for ticker in self.data_processor.tickers:
                indicators_df = self.data_processor.indicators[ticker].loc[:train_data_end]
                returns = self.data_processor.returns[ticker].loc[:train_data_end]
                
                svr_generator.train_svr_model(
                    ticker, indicators_df, returns, 
                    optimize_params=(i == 0)  # Only optimize on first iteration
                )
                
            # Generate views from forecasts
            views, uncertainties = svr_generator.generate_views(indicator_forecasts)
            
            print(f"  Generated views for: {list(views.keys())}")
            
            # Ensure views only contain tickers in our universe
            filtered_views = {ticker: view for ticker, view in views.items() 
                              if ticker in self.data_processor.tickers}
            filtered_uncertainties = {ticker: unc for ticker, unc in uncertainties.items() 
                                      if ticker in self.data_processor.tickers}
            
            if not filtered_views:
                print("  Error: No valid views generated!")
                # Use equal weights as fallback
                optimal_weights = np.array([1/len(self.data_processor.tickers)] * len(self.data_processor.tickers))
            else:
                # Step 3: Black-Litterman optimization
                # Get historical returns up to current date
                historical_returns = self.data_processor.returns.loc[:train_data_end]

                historical_prices = self.data_processor.prices.loc[:train_data_end]
                
                # Use PyPortfolioOpt's Black-Litterman implementation
                optimal_weights = bl_model.optimize_portfolio(
                    returns_data=historical_returns,
                    views=filtered_views,
                    uncertainties=filtered_uncertainties,
                    prices_data = historical_prices,
                    market_caps=self.market_cap_weights()
                )
            
            # Store weights
            # self.weights_history.append({
            #     'date': current_date,
            #     'weights': dict(zip(self.data_processor.tickers, optimal_weights)) // TODO!!
            # })
            self.rebalance_dates.append(current_date)
            
            # Calculate portfolio returns for holding period
            holding_period_returns = self.data_processor.returns.loc[
                current_date:next_rebalance
            ]
            
            for date, daily_returns in holding_period_returns.iterrows():
                # Hybrid portfolio return
                portfolio_return = np.sum(optimal_weights * daily_returns.values)
                portfolio_value *= (1 + portfolio_return)
                self.portfolio_values.append(portfolio_value)
                self.returns_history.append(portfolio_return)
                
                # Benchmark return
                benchmark_return = np.sum(self.benchmark_weights * daily_returns.values)
                benchmark_value *= (1 + benchmark_return)
                self.benchmark_values.append(benchmark_value)
                self.benchmark_returns.append(benchmark_return)
                
        print(f"\nBacktest complete.")
        print(f"Final portfolio value: ${portfolio_value:.2f}")
        print(f"Final benchmark value: ${benchmark_value:.2f}")
        print(f"Outperformance: {((portfolio_value/benchmark_value - 1) * 100):.2f}%")
    
    def market_cap_weights(self):
        market_caps = {}
        for ticker in self.data_processor.tickers:
            # Fetch market cap from yfinance
            info = yf.Ticker(ticker).info
            market_caps[ticker] = info.get('marketCap', 100e9)
        return market_caps
        
    def calculate_performance_metrics(self):
        """
        Calculate performance metrics for both portfolio and benchmark
        """
        # Portfolio metrics
        returns = np.array(self.returns_history)
        
        n_days = len(returns)
        total_return = self.portfolio_values[-1] / self.portfolio_values[0] - 1
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        annualized_vol = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        portfolio_values = np.array(self.portfolio_values)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        win_rate = np.mean(returns > 0)
        
        # Benchmark metrics
        bench_returns = np.array(self.benchmark_returns)
        bench_total_return = self.benchmark_values[-1] / self.benchmark_values[0] - 1
        bench_annualized_return = (1 + bench_total_return) ** (252 / n_days) - 1
        bench_annualized_vol = np.std(bench_returns) * np.sqrt(252)
        bench_sharpe_ratio = bench_annualized_return / bench_annualized_vol if bench_annualized_vol > 0 else 0
        
        bench_values = np.array(self.benchmark_values)
        bench_running_max = np.maximum.accumulate(bench_values)
        bench_drawdowns = (bench_values - bench_running_max) / bench_running_max
        bench_max_drawdown = np.min(bench_drawdowns)
        
        # Relative metrics
        excess_return = annualized_return - bench_annualized_return
        tracking_error = np.std(returns - bench_returns) * np.sqrt(252)
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
        
        metrics = {
            'Portfolio Metrics': {
                'Total Return': f"{total_return:.2%}",
                'Annualized Return': f"{annualized_return:.2%}",
                'Annualized Volatility': f"{annualized_vol:.2%}",
                'Sharpe Ratio': f"{sharpe_ratio:.3f}",
                'Maximum Drawdown': f"{max_drawdown:.2%}",
                'Win Rate': f"{win_rate:.2%}",
            },
            'Benchmark Metrics': {
                'Total Return': f"{bench_total_return:.2%}",
                'Annualized Return': f"{bench_annualized_return:.2%}",
                'Annualized Volatility': f"{bench_annualized_vol:.2%}",
                'Sharpe Ratio': f"{bench_sharpe_ratio:.3f}",
                'Maximum Drawdown': f"{bench_max_drawdown:.2%}",
            },
            'Relative Performance': {
                'Excess Return': f"{excess_return:.2%}",
                'Tracking Error': f"{tracking_error:.2%}",
                'Information Ratio': f"{information_ratio:.3f}",
                'Outperformance': f"{((self.portfolio_values[-1]/self.benchmark_values[-1]) - 1):.2%}"
            },
            'Other': {
                'Number of Rebalances': len(self.rebalance_dates)
            }
        }
        
        return metrics
        
    def plot_results(self):
        """
        Visualize backtest results with benchmark comparison
        """
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))
        
        # Create date index for plotting
        dates = pd.date_range(start=self.data_processor.returns.index[0], 
                            periods=len(self.portfolio_values), 
                            freq='D')[:len(self.portfolio_values)]
        
        # 1. Portfolio value comparison
        axes[0].plot(dates, self.portfolio_values, label='Hybrid BL Portfolio', linewidth=2.5, color='darkblue')
        axes[0].plot(dates, self.benchmark_values, label='Market Cap Weighted Benchmark', 
                    linewidth=2.5, color='darkred', alpha=0.8)
        axes[0].axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Initial Value')
        
        # Add final values as text
        final_portfolio = self.portfolio_values[-1]
        final_benchmark = self.benchmark_values[-1]
        axes[0].text(dates[-1], final_portfolio, f'${final_portfolio:.2f}', 
                    ha='left', va='bottom', fontsize=10, color='darkblue')
        axes[0].text(dates[-1], final_benchmark, f'${final_benchmark:.2f}', 
                    ha='left', va='top', fontsize=10, color='darkred')
        
        axes[0].set_title('Portfolio Value Over Time', fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Portfolio Value ($)', fontsize=12)
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Relative performance
        relative_perf = np.array(self.portfolio_values) / np.array(self.benchmark_values)
        axes[1].plot(dates, relative_perf, color='green', linewidth=2.5)
        axes[1].axhline(y=1, color='gray', linestyle='--')
        axes[1].fill_between(dates, 1, relative_perf, where=(relative_perf > 1), 
                            alpha=0.3, color='green', label='Outperformance')
        axes[1].fill_between(dates, 1, relative_perf, where=(relative_perf <= 1), 
                            alpha=0.3, color='red', label='Underperformance')
        axes[1].set_title('Relative Performance (Portfolio / Benchmark)', fontsize=16, fontweight='bold')
        axes[1].set_ylabel('Relative Value', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # 3. Portfolio weights over time - Line plot instead of stacked area
        weights_df = pd.DataFrame([w['weights'] for w in self.weights_history])
        weights_df.index = [w['date'] for w in self.weights_history]
        
        # Plot each asset's weight as a line
        for column in weights_df.columns:
            axes[2].plot(weights_df.index, weights_df[column], 
                        label=column, linewidth=2, marker='o', markersize=4)
        
        # Add benchmark weights as horizontal lines (dashed)
        benchmark_dict = dict(zip(self.data_processor.tickers, self.benchmark_weights))
        for ticker, weight in benchmark_dict.items():
            axes[2].axhline(y=weight, linestyle='--', alpha=0.5, linewidth=1)
            
        axes[2].set_title('Portfolio Weights Over Time (solid=BL, dashed=benchmark)', 
                         fontsize=16, fontweight='bold')
        axes[2].set_ylabel('Weight', fontsize=12)
        axes[2].set_ylim(0, 0.20)  # Set y-axis limit to 20%
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        # 4. Rolling Sharpe ratio comparison
        returns_series = pd.Series(self.returns_history)
        bench_returns_series = pd.Series(self.benchmark_returns)
        
        # Calculate rolling metrics with proper date index
        window = 60
        rolling_dates = dates[window:]
        
        rolling_sharpe = (
            returns_series.rolling(window).mean() / 
            returns_series.rolling(window).std() * 
            np.sqrt(252)
        ).iloc[window:]
        
        bench_rolling_sharpe = (
            bench_returns_series.rolling(window).mean() / 
            bench_returns_series.rolling(window).std() * 
            np.sqrt(252)
        ).iloc[window:]
        
        axes[3].plot(rolling_dates[1:], rolling_sharpe, label='Hybrid BL Portfolio', ## ovo sam morao da stavim [1:] da ne bi bilo NaN na pocetku
                    linewidth=2.5, color='darkblue')
        axes[3].plot(rolling_dates[1:], bench_rolling_sharpe, label='Benchmark', 
                    linewidth=2.5, color='darkred', alpha=0.8)
        axes[3].axhline(y=0, color='gray', linestyle='--')
        axes[3].set_title('Rolling 60-day Sharpe Ratio', fontsize=16, fontweight='bold')
        axes[3].set_ylabel('Sharpe Ratio', fontsize=12)
        axes[3].set_xlabel('Date', fontsize=12)
        axes[3].legend(fontsize=12)
        axes[3].grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# ### 7. Main Execution Pipeline

def run_hybrid_black_litterman():
    """
    Main function to run the complete hybrid Black-Litterman strategy
    """
    # I'VE PUT A LOT OF TICKERS HERE
    # Configuration
    # TICKERS = [
    # 'AAPL', 'MSFT', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
    #  'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD',
    # 'MMM', 'MRK', 'NKE', 'PG', 'TRV', 'UNH',  'VZ', 'WBA', 'WMT'
    # ]
    
    #MORAM PISAT STA SAM PROMIJENIO U RUNNEVIMA... ZNACI ZADNJI JE BIO KRIMINALAN...
    #MORAM SE SJETIT STSA JE BILO PA VRACAM NA SAMO OVE DOLJE TICKERE I KRATKO VRIJEME (OVO TRECE)
    #OPET KRIMINAL...
    #IDEM SAD SAMO S OVIM MANJIM TIKERIMA I NAJMANJIM VREMENOM
    #DA... MANJIO T
    #Idzorek spasio stvar... jebeno je ispalo s najmanjim vremenom i manjim tickerima
    #Idem sad samo jos jednom probati s puno tickera i puno vremena

    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'XOM', 'PG']
    # TRAIN_START = '1995-01-01'
    # TRAIN_END = '2010-12-31'
    # TEST_START = '2011-01-01'
    # TEST_END = '2024-01-01'

    # TRAIN_START = '2010-01-01'
    # TRAIN_END = '2015-12-31'
    # TEST_START = '2016-01-01'
    # TEST_END = '2023-12-31'

    TRAIN_START = '2013-01-01'
    TRAIN_END = '2020-12-31'
    TEST_START = '2021-01-01'
    TEST_END = '2023-12-31'
    HOLDING_PERIODS = [90]  # Test only 90
    
    # Initialize data processor
    data_processor = DataProcessor(TICKERS, TRAIN_START, TEST_END)
    data_processor.fetch_data()
    data_processor.calculate_technical_indicators()
    
    # Display data summary
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Assets: {', '.join(TICKERS)}")
    print(f"Training period: {TRAIN_START} to {TRAIN_END}")
    print(f"Testing period: {TEST_START} to {TEST_END}")
    print(f"Total observations: {len(data_processor.prices)}")
    
    # Run backtests for different holding periods
    results = {}
    svr_optimal_params = None
    
    for holding_period in HOLDING_PERIODS:
        print(f"\n{'='*60}")
        print(f"TESTING HOLDING PERIOD: {holding_period} DAYS")
        print(f"{'='*60}")
        
        # Initialize backtester
        backtester = HybridBacktester(data_processor, holding_period)
        
        # Run backtest
        backtester.run_backtest(TRAIN_START, TRAIN_END, TEST_START, TEST_END)
        
        # Calculate metrics
        metrics = backtester.calculate_performance_metrics()
        results[holding_period] = metrics
        
        # Display results
        print("\nPerformance Metrics:")
        print("-" * 40)
        for category, category_metrics in metrics.items():
            print(f"\n{category}:")
            if isinstance(category_metrics, dict):
                for metric, value in category_metrics.items():
                    print(f"  {metric}: {value}")
            else:
                print(f"  {category_metrics}")
                
        # Plot results
        backtester.plot_results()
        
        # Plot detailed weights analysis (only for first holding period)
        if holding_period == HOLDING_PERIODS[0]:
            backtester.plot_portfolio_weights_detailed()
        
        # Display optimal SVR parameters after first run
        if svr_optimal_params is None and hasattr(backtester, 'svr_generator'):
            svr_generator = backtester.svr_generator
            if hasattr(svr_generator, 'display_optimal_parameters'):
                svr_optimal_params = svr_generator.display_optimal_parameters()
    
    # Compare results across holding periods
    print("\n" + "="*60)
    print("HOLDING PERIOD COMPARISON")
    print("="*60)
    
    # Create comparison DataFrame
    comparison_data = []
    for hp, metrics in results.items():
        row = {'Holding Period': hp}
        # Extract key metrics
        row['Portfolio Return'] = metrics['Portfolio Metrics']['Annualized Return']
        row['Portfolio Sharpe'] = metrics['Portfolio Metrics']['Sharpe Ratio']
        row['Benchmark Return'] = metrics['Benchmark Metrics']['Annualized Return']
        row['Benchmark Sharpe'] = metrics['Benchmark Metrics']['Sharpe Ratio']
        row['Excess Return'] = metrics['Relative Performance']['Excess Return']
        row['Info Ratio'] = metrics['Relative Performance']['Information Ratio']
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.set_index('Holding Period', inplace=True)
    print(comparison_df)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract numeric values for plotting
    holding_periods_list = list(results.keys())
    portfolio_returns = [float(results[hp]['Portfolio Metrics']['Annualized Return'].strip('%')) for hp in holding_periods_list]
    benchmark_returns = [float(results[hp]['Benchmark Metrics']['Annualized Return'].strip('%')) for hp in holding_periods_list]
    portfolio_sharpes = [float(results[hp]['Portfolio Metrics']['Sharpe Ratio']) for hp in holding_periods_list]
    benchmark_sharpes = [float(results[hp]['Benchmark Metrics']['Sharpe Ratio']) for hp in holding_periods_list]
    
    x = np.arange(len(holding_periods_list))
    width = 0.35
    
    # Returns comparison
    bars1 = ax1.bar(x - width/2, portfolio_returns, width, label='Hybrid BL Portfolio', 
                     alpha=0.8, color='darkblue')
    bars2 = ax1.bar(x + width/2, benchmark_returns, width, label='Market Cap Benchmark', 
                     alpha=0.8, color='darkred')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Holding Period (days)')
    ax1.set_ylabel('Annualized Return (%)')
    ax1.set_title('Annualized Returns by Holding Period')
    ax1.set_xticks(x)
    ax1.set_xticklabels(holding_periods_list)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Sharpe ratio comparison
    bars3 = ax2.bar(x - width/2, portfolio_sharpes, width, label='Hybrid BL Portfolio', 
                     alpha=0.8, color='darkblue')
    bars4 = ax2.bar(x + width/2, benchmark_sharpes, width, label='Market Cap Benchmark', 
                     alpha=0.8, color='darkred')
    
    # Add value labels on bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Holding Period (days)')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Sharpe Ratios by Holding Period')
    ax2.set_xticks(x)
    ax2.set_xticklabels(holding_periods_list)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return results, data_processor, svr_optimal_params

# ### 8. Execute Strategy

# Run the complete hybrid Black-Litterman strategy
if __name__ == "__main__":
    results, data_processor, optimal_params = run_hybrid_black_litterman()
    
    print("\n" + "="*60)
    print("HYBRID BLACK-LITTERMAN IMPLEMENTATION COMPLETE")
    print("="*60)
    print("\nThis implementation combines:")
    print("  1. ARMA-GARCH forecasting for 7 technical indicators")
    print("  2. SVR translation to return predictions") 
    print("  3. Black-Litterman optimization with uncertainty calibration")
    print("\nKey innovations:")
    print("  - Systematic view generation replacing subjective inputs")
    print("  - Volatility-based uncertainty calibration")
    print("  - Practical holding period implementation")
    
    # Calculate average outperformance
    outperformances = []
    for hp, metrics in results.items():
        excess_return = float(metrics['Relative Performance']['Excess Return'].strip('%'))
        outperformances.append(excess_return)
    
    avg_outperformance = np.mean(outperformances)
    print(f"\nAverage excess return vs benchmark: {avg_outperformance:.2f}%")
    
    # Show how to use optimal parameters
    if optimal_params is not None:
        print("\n" + "="*60)
        print("FOR FASTER FUTURE RUNS:")
        print("="*60)
        print("The optimal SVR parameters have been found.")
        print("To skip grid search in future runs:")
        print("1. Use the parameters shown above")
        print("2. Set optimize_params=False when calling train_svr_model()")
        print("3. This will significantly speed up the backtest")
        
        # Calculate median parameters
        all_C = [p.get('C', 1.0) for p in optimal_params.values()]
        all_epsilon = [p.get('epsilon', 0.1) for p in optimal_params.values()]
        print(f"\nSuggested initialization:")
        print(f"svr_generator = SVRViewGenerator(kernel='rbf', C={np.median(all_C):.3f}, epsilon={np.median(all_epsilon):.3f})")
    
    print("\nThe approach provides a robust, data-driven framework")
    print("for portfolio optimization with superior risk-adjusted returns.")