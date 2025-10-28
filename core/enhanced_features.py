"""
Enhanced Feature Engineering Pipeline - COMPREHENSIVE VERSION
Includes all advanced technical indicators requested by Karan
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import talib
    USE_TALIB = True
except ImportError:
    import ta
    USE_TALIB = False
    logging.warning("TA-Lib not found, using 'ta' library instead")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class EnhancedFeaturePipeline:
    """
    COMPREHENSIVE: Enhanced feature pipeline with all advanced indicators
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        feature_store_dir: str = "data/features"
    ):
        self.cache_dir = Path(cache_dir)
        self.feature_store_dir = Path(feature_store_dir)
        self.feature_store_dir.mkdir(parents=True, exist_ok=True)

        self.use_talib = USE_TALIB
        logger.info(f"Using {'TA-Lib' if self.use_talib else 'ta library'} for indicators")

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names from Yahoo Finance"""
        df = df.copy()

        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
        }

        df.columns = df.columns.str.lower()
        return df

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract feature columns (exclude OHLCV, metadata, targets)"""
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume', 'adj_close',
            'symbol', 'source', 'fetch_timestamp', 'date', 'Date',
            'target', 'target_return', 'target_direction', 'target_binary'
        ]
        return [col for col in df.columns if col.lower() not in [c.lower() for c in exclude_cols]]

    # ===================== MOVING AVERAGES =====================
    def compute_sma(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """Compute Simple Moving Averages"""
        for period in periods:
            if len(df) < period:
                df[f'sma_{period}'] = np.nan
                continue
            if self.use_talib:
                df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            else:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df

    def compute_ema(self, df: pd.DataFrame, periods: List[int] = [5, 10, 12, 20, 26, 50, 200]) -> pd.DataFrame:
        """Compute Exponential Moving Averages"""
        for period in periods:
            if len(df) < period:
                df[f'ema_{period}'] = np.nan
                continue
            if self.use_talib:
                df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            else:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df

    # ===================== MOMENTUM OSCILLATORS =====================
    def compute_rsi(self, df: pd.DataFrame, periods: List[int] = [14, 21]) -> pd.DataFrame:
        """Compute RSI indicators"""
        for period in periods:
            if len(df) < period + 1:
                df[f'rsi_{period}'] = np.nan
                continue
            if self.use_talib:
                df[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
            else:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        return df

    def compute_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Compute Stochastic Oscillator"""
        if len(df) < k_period:
            df['stoch_k'] = np.nan
            df['stoch_d'] = np.nan
            return df
        
        if self.use_talib:
            df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'], 
                                                      fastk_period=k_period, slowk_period=d_period)
        else:
            lowest_low = df['low'].rolling(window=k_period).min()
            highest_high = df['high'].rolling(window=k_period).max()
            df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
            df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        return df

    def compute_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Williams %R"""
        if len(df) < period:
            df['williams_r'] = np.nan
            return df
        
        if self.use_talib:
            df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=period)
        else:
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        return df

    def compute_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Compute Commodity Channel Index"""
        if len(df) < period:
            df['cci'] = np.nan
            return df
        
        if self.use_talib:
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=period)
        else:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        return df

    def compute_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Money Flow Index"""
        if len(df) < period:
            df['mfi'] = np.nan
            return df
        
        if self.use_talib:
            df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=period)
        else:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
            df['mfi'] = 100 - (100 / (1 + positive_flow / negative_flow))
        return df

    def compute_trix(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute TRIX (Triple Exponential Moving Average)"""
        if len(df) < period * 3:
            df['trix'] = np.nan
            return df
        
        if self.use_talib:
            df['trix'] = talib.TRIX(df['close'], timeperiod=period)
        else:
            ema1 = df['close'].ewm(span=period).mean()
            ema2 = ema1.ewm(span=period).mean()
            ema3 = ema2.ewm(span=period).mean()
            df['trix'] = ema3.pct_change() * 10000
        return df

    def compute_cmo(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Chande Momentum Oscillator"""
        if len(df) < period:
            df['cmo'] = np.nan
            return df
        
        if self.use_talib:
            df['cmo'] = talib.CMO(df['close'], timeperiod=period)
        else:
            delta = df['close'].diff()
            gains = delta.where(delta > 0, 0).rolling(window=period).sum()
            losses = (-delta.where(delta < 0, 0)).rolling(window=period).sum()
            df['cmo'] = 100 * (gains - losses) / (gains + losses)
        return df

    def compute_aroon(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Aroon Up/Down and Oscillator"""
        if len(df) < period:
            df['aroon_up'] = np.nan
            df['aroon_down'] = np.nan
            df['aroon_oscillator'] = np.nan
            return df
        
        if self.use_talib:
            df['aroon_up'], df['aroon_down'] = talib.AROON(df['high'], df['low'], timeperiod=period)
        else:
            high_periods = df['high'].rolling(window=period).apply(lambda x: x.argmax())
            low_periods = df['low'].rolling(window=period).apply(lambda x: x.argmin())
            df['aroon_up'] = 100 * (period - high_periods) / period
            df['aroon_down'] = 100 * (period - low_periods) / period
        
        df['aroon_oscillator'] = df['aroon_up'] - df['aroon_down']
        return df

    def compute_ultimate_oscillator(self, df: pd.DataFrame, periods: List[int] = [7, 14, 28]) -> pd.DataFrame:
        """Compute Ultimate Oscillator"""
        if len(df) < max(periods):
            df['ultimate_oscillator'] = np.nan
            return df
        
        if self.use_talib:
            df['ultimate_oscillator'] = talib.ULTOSC(df['high'], df['low'], df['close'], 
                                                   timeperiod1=periods[0], timeperiod2=periods[1], timeperiod3=periods[2])
        else:
            tr = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                   abs(df['low'] - df['close'].shift(1))))
            bp = df['close'] - np.minimum(df['low'], df['close'].shift(1))
            
            avg7 = bp.rolling(window=periods[0]).sum() / tr.rolling(window=periods[0]).sum()
            avg14 = bp.rolling(window=periods[1]).sum() / tr.rolling(window=periods[1]).sum()
            avg28 = bp.rolling(window=periods[2]).sum() / tr.rolling(window=periods[2]).sum()
            
            df['ultimate_oscillator'] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
        return df

    # ===================== TREND INDICATORS =====================
    def compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Average Directional Index"""
        if len(df) < period * 2:
            df['adx'] = np.nan
            df['plus_di'] = np.nan
            df['minus_di'] = np.nan
            return df
        
        if self.use_talib:
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=period)
            df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=period)
            df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=period)
        else:
            # Simplified ADX calculation
            high_diff = df['high'].diff()
            low_diff = df['low'].diff()
            tr = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                   abs(df['low'] - df['close'].shift(1))))
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            df['plus_di'] = 100 * pd.Series(plus_dm).rolling(window=period).mean() / tr.rolling(window=period).mean()
            df['minus_di'] = 100 * pd.Series(minus_dm).rolling(window=period).mean() / tr.rolling(window=period).mean()
            df['adx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        return df

    def compute_parabolic_sar(self, df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> pd.DataFrame:
        """Compute Parabolic SAR"""
        if len(df) < 2:
            df['psar'] = np.nan
            return df
        
        if self.use_talib:
            df['psar'] = talib.SAR(df['high'], df['low'], acceleration=acceleration, maximum=maximum)
        else:
            # Simplified Parabolic SAR
            df['psar'] = np.nan
            # This is a complex calculation - using simplified version
            df['psar'] = df['close'].rolling(window=5).mean()
        return df

    def compute_keltner_channels(self, df: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
        """Compute Keltner Channels"""
        if len(df) < period:
            df['keltner_upper'] = np.nan
            df['keltner_middle'] = np.nan
            df['keltner_lower'] = np.nan
            return df
        
        df['keltner_middle'] = df['close'].rolling(window=period).mean()
        tr = np.maximum(df['high'] - df['low'], 
                       np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                abs(df['low'] - df['close'].shift(1))))
        atr = tr.rolling(window=period).mean()
        
        df['keltner_upper'] = df['keltner_middle'] + (multiplier * atr)
        df['keltner_lower'] = df['keltner_middle'] - (multiplier * atr)
        return df

    def compute_donchian_channels(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Compute Donchian Channels"""
        if len(df) < period:
            df['donchian_upper'] = np.nan
            df['donchian_middle'] = np.nan
            df['donchian_lower'] = np.nan
            return df
        
        df['donchian_upper'] = df['high'].rolling(window=period).max()
        df['donchian_lower'] = df['low'].rolling(window=period).min()
        df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2
        return df

    # ===================== VOLATILITY INDICATORS =====================
    def compute_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        """Compute Bollinger Bands"""
        if len(df) < period:
            df['bb_upper'] = np.nan
            df['bb_middle'] = np.nan
            df['bb_lower'] = np.nan
            df['bb_width'] = np.nan
            df['bb_position'] = np.nan
            return df
        
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (std * bb_std)
        df['bb_lower'] = df['bb_middle'] - (std * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df

    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Average True Range"""
        if len(df) < period:
            df['atr'] = np.nan
            return df
        
        if self.use_talib:
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
        else:
            tr = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                   abs(df['low'] - df['close'].shift(1))))
            df['atr'] = pd.Series(tr).rolling(window=period).mean()
        return df

    def compute_standard_deviation(self, df: pd.DataFrame, periods: List[int] = [10, 20]) -> pd.DataFrame:
        """Compute Standard Deviation"""
        for period in periods:
            if len(df) < period:
                df[f'std_{period}'] = np.nan
                continue
            df[f'std_{period}'] = df['close'].rolling(window=period).std()
        return df

    # ===================== VOLUME INDICATORS =====================
    def compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-based features"""
        df['volume_change'] = df['volume'].pct_change()

        if len(df) >= 20:
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
        else:
            df['volume_sma_20'] = np.nan
            df['volume_ratio'] = np.nan

        if self.use_talib:
            df['obv'] = talib.OBV(df['close'], df['volume'])
        else:
            obv = [0]
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.append(obv[-1] + df['volume'].iloc[i])
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.append(obv[-1] - df['volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            df['obv'] = obv

        return df

    def compute_adl(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Accumulation/Distribution Line"""
        if len(df) < 2:
            df['adl'] = np.nan
            return df
        
        if self.use_talib:
            df['adl'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        else:
            clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            clv = clv.fillna(0)  # Handle division by zero
            df['adl'] = (clv * df['volume']).cumsum()
        return df

    def compute_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Compute Chaikin Money Flow"""
        if len(df) < period:
            df['cmf'] = np.nan
            return df
        
        if self.use_talib:
            df['cmf'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], 
                                   fastperiod=3, slowperiod=10)
        else:
            clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            clv = clv.fillna(0)
            money_flow_volume = clv * df['volume']
            df['cmf'] = money_flow_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return df

    def compute_emv(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Ease of Movement"""
        if len(df) < period:
            df['emv'] = np.nan
            return df
        
        distance_moved = (df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2
        box_height = df['volume'] / (df['high'] - df['low'])
        box_height = box_height.replace([np.inf, -np.inf], np.nan)
        df['emv'] = distance_moved / box_height
        df['emv'] = df['emv'].rolling(window=period).mean()
        return df

    def compute_volume_trend(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Volume Trend Analysis"""
        if len(df) < period:
            df['volume_trend'] = np.nan
            return df
        
        df['volume_trend'] = df['volume'].rolling(window=period).apply(
            lambda x: 1 if x.iloc[-1] > x.mean() else -1 if x.iloc[-1] < x.mean() else 0
        )
        return df

    # ===================== MACD =====================
    def compute_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Compute MACD"""
        if len(df) < slow:
            df['macd'] = np.nan
            df['macd_signal'] = np.nan
            df['macd_hist'] = np.nan
            return df
        
        if self.use_talib:
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'], 
                                                                       fastperiod=fast, 
                                                                       slowperiod=slow, 
                                                                       signalperiod=signal)
        else:
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=signal).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        return df

    # ===================== PRICE FEATURES =====================
    def compute_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price-based features"""
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(periods=5)
        df['price_change_20'] = df['close'].pct_change(periods=20)
        df['hl_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        if len(df) >= 10:
            df['volatility_10'] = df['price_change'].rolling(window=10).std()
        else:
            df['volatility_10'] = np.nan

        if len(df) >= 20:
            df['volatility_20'] = df['price_change'].rolling(window=20).std()
        else:
            df['volatility_20'] = np.nan

        return df

    # ===================== DERIVED/AGGREGATE FEATURES =====================
    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute higher-level, derived features used in analysis output"""
        # Moving Average Alignment: count of bullish alignments across common MAs
        try:
            sma_cols = [c for c in df.columns if c.startswith('sma_')]
            ema_cols = [c for c in df.columns if c.startswith('ema_')]
            for cols in [sma_cols, ema_cols]:
                if {'sma_20', 'sma_50', 'sma_200'}.issubset(df.columns):
                    df['ma_alignment_sma'] = ((df['sma_20'] > df['sma_50']) & (df['sma_50'] > df['sma_200'])).astype(int)
                if {'ema_12', 'ema_26', 'ema_50'}.issubset(df.columns):
                    df['ma_alignment_ema'] = ((df['ema_12'] > df['ema_26']) & (df['ema_26'] > df['ema_50'])).astype(int)
        except Exception:
            df['ma_alignment_sma'] = df.get('ma_alignment_sma', pd.Series(np.nan, index=df.index))
            df['ma_alignment_ema'] = df.get('ma_alignment_ema', pd.Series(np.nan, index=df.index))

        # Trend Direction: sign of short EMA slope
        try:
            if 'ema_20' in df.columns:
                df['trend_direction'] = np.sign(df['ema_20'] - df['ema_20'].shift(3)).replace({np.nan: 0})
            else:
                df['trend_direction'] = np.sign(df['close'] - df['close'].shift(3)).replace({np.nan: 0})
        except Exception:
            df['trend_direction'] = 0

        # Price-to-SMA ratios
        for p in [10, 20, 50, 200]:
            col = f'sma_{p}'
            ratio_col = f'price_to_sma_{p}'
            if col in df.columns:
                df[ratio_col] = df['close'] / (df[col] + 1e-10)
            else:
                df[ratio_col] = np.nan

        # Bollinger Band width already computed as bb_width
        # Position in range already computed as close_position

        # Volume Rate of Change (explicit)
        df['volume_roc_5'] = df['volume'].pct_change(5)

        # RSI Divergence (simple heuristic): price up while RSI down or vice versa over 5 periods
        try:
            if 'rsi_14' in df.columns:
                price_delta = df['close'] - df['close'].shift(5)
                rsi_delta = df['rsi_14'] - df['rsi_14'].shift(5)
                df['rsi_divergence'] = np.where(
                    ((price_delta > 0) & (rsi_delta < 0)) | ((price_delta < 0) & (rsi_delta > 0)),
                    1, 0
                )
            else:
                df['rsi_divergence'] = 0
        except Exception:
            df['rsi_divergence'] = 0

        # MACD Histogram Analysis: normalized histogram
        if 'macd_hist' in df.columns:
            hist_std = df['macd_hist'].rolling(20).std()
            df['macd_hist_z'] = df['macd_hist'] / (hist_std + 1e-10)
        else:
            df['macd_hist_z'] = np.nan

        return df

    def compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum indicators"""
        for period in [5, 10, 20]:
            if len(df) < period:
                df[f'roc_{period}'] = np.nan
                continue
            if self.use_talib:
                df[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)
            else:
                df[f'roc_{period}'] = df['close'].pct_change(periods=period) * 100

        if len(df) >= 10:
            if self.use_talib:
                df['momentum'] = talib.MOM(df['close'], timeperiod=10)
            else:
                df['momentum'] = df['close'] - df['close'].shift(10)
        else:
            df['momentum'] = np.nan

        return df

    # ===================== SUPPORT/RESISTANCE =====================
    def compute_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Pivot Points"""
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['resistance_1'] = 2 * df['pivot'] - df['low']
        df['support_1'] = 2 * df['pivot'] - df['high']
        df['resistance_2'] = df['pivot'] + (df['high'] - df['low'])
        df['support_2'] = df['pivot'] - (df['high'] - df['low'])
        return df

    def compute_fibonacci_levels(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Compute Fibonacci Retracement Levels"""
        if len(df) < period:
            df['fib_23.6'] = np.nan
            df['fib_38.2'] = np.nan
            df['fib_50.0'] = np.nan
            df['fib_61.8'] = np.nan
            return df
        
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        range_size = high_max - low_min
        
        df['fib_23.6'] = high_max - 0.236 * range_size
        df['fib_38.2'] = high_max - 0.382 * range_size
        df['fib_50.0'] = high_max - 0.500 * range_size
        df['fib_61.8'] = high_max - 0.618 * range_size
        return df

    # ===================== CANDLESTICK PATTERNS =====================
    def compute_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Candlestick Patterns"""
        if len(df) < 3:
            df['doji'] = 0
            df['hammer'] = 0
            df['engulfing'] = 0
            return df
        
        # Doji Pattern
        body_size = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        df['doji'] = (body_size / (total_range + 1e-10) < 0.1).astype(int)
        
        # Hammer Pattern
        lower_shadow = df['open'].combine(df['close'], min) - df['low']
        upper_shadow = df['high'] - df['open'].combine(df['close'], max)
        body = abs(df['close'] - df['open'])
        df['hammer'] = ((lower_shadow > 2 * body) & (upper_shadow < body)).astype(int)
        
        # Engulfing Pattern (simplified)
        prev_body = abs(df['close'].shift(1) - df['open'].shift(1))
        curr_body = abs(df['close'] - df['open'])
        df['engulfing'] = (curr_body > prev_body).astype(int)
        
        return df

    # ===================== ADVANCED ANALYSIS =====================
    def compute_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Volume Weighted Average Price"""
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        return df

    def compute_sharpe_ratio(self, df: pd.DataFrame, period: int = 20, risk_free_rate: float = 0.02) -> pd.DataFrame:
        """Compute Sharpe Ratio"""
        if len(df) < period:
            df['sharpe_ratio'] = np.nan
            return df
        
        returns = df['close'].pct_change()
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        df['sharpe_ratio'] = excess_returns.rolling(window=period).mean() / returns.rolling(window=period).std()
        return df

    def compute_beta_alpha(self, df: pd.DataFrame, benchmark_returns: pd.Series = None, period: int = 20) -> pd.DataFrame:
        """Compute Beta and Alpha (simplified - would need benchmark data)"""
        if len(df) < period:
            df['beta'] = np.nan
            df['alpha'] = np.nan
            return df
        
        returns = df['close'].pct_change()
        if benchmark_returns is not None:
            covariance = returns.rolling(window=period).cov(benchmark_returns)
            benchmark_variance = benchmark_returns.rolling(window=period).var()
            df['beta'] = covariance / benchmark_variance
            df['alpha'] = returns.rolling(window=period).mean() - df['beta'] * benchmark_returns.rolling(window=period).mean()
        else:
            df['beta'] = 1.0  # Default beta
            df['alpha'] = 0.0  # Default alpha
        return df

    # ===================== MAIN COMPUTATION METHOD =====================
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators and features"""
        logger.info(f"Computing comprehensive features for {len(df)} rows")

        df = self._standardize_columns(df)

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if len(df) < 50:
            logger.warning(f"Limited data ({len(df)} rows). Features may not be reliable.")

        try:
            # Moving Averages
            df = self.compute_sma(df)
            df = self.compute_ema(df)
            
            # Momentum Oscillators
            df = self.compute_rsi(df)
            df = self.compute_stochastic(df)
            df = self.compute_williams_r(df)
            df = self.compute_cci(df)
            df = self.compute_mfi(df)
            df = self.compute_trix(df)
            df = self.compute_cmo(df)
            df = self.compute_aroon(df)
            df = self.compute_ultimate_oscillator(df)
            
            # Trend Indicators
            df = self.compute_adx(df)
            df = self.compute_parabolic_sar(df)
            df = self.compute_keltner_channels(df)
            df = self.compute_donchian_channels(df)
            
            # Volatility Indicators
            df = self.compute_bollinger_bands(df)
            df = self.compute_atr(df)
            df = self.compute_standard_deviation(df)
            
            # Volume Indicators
            df = self.compute_volume_features(df)
            df = self.compute_adl(df)
            df = self.compute_cmf(df)
            df = self.compute_emv(df)
            df = self.compute_volume_trend(df)
            
            # MACD
            df = self.compute_macd(df)
            
            # Price Features
            df = self.compute_price_features(df)
            df = self.compute_momentum_features(df)

            # Derived/Aggregate Features
            df = self.compute_derived_features(df)
            
            # Support/Resistance
            df = self.compute_pivot_points(df)
            df = self.compute_fibonacci_levels(df)
            
            # Candlestick Patterns
            df = self.compute_candlestick_patterns(df)
            
            # Advanced Analysis
            df = self.compute_vwap(df)
            df = self.compute_sharpe_ratio(df)
            df = self.compute_beta_alpha(df)
            
        except Exception as e:
            logger.error(f"Error computing features: {e}", exc_info=True)
            raise

        # Clean up data
        initial_rows = len(df)
        feature_cols = self._get_feature_columns(df)
        df = df.dropna(subset=feature_cols, how='all')
        df[feature_cols] = df[feature_cols].ffill().bfill()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()

        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with insufficient data")

        df = self.generate_targets(df)

        logger.info(f"Comprehensive feature computation complete: {len(df)} rows, {df.shape[1]} features")
        return df

    def generate_targets(self, df: pd.DataFrame, lookahead_days: int = 5) -> pd.DataFrame:
        """Generate target variables for supervised learning"""
        df = df.copy()
        
        # Future returns
        df['target_return'] = df['close'].shift(-lookahead_days) / df['close'] - 1
        
        # Direction targets
        df['target_direction'] = np.where(df['target_return'] > 0.02, 1, 
                                        np.where(df['target_return'] < -0.02, -1, 0))
        
        # Binary classification
        df['target_binary'] = (df['target_return'] > 0).astype(int)
        
        return df

    def load_feature_store(self) -> Dict[str, pd.DataFrame]:
        """Load the feature store"""
        feature_store = {}
        feature_dir = Path(self.feature_store_dir)
        
        if not feature_dir.exists():
            logger.warning("Feature store directory does not exist")
            return feature_store
        
        for file_path in feature_dir.glob("*.parquet"):
            symbol = file_path.stem
            try:
                df = pd.read_parquet(file_path)
                feature_store[symbol] = df
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
        
        logger.info(f"Loaded {len(feature_store)} symbols from feature store")
        return feature_store

    def save_feature_store(self, feature_store: Dict[str, pd.DataFrame]):
        """Save the feature store"""
        feature_dir = Path(self.feature_store_dir)
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        for symbol, df in feature_store.items():
            try:
                file_path = feature_dir / f"{symbol}.parquet"
                df.to_parquet(file_path)
            except Exception as e:
                logger.error(f"Error saving {symbol}: {e}")
        
        logger.info(f"Saved {len(feature_store)} symbols to feature store")
