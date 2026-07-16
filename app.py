"""
MSA — Market Sentiment Analyzer
FastAPI backend for stock analysis deployed on Hugging Face Spaces.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup
from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
HF_API_TOKEN: str = os.environ.get("HF_API_TOKEN", "")
ZOYA_API_KEY: str = os.environ.get("ZOYA_API_KEY", "")
ZOYA_GRAPHQL_URL = "https://sandbox-api.zoya.finance/graphql"
CACHE_TTL_SECONDS: int = 300
FINBERT_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("msa")

app = FastAPI(
    title="MSA — Market Sentiment Analyzer",
    description=(
        "Stock analysis API: historical data, sentiment, news, and GPT-4 insights. "
        "Results are cached for 5 minutes per ticker."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_analysis_cache: TTLCache[str, dict[str, Any]] = TTLCache(
    maxsize=256, ttl=CACHE_TTL_SECONDS
)

# raw OHLCV history cache — shared by /analyze, /montecarlo, /backtest,
# /position-size and /optimize so one ticker is only fetched once per TTL
_history_cache: TTLCache[str, list[dict[str, Any]]] = TTLCache(
    maxsize=256, ttl=CACHE_TTL_SECONDS
)


class MovingAverages(BaseModel):
    sma_50: float | None = Field(None, description="50-day Simple Moving Average")
    sma_200: float | None = Field(None, description="200-day Simple Moving Average")
    signal: str | None = Field(
        None,
        description="'Golden Cross' if SMA-50 > SMA-200, 'Death Cross' otherwise",
    )


class FearGreed(BaseModel):
    value: int = Field(50, description="Fear & Greed index (0-100)")
    label: str = Field("Neutral", description="Human-readable label, e.g. 'Greed'")


class NewsHeadline(BaseModel):
    title: str
    publisher: str | None = None
    link: str | None = None
    published: str | None = None
    sentiment: str | None = Field(None, description="FinBERT: positive / negative / neutral")
    sentiment_score: float | None = Field(None, description="FinBERT confidence 0-1")


class NewsSentiment(BaseModel):
    positive: int = Field(0, description="Number of positive headlines")
    negative: int = Field(0, description="Number of negative headlines")
    neutral: int = Field(0, description="Number of neutral headlines")
    avg_score: float = Field(0.5, description="Average sentiment score (0=bearish, 1=bullish)")
    label: str = Field("Neutral", description="Overall: Bullish / Bearish / Neutral / Mixed")


class TechnicalIndicators(BaseModel):
    rsi_14: float | None = Field(None, description="14-day RSI")
    rsi_signal: str | None = Field(None, description="Oversold / Neutral / Overbought")
    macd: float | None = Field(None, description="MACD line value")
    macd_signal: float | None = Field(None, description="MACD signal line value")
    macd_histogram: float | None = Field(None, description="MACD histogram")
    macd_trend: str | None = Field(None, description="Bullish / Bearish")
    overall_signal: str | None = Field(None, description="Buy / Sell / Neutral")
    score: int = Field(50, description="0-100 gauge score: 0=Strong Sell, 100=Strong Buy")


class AnalystRating(BaseModel):
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0
    total: int = 0
    recommendation: str | None = Field(None, description="Overall recommendation label")
    score: int = Field(50, description="0-100 gauge score: 0=Strong Sell, 100=Strong Buy")


class PriceTarget(BaseModel):
    current: float | None = None
    daily_change: float | None = Field(None, description="Today's $ change")
    daily_change_pct: float | None = Field(None, description="Today's % change")
    target_mean: float | None = None
    target_high: float | None = None
    target_low: float | None = None
    upside_pct: float | None = Field(None, description="% upside to mean target")


class BollingerBands(BaseModel):
    upper: float | None = Field(None, description="Upper Bollinger Band (SMA20 + 2*stddev)")
    middle: float | None = Field(None, description="Middle band (SMA 20)")
    lower: float | None = Field(None, description="Lower Bollinger Band (SMA20 - 2*stddev)")
    bandwidth: float | None = Field(None, description="Band width as % of middle band")
    position: str | None = Field(None, description="Price position: Upper Band / Mid Band / Lower Band")
    squeeze: bool = Field(False, description="True if bandwidth is historically low (squeeze)")


class OBVAnalysis(BaseModel):
    obv_current: float | None = Field(None, description="Current On-Balance Volume")
    obv_trend: str | None = Field(None, description="Rising / Falling / Flat")
    price_trend: str | None = Field(None, description="Rising / Falling / Flat (last 20 days)")
    divergence: str | None = Field(None, description="Accumulation / Distribution / Confirmation / None")


class VolumeProfile(BaseModel):
    avg_volume_20d: int | None = Field(None, description="20-day average volume")
    latest_volume: int | None = Field(None, description="Most recent day's volume")
    volume_ratio: float | None = Field(None, description="Latest volume / 20-day avg")
    spike: bool = Field(False, description="True if volume > 1.5x average")
    spike_days: int = Field(0, description="Number of spike days in last 20")


class DenoisedTrend(BaseModel):
    slope: float | None = Field(None, description="Denoised price velocity (% per day)")
    slope_direction: str | None = Field(None, description="Rising / Falling / Flat")
    acceleration: float | None = Field(None, description="Rate of velocity change (% per day²)")
    momentum_exhaustion: bool = Field(False, description="True when denoised slope diverges from RSI")
    exhaustion_type: str | None = Field(None, description="Bullish Exhaustion / Bearish Exhaustion")
    denoised_prices: list[float] = Field(default_factory=list, description="Last 30 denoised closing prices (oldest first)")


class ZScoreAnalysis(BaseModel):
    zscore: float | None = Field(None, description="Current 20-day rolling Z-Score")
    mean_20d: float | None = Field(None, description="20-day rolling mean price")
    stddev_20d: float | None = Field(None, description="20-day rolling standard deviation")
    signal: str | None = Field(None, description="Statistical signal")
    reversal_probability: float | None = Field(None, description="Probability of mean reversion (0-100%)")


class InferenceWeights(BaseModel):
    technical_score: float = Field(50, description="Technical component (0-100) — 35% weight")
    sentiment_score: float = Field(50, description="Sentiment component (0-100) — 25% weight")
    analyst_score: float = Field(50, description="Analyst component (0-100) — 20% weight")
    volume_score: float = Field(50, description="Volume component (0-100) — 20% weight")
    composite_score: float = Field(50, description="Final weighted composite (0-100)")
    composite_signal: str = Field("Neutral", description="Buy / Sell / Hold / Watch based on composite")


class GPTInsight(BaseModel):
    actionable_insight: str = Field(..., description="GPT-4 actionable recommendation")
    confidence_score: int = Field(
        ..., ge=0, le=100, description="Confidence score 0-100"
    )
    reasoning: str = Field(..., description="Brief reasoning behind the recommendation")


class PricePoint(BaseModel):
    date: str
    close: float


class Range6M(BaseModel):
    high: float | None = Field(None, description="Highest intraday price over ~6 months (126 trading days)")
    high_date: str | None = None
    low: float | None = Field(None, description="Lowest intraday price over ~6 months")
    low_date: str | None = None


def calculate_range_6m(daily_rows: list[dict[str, Any]]) -> Range6M:
    """Peak / trough over the last ~126 trading days. daily_rows[0] = newest."""
    window = daily_rows[:126]
    if not window:
        return Range6M()
    hi = max(window, key=lambda r: r["high"])
    lo = min(window, key=lambda r: r["low"])
    return Range6M(
        high=round(hi["high"], 2), high_date=hi["date"],
        low=round(lo["low"], 2), low_date=lo["date"],
    )


class AnalysisResponse(BaseModel):
    ticker: str
    timestamp: str
    cached: bool = Field(False, description="True if this result came from cache")
    moving_averages: MovingAverages
    technicals: TechnicalIndicators | None = None
    bollinger_bands: BollingerBands | None = None
    obv_analysis: OBVAnalysis | None = None
    volume_profile: VolumeProfile | None = None
    denoised_trend: DenoisedTrend | None = None
    zscore_analysis: ZScoreAnalysis | None = None
    inference_weights: InferenceWeights | None = None
    analyst_ratings: AnalystRating | None = None
    price_target: PriceTarget | None = None
    fear_greed: FearGreed
    news: list[NewsHeadline]
    news_sentiment: NewsSentiment | None = None
    gpt_analysis: GPTInsight | None = None
    price_history: list[PricePoint] = Field(default_factory=list, description="Last 30 days of closing prices (oldest first)")
    range_6m: Range6M | None = Field(None, description="6-month high / low with dates")


def _yf_fetch_history(ticker: str, period: str = "1y") -> list[dict[str, Any]]:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")

    rows: list[dict[str, Any]] = []
    for date, row in df.iterrows():
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
            }
        )

    rows.reverse()
    return rows


async def fetch_daily_ohlcv(ticker: str, period: str = "1y") -> list[dict[str, Any]]:
    key = f"{ticker.upper()}::{period}"
    if key in _history_cache:
        return _history_cache[key]
    try:
        rows = await asyncio.to_thread(_yf_fetch_history, ticker.upper(), period)
        _history_cache[key] = rows
        return rows
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch historical data for {ticker}: {exc}",
        ) from exc


def calculate_smas(daily_rows: list[dict[str, Any]]) -> MovingAverages:
    closes = [r["close"] for r in daily_rows]

    sma_50: float | None = None
    sma_200: float | None = None

    if len(closes) >= 50:
        sma_50 = round(sum(closes[:50]) / 50, 4)
    if len(closes) >= 200:
        sma_200 = round(sum(closes[:200]) / 200, 4)

    signal: str | None = None
    if sma_50 is not None and sma_200 is not None:
        signal = "Golden Cross" if sma_50 > sma_200 else "Death Cross"

    return MovingAverages(sma_50=sma_50, sma_200=sma_200, signal=signal)


def calculate_technicals(daily_rows: list[dict[str, Any]]) -> TechnicalIndicators:
    closes = [r["close"] for r in daily_rows]

    # RSI-14
    rsi_14: float | None = None
    rsi_signal: str | None = None
    if len(closes) >= 15:
        gains, losses = [], []
        for i in range(1, 15):
            delta = closes[i - 1] - closes[i]  # rows are newest-first
            gains.append(max(delta, 0))
            losses.append(max(-delta, 0))
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        if avg_loss == 0:
            rsi_14 = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_14 = round(100 - (100 / (1 + rs)), 2)
        if rsi_14 <= 30:
            rsi_signal = "Oversold"
        elif rsi_14 >= 70:
            rsi_signal = "Overbought"
        else:
            rsi_signal = "Neutral"

    # MACD (12, 26, 9) — we need at least ~35 days of data
    macd_val: float | None = None
    macd_sig: float | None = None
    macd_hist: float | None = None
    macd_trend: str | None = None

    if len(closes) >= 35:
        ordered = list(reversed(closes))  # oldest-first for EMA calc

        def ema(data: list[float], span: int) -> list[float]:
            k = 2 / (span + 1)
            result = [data[0]]
            for price in data[1:]:
                result.append(price * k + result[-1] * (1 - k))
            return result

        ema12 = ema(ordered, 12)
        ema26 = ema(ordered, 26)
        macd_line = [a - b for a, b in zip(ema12[25:], ema26[25:])]

        if len(macd_line) >= 9:
            signal_line = ema(macd_line, 9)
            macd_val = round(macd_line[-1], 4)
            macd_sig = round(signal_line[-1], 4)
            macd_hist = round(macd_val - macd_sig, 4)
            macd_trend = "Bullish" if macd_hist > 0 else "Bearish"

    # gauge score: map RSI + MACD into 0-100 (0=Strong Sell, 100=Strong Buy)
    # RSI contributes: <20 → +2, 20-30 → +1, 30-70 → 0, 70-80 → -1, >80 → -2
    rsi_pts = 0
    if rsi_14 is not None:
        if rsi_14 < 20: rsi_pts = 2
        elif rsi_14 < 30: rsi_pts = 1
        elif rsi_14 > 80: rsi_pts = -2
        elif rsi_14 > 70: rsi_pts = -1

    macd_pts = 0
    if macd_hist is not None:
        if macd_hist > 0: macd_pts = 1
        else: macd_pts = -1

    raw = rsi_pts + macd_pts  # range [-3, +3]
    gauge = int(((raw + 3) / 6) * 100)
    gauge = max(0, min(100, gauge))

    if gauge >= 70:
        overall = "Buy" if gauge < 85 else "Strong Buy"
    elif gauge <= 30:
        overall = "Sell" if gauge > 15 else "Strong Sell"
    else:
        overall = "Neutral"

    return TechnicalIndicators(
        rsi_14=rsi_14,
        rsi_signal=rsi_signal,
        macd=macd_val,
        macd_signal=macd_sig,
        macd_histogram=macd_hist,
        macd_trend=macd_trend,
        overall_signal=overall,
        score=gauge,
    )


def calculate_bollinger(daily_rows: list[dict[str, Any]]) -> BollingerBands:
    """Bollinger Bands: 20-period SMA +/- 2 standard deviations. daily_rows[0] = newest."""
    try:
        ordered = [r["close"] for r in reversed(daily_rows)]  # oldest first
        if len(ordered) < 20:
            return BollingerBands()

        window = ordered[-20:]
        sma20 = sum(window) / 20
        variance = sum((x - sma20) ** 2 for x in window) / 20
        stddev = variance ** 0.5

        upper = round(sma20 + 2 * stddev, 2)
        lower = round(sma20 - 2 * stddev, 2)
        middle = round(sma20, 2)
        bw = round((upper - lower) / middle * 100, 2) if middle else None

        # detect squeeze: compare current bandwidth to avg of last 100 days' bandwidths
        squeeze = False
        if len(ordered) >= 100:
            bws = []
            for i in range(20, min(len(ordered) + 1, 101)):
                w = ordered[i - 20:i]
                m = sum(w) / 20
                sd = (sum((x - m) ** 2 for x in w) / 20) ** 0.5
                if m > 0:
                    bws.append((m + 2 * sd - (m - 2 * sd)) / m * 100)
            if bws and bw is not None:
                avg_bw = sum(bws) / len(bws)
                squeeze = bw < avg_bw * 0.5

        current_price = ordered[-1]
        if current_price >= upper * 0.98:
            position = "Upper Band"
        elif current_price <= lower * 1.02:
            position = "Lower Band"
        else:
            position = "Mid Band"

        return BollingerBands(
            upper=upper, middle=middle, lower=lower,
            bandwidth=bw, position=position, squeeze=squeeze,
        )
    except Exception as exc:
        logger.warning("Bollinger Bands calculation failed: %s", exc)
        return BollingerBands()


def calculate_obv(daily_rows: list[dict[str, Any]]) -> OBVAnalysis:
    """On-Balance Volume with trend and divergence detection. daily_rows[0] = newest."""
    try:
        ordered = list(reversed(daily_rows))  # oldest first
        if len(ordered) < 20:
            return OBVAnalysis()

        # build OBV series
        obv = [0.0]
        for i in range(1, len(ordered)):
            if ordered[i]["close"] > ordered[i - 1]["close"]:
                obv.append(obv[-1] + ordered[i]["volume"])
            elif ordered[i]["close"] < ordered[i - 1]["close"]:
                obv.append(obv[-1] - ordered[i]["volume"])
            else:
                obv.append(obv[-1])

        obv_now = obv[-1]

        # OBV trend over last 20 days: linear regression slope direction
        obv_20 = obv[-20:]
        obv_slope = obv_20[-1] - obv_20[0]
        if obv_slope > 0:
            obv_trend = "Rising"
        elif obv_slope < 0:
            obv_trend = "Falling"
        else:
            obv_trend = "Flat"

        # price trend over last 20 days
        closes_20 = [r["close"] for r in ordered[-20:]]
        price_slope = closes_20[-1] - closes_20[0]
        pct_change = abs(price_slope) / closes_20[0] * 100 if closes_20[0] else 0
        if pct_change < 2:
            price_trend = "Flat"
        elif price_slope > 0:
            price_trend = "Rising"
        else:
            price_trend = "Falling"

        # divergence detection
        if obv_trend == "Rising" and price_trend == "Flat":
            divergence = "Accumulation"
        elif obv_trend == "Rising" and price_trend == "Falling":
            divergence = "Accumulation"
        elif obv_trend == "Falling" and price_trend == "Rising":
            divergence = "Distribution"
        elif obv_trend == "Falling" and price_trend == "Flat":
            divergence = "Distribution"
        elif obv_trend == price_trend:
            divergence = "Confirmation"
        else:
            divergence = "None"

        return OBVAnalysis(
            obv_current=round(obv_now),
            obv_trend=obv_trend,
            price_trend=price_trend,
            divergence=divergence,
        )
    except Exception as exc:
        logger.warning("OBV calculation failed: %s", exc)
        return OBVAnalysis()


def calculate_volume_profile(daily_rows: list[dict[str, Any]]) -> VolumeProfile:
    """Volume analysis: 20-day average, spikes, ratio. daily_rows[0] = newest."""
    try:
        if len(daily_rows) < 20:
            return VolumeProfile()

        last_20 = daily_rows[:20]  # newest 20 days
        volumes = [r["volume"] for r in last_20]
        avg_vol = int(sum(volumes) / 20)
        latest_vol = volumes[0]
        ratio = round(latest_vol / avg_vol, 2) if avg_vol > 0 else 0
        spike = ratio > 1.5
        spike_days = sum(1 for v in volumes if avg_vol > 0 and v > avg_vol * 1.5)

        return VolumeProfile(
            avg_volume_20d=avg_vol,
            latest_volume=latest_vol,
            volume_ratio=ratio,
            spike=spike,
            spike_days=spike_days,
        )
    except Exception as exc:
        logger.warning("Volume profile calculation failed: %s", exc)
        return VolumeProfile()


BULL_KEYWORDS = {
    "surge", "soar", "rally", "gain", "beat", "upgrade", "rise", "jump",
    "bull", "record", "high", "buy", "outperform", "breakout", "growth",
}
BEAR_KEYWORDS = {
    "crash", "fall", "drop", "plunge", "sell", "downgrade", "cut", "loss",
    "bear", "low", "miss", "warn", "risk", "decline", "slump", "fear",
}


def _savgol_smooth(prices: np.ndarray, window: int = 21, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay polynomial smoothing via numpy (no scipy dependency).
    Computes convolution coefficients from the Vandermonde pseudo-inverse."""
    n = len(prices)
    if n < window:
        window = n if n % 2 == 1 else n - 1
    if window < polyorder + 2:
        return prices.copy()
    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    A = np.vander(x, N=polyorder + 1, increasing=True)
    smooth_coeffs = np.linalg.pinv(A)[0]
    padded = np.pad(prices, half, mode="edge")
    return np.convolve(padded, smooth_coeffs[::-1], mode="valid")


def calculate_denoised_trend(
    daily_rows: list[dict[str, Any]], rsi: float | None,
) -> DenoisedTrend:
    """Savitzky-Golay denoising to extract price velocity and momentum exhaustion."""
    try:
        ordered = [r["close"] for r in reversed(daily_rows)]
        n = len(ordered)
        if n < 30:
            return DenoisedTrend()

        prices = np.array(ordered, dtype=float)
        smoothed = _savgol_smooth(prices)

        recent = smoothed[-10:]
        x10 = np.arange(10, dtype=float)
        fit = np.polyfit(x10, recent, 1)
        slope = float(fit[0])
        slope_pct = (slope / float(prices[-1])) * 100 if prices[-1] > 0 else 0.0

        accel_pct = 0.0
        if n >= 30:
            prev = smoothed[-20:-10]
            prev_fit = np.polyfit(np.arange(10, dtype=float), prev, 1)
            accel = slope - float(prev_fit[0])
            accel_pct = (accel / float(prices[-1])) * 100 if prices[-1] > 0 else 0.0

        if abs(slope_pct) < 0.05:
            direction = "Flat"
        elif slope_pct > 0:
            direction = "Rising"
        else:
            direction = "Falling"

        exhaustion = False
        exhaustion_type = None
        if rsi is not None:
            if slope_pct > 0.1 and rsi > 65 and accel_pct < -0.01:
                exhaustion = True
                exhaustion_type = "Bullish Exhaustion"
            elif slope_pct < -0.1 and rsi < 35 and accel_pct > 0.01:
                exhaustion = True
                exhaustion_type = "Bearish Exhaustion"

        hist_len = min(30, n)
        denoised_last = smoothed[-hist_len:].tolist()

        return DenoisedTrend(
            slope=round(slope_pct, 4),
            slope_direction=direction,
            acceleration=round(accel_pct, 4),
            momentum_exhaustion=exhaustion,
            exhaustion_type=exhaustion_type,
            denoised_prices=[round(float(p), 2) for p in denoised_last],
        )
    except Exception as exc:
        logger.warning("Denoised trend calculation failed: %s", exc)
        return DenoisedTrend()


def calculate_zscore(daily_rows: list[dict[str, Any]]) -> ZScoreAnalysis:
    """20-day rolling Z-Score for statistical mean reversion signals."""
    try:
        ordered = [r["close"] for r in reversed(daily_rows)]
        if len(ordered) < 20:
            return ZScoreAnalysis()

        window = ordered[-20:]
        mean = sum(window) / 20
        variance = sum((x - mean) ** 2 for x in window) / 20
        stddev = variance ** 0.5

        current = ordered[-1]
        zscore = (current - mean) / stddev if stddev > 0 else 0.0

        abs_z = abs(zscore)
        reversal_prob = round(math.erf(abs_z / math.sqrt(2)) * 100, 1)

        if zscore > 2.0:
            signal = "Overbought — Pullback Likely (>2σ)"
        elif zscore > 1.5:
            signal = "Stretched — Watch for Reversal"
        elif zscore < -2.0:
            signal = "Oversold — Bounce Likely (>2σ)"
        elif zscore < -1.5:
            signal = "Compressed — Watch for Bounce"
        else:
            signal = "Normal Range"

        return ZScoreAnalysis(
            zscore=round(zscore, 3),
            mean_20d=round(mean, 2),
            stddev_20d=round(stddev, 2),
            signal=signal,
            reversal_probability=reversal_prob,
        )
    except Exception as exc:
        logger.warning("Z-Score calculation failed: %s", exc)
        return ZScoreAnalysis()


def calculate_inference_weights(
    technicals: TechnicalIndicators,
    denoised: DenoisedTrend,
    zscore: ZScoreAnalysis,
    fear_greed: FearGreed,
    news_sent: NewsSentiment,
    analyst: AnalystRating,
    obv: OBVAnalysis,
    volume: VolumeProfile,
) -> InferenceWeights:
    """Pre-calculate the 4-pillar weighted inference scores."""
    rsi_macd = technicals.score if technicals else 50
    slope_score = 50.0
    if denoised.slope is not None:
        slope_score = max(0.0, min(100.0, 50 + denoised.slope * 25))

    z_component = 50.0
    if zscore.zscore is not None:
        z_component = max(0.0, min(100.0, 50 - zscore.zscore * 20))

    technical_score = rsi_macd * 0.40 + slope_score * 0.35 + z_component * 0.25

    contrarian_fg = 100 - fear_greed.value
    finbert_score = news_sent.avg_score * 100
    sentiment_score = contrarian_fg * 0.60 + finbert_score * 0.40

    analyst_score_val = float(analyst.score) if analyst else 50.0

    vol_base = 50.0
    if obv.divergence == "Accumulation":
        vol_base = 80.0
    elif obv.divergence == "Distribution":
        vol_base = 20.0
    elif obv.divergence == "Confirmation":
        vol_base = 65.0 if obv.obv_trend == "Rising" else 35.0

    if volume and volume.volume_ratio:
        if volume.volume_ratio > 1.5 and obv.obv_trend == "Rising":
            vol_base = min(100, vol_base + 12)
        elif volume.volume_ratio > 1.5 and obv.obv_trend == "Falling":
            vol_base = max(0, vol_base - 12)

    composite = (
        technical_score * 0.35
        + sentiment_score * 0.25
        + analyst_score_val * 0.20
        + vol_base * 0.20
    )

    if composite >= 70:
        sig = "Buy"
    elif composite >= 55:
        sig = "Hold"
    elif composite >= 40:
        sig = "Watch"
    else:
        sig = "Sell"

    return InferenceWeights(
        technical_score=round(technical_score, 1),
        sentiment_score=round(sentiment_score, 1),
        analyst_score=round(analyst_score_val, 1),
        volume_score=round(vol_base, 1),
        composite_score=round(composite, 1),
        composite_signal=sig,
    )


def _yf_fetch_analyst_ratings(ticker: str) -> AnalystRating:
    try:
        stock = yf.Ticker(ticker.upper())

        rec = None
        for attr in ("recommendations_summary", "recommendations"):
            try:
                df = getattr(stock, attr)
                if df is not None and not df.empty:
                    rec = df
                    break
            except Exception as exc:
                logger.warning("Analyst %s fetch failed for %s: %s", attr, ticker, exc)

        sb = b = h = s = ss = 0
        if rec is not None:
            # modern yfinance returns rows for periods 0m, -1m, -2m, -3m —
            # the CURRENT month is '0m' (first row), not iloc[-1]
            latest = rec.iloc[0]
            if "period" in rec.columns and (rec["period"] == "0m").any():
                latest = rec[rec["period"] == "0m"].iloc[0]
            sb = int(latest.get("strongBuy", 0) or 0)
            b = int(latest.get("buy", 0) or 0)
            h = int(latest.get("hold", 0) or 0)
            s = int(latest.get("sell", 0) or 0)
            ss = int(latest.get("strongSell", 0) or 0)
        total = sb + b + h + s + ss

        if total == 0:
            # fallback: consensus fields from the quote summary
            try:
                info = stock.info or {}
            except Exception:
                info = {}
            n = int(info.get("numberOfAnalystOpinions") or 0)
            mean = info.get("recommendationMean")  # 1 = Strong Buy … 5 = Strong Sell
            key = (info.get("recommendationKey") or "").replace("_", " ").title().strip()
            if n > 0 and mean:
                a_score = max(0, min(100, int(((5 - float(mean)) / 4) * 100)))
                return AnalystRating(
                    total=n,
                    recommendation=key or None,
                    score=a_score,
                )
            return AnalystRating()

        if total == 0:
            label = "No Data"
        else:
            buy_pct = (sb + b) / total
            sell_pct = (s + ss) / total
            if buy_pct >= 0.6:
                label = "Strong Buy" if sb > b else "Buy"
            elif sell_pct >= 0.6:
                label = "Strong Sell" if ss > s else "Sell"
            else:
                label = "Hold"

        # weighted score: strong_buy=5, buy=4, hold=3, sell=2, strong_sell=1
        if total > 0:
            weighted = (sb * 5 + b * 4 + h * 3 + s * 2 + ss * 1) / total
            a_score = int(((weighted - 1) / 4) * 100)
            a_score = max(0, min(100, a_score))
        else:
            a_score = 50

        return AnalystRating(
            strong_buy=sb, buy=b, hold=h, sell=s, strong_sell=ss,
            total=total, recommendation=label, score=a_score,
        )
    except Exception as exc:
        logger.warning("Analyst ratings fetch failed for %s: %s", ticker, exc)
        return AnalystRating()


def _yf_fetch_price_target(ticker: str) -> PriceTarget:
    try:
        stock = yf.Ticker(ticker.upper())

        # 1) dedicated analyst-targets endpoint (most reliable on recent yfinance)
        targets: dict[str, Any] = {}
        try:
            t = stock.analyst_price_targets
            if isinstance(t, dict):
                targets = t
        except Exception as exc:
            logger.warning("analyst_price_targets failed for %s: %s", ticker, exc)

        # 2) quote summary as secondary source
        info: dict[str, Any] = {}
        try:
            info = stock.info or {}
        except Exception as exc:
            logger.warning("info fetch failed for %s: %s", ticker, exc)

        current = (
            targets.get("current")
            or info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        daily_chg = info.get("regularMarketChange")
        daily_chg_pct = info.get("regularMarketChangePercent")

        # 3) fast_info for live price / previous close
        if current is None or daily_chg_pct is None:
            try:
                fi = stock.fast_info
                last = getattr(fi, "last_price", None)
                prev = getattr(fi, "previous_close", None)
                if current is None and last:
                    current = float(last)
                if daily_chg_pct is None and last and prev:
                    daily_chg = float(last) - float(prev)
                    daily_chg_pct = (daily_chg / float(prev)) * 100
            except Exception as exc:
                logger.warning("fast_info failed for %s: %s", ticker, exc)

        # 4) last resort: compute from recent daily closes
        if current is None or daily_chg_pct is None:
            try:
                hist = stock.history(period="5d")
                closes = hist["Close"].dropna()
                if len(closes) >= 1 and current is None:
                    current = float(closes.iloc[-1])
                if len(closes) >= 2 and daily_chg_pct is None:
                    daily_chg = float(closes.iloc[-1]) - float(closes.iloc[-2])
                    daily_chg_pct = daily_chg / float(closes.iloc[-2]) * 100
            except Exception as exc:
                logger.warning("history fallback failed for %s: %s", ticker, exc)

        target_mean = (
            targets.get("mean")
            or targets.get("median")
            or info.get("targetMeanPrice")
            or info.get("targetMedianPrice")
        )
        target_high = (
            targets.get("high")
            or info.get("targetHighPrice")
            or info.get("targetMaxPrice")
        )
        target_low = (
            targets.get("low")
            or info.get("targetLowPrice")
            or info.get("targetMinPrice")
        )

        upside = None
        if current and target_mean:
            try:
                upside = round(((target_mean - current) / current) * 100, 2)
            except (TypeError, ZeroDivisionError):
                pass

        return PriceTarget(
            current=round(float(current), 2) if current else None,
            daily_change=round(float(daily_chg), 2) if daily_chg is not None else None,
            daily_change_pct=round(float(daily_chg_pct), 2) if daily_chg_pct is not None else None,
            target_mean=round(float(target_mean), 2) if target_mean else None,
            target_high=round(float(target_high), 2) if target_high else None,
            target_low=round(float(target_low), 2) if target_low else None,
            upside_pct=upside,
        )
    except Exception as exc:
        logger.warning("Price target fetch failed for %s: %s", ticker, exc)
        return PriceTarget()


async def fetch_analyst_ratings(ticker: str) -> AnalystRating:
    return await asyncio.to_thread(_yf_fetch_analyst_ratings, ticker.upper())


async def fetch_price_target(ticker: str) -> PriceTarget:
    return await asyncio.to_thread(_yf_fetch_price_target, ticker.upper())


FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
_NEUTRAL_FALLBACK = FearGreed(value=50, label="Neutral")


def _fg_label(value: int) -> str:
    if value <= 25:
        return "Extreme Fear"
    if value <= 45:
        return "Fear"
    if value <= 55:
        return "Neutral"
    if value <= 75:
        return "Greed"
    return "Extreme Greed"


async def fetch_fear_greed() -> FearGreed:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(FEAR_GREED_URL, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        fg = data.get("fear_and_greed", {})
        score = fg.get("score")
        rating = fg.get("rating")

        if score is not None:
            return FearGreed(value=int(round(score)), label=rating or _fg_label(int(round(score))))
    except Exception as exc:
        logger.warning("Fear & Greed primary fetch failed: %s", exc)

    try:
        fallback_url = "https://money.cnn.com/data/fear-and-greed/"
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(fallback_url, headers=headers)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        needle = soup.find("div", id="needleChart")
        if needle:
            score_text = needle.find("li")
            if score_text:
                val = int("".join(c for c in score_text.text if c.isdigit())[:3])
                return FearGreed(value=val, label=_fg_label(val))
    except Exception as exc2:
        logger.warning("Fear & Greed fallback failed: %s", exc2)

    logger.warning("Returning Neutral (50) default for Fear & Greed")
    return _NEUTRAL_FALLBACK


def _yf_fetch_news(ticker: str, max_items: int = 10) -> list[NewsHeadline]:
    try:
        stock = yf.Ticker(ticker.upper())
        raw_news = stock.news or []
    except Exception as exc:
        logger.warning("yfinance news error for %s: %s", ticker, exc)
        return []

    headlines: list[NewsHeadline] = []
    for item in raw_news[:max_items]:
        content = item.get("content", item)

        title = content.get("title", "")
        publisher = None
        provider = content.get("provider")
        if isinstance(provider, dict):
            publisher = provider.get("displayName")

        link = None
        canonical = content.get("canonicalUrl") or content.get("clickThroughUrl")
        if isinstance(canonical, dict):
            link = canonical.get("url")

        published_iso = content.get("pubDate")

        if title:
            headlines.append(
                NewsHeadline(
                    title=title,
                    publisher=publisher,
                    link=link,
                    published=published_iso,
                )
            )
    return headlines


async def fetch_news(ticker: str, max_items: int = 10) -> list[NewsHeadline]:
    return await asyncio.to_thread(_yf_fetch_news, ticker.upper(), max_items)


async def finbert_sentiment(headlines: list[NewsHeadline]) -> NewsSentiment:
    """Run headlines through ProsusAI/finbert via HF Inference API."""
    if not headlines:
        return NewsSentiment()

    titles = [h.title for h in headlines if h.title]
    if not titles:
        return NewsSentiment()

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                FINBERT_URL,
                json={"inputs": titles, "options": {"wait_for_model": True}},
                headers=headers,
            )
            resp.raise_for_status()
            results = resp.json()

        pos, neg, neu = 0, 0, 0
        score_sum = 0.0

        for i, result in enumerate(results):
            if not isinstance(result, list) or not result:
                continue

            top = max(result, key=lambda x: x.get("score", 0))
            label = top.get("label", "neutral").lower()
            conf = top.get("score", 0)

            if label == "positive":
                pos += 1
                score_sum += 0.5 + conf * 0.5
            elif label == "negative":
                neg += 1
                score_sum += 0.5 - conf * 0.5
            else:
                neu += 1
                score_sum += 0.5

            if i < len(headlines):
                headlines[i].sentiment = label
                headlines[i].sentiment_score = round(conf, 3)

        total = pos + neg + neu
        avg = score_sum / total if total > 0 else 0.5

        if pos > neg * 2:
            overall = "Bullish"
        elif neg > pos * 2:
            overall = "Bearish"
        elif pos > neg:
            overall = "Slightly Bullish"
        elif neg > pos:
            overall = "Slightly Bearish"
        elif pos == 0 and neg == 0:
            overall = "Neutral"
        else:
            overall = "Mixed"

        return NewsSentiment(
            positive=pos, negative=neg, neutral=neu,
            avg_score=round(avg, 3), label=overall,
        )
    except Exception as exc:
        logger.warning("FinBERT sentiment analysis failed: %s — falling back to keyword method", exc)
        return _keyword_sentiment_fallback(headlines)


def _keyword_sentiment_fallback(headlines: list[NewsHeadline]) -> NewsSentiment:
    """Fallback to keyword matching if FinBERT API is unavailable."""
    pos, neg, neu = 0, 0, 0
    for h in headlines:
        title = h.title.lower()
        if any(k in title for k in BULL_KEYWORDS):
            pos += 1
            h.sentiment = "positive"
            h.sentiment_score = 0.7
        elif any(k in title for k in BEAR_KEYWORDS):
            neg += 1
            h.sentiment = "negative"
            h.sentiment_score = 0.7
        else:
            neu += 1
            h.sentiment = "neutral"
            h.sentiment_score = 0.5

    total = max(pos + neg + neu, 1)
    avg = (pos * 0.8 + neu * 0.5 + neg * 0.2) / total

    if pos > neg * 2:
        overall = "Bullish"
    elif neg > pos * 2:
        overall = "Bearish"
    elif pos > neg:
        overall = "Slightly Bullish"
    elif neg > pos:
        overall = "Slightly Bearish"
    else:
        overall = "Neutral"

    return NewsSentiment(
        positive=pos, negative=neg, neutral=neu,
        avg_score=round(avg, 3), label=overall,
    )


async def fetch_stock_data(
    ticker: str,
) -> tuple[list[dict[str, Any]], list[NewsHeadline]]:
    history, news = await asyncio.gather(
        fetch_daily_ohlcv(ticker),
        fetch_news(ticker),
    )
    return history, news


async def gpt_analysis(
    ticker: str,
    weights: InferenceWeights,
    denoised: DenoisedTrend,
    zscore: ZScoreAnalysis,
    smas: MovingAverages,
    technicals: TechnicalIndicators,
    bollinger: BollingerBands,
    obv: OBVAnalysis,
    volume: VolumeProfile,
    analyst: AnalystRating,
    fear_greed: FearGreed,
    news_sent: NewsSentiment,
    headlines: list[NewsHeadline],
) -> GPTInsight:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured.")

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    news_text = "\n".join(
        f"- [{h.sentiment or 'unknown'} {h.sentiment_score or 0:.0%}] {h.title} ({h.publisher})"
        for h in headlines
    ) or "No recent headlines available."

    payload = {
        "ticker": ticker.upper(),
        "inference_weights": {
            "technical_score": weights.technical_score,
            "sentiment_score": weights.sentiment_score,
            "analyst_score": weights.analyst_score,
            "volume_score": weights.volume_score,
            "composite_score": weights.composite_score,
            "composite_signal": weights.composite_signal,
            "formula": "35% Technical + 25% Sentiment + 20% Analyst + 20% Volume",
        },
        "denoised_trend": {
            "savitzky_golay_slope_pct_per_day": denoised.slope,
            "slope_direction": denoised.slope_direction,
            "acceleration_pct_per_day2": denoised.acceleration,
            "momentum_exhaustion": denoised.momentum_exhaustion,
            "exhaustion_type": denoised.exhaustion_type,
        },
        "zscore_20d": {
            "value": zscore.zscore,
            "mean": zscore.mean_20d,
            "stddev": zscore.stddev_20d,
            "signal": zscore.signal,
            "reversal_probability_pct": zscore.reversal_probability,
        },
        "moving_averages": {
            "sma_50": smas.sma_50,
            "sma_200": smas.sma_200,
            "signal": smas.signal,
        },
        "technical_indicators": {
            "rsi_14": technicals.rsi_14,
            "rsi_signal": technicals.rsi_signal,
            "macd_histogram": technicals.macd_histogram,
            "macd_trend": technicals.macd_trend,
        },
        "bollinger_bands": {
            "upper": bollinger.upper,
            "lower": bollinger.lower,
            "bandwidth_pct": bollinger.bandwidth,
            "price_position": bollinger.position,
            "squeeze_detected": bollinger.squeeze,
        },
        "on_balance_volume": {
            "obv_trend": obv.obv_trend,
            "price_trend_20d": obv.price_trend,
            "divergence": obv.divergence,
        },
        "volume_profile": {
            "volume_ratio": volume.volume_ratio,
            "volume_spike": volume.spike,
        },
        "analyst_consensus": {
            "recommendation": analyst.recommendation,
            "score": analyst.score,
        },
        "fear_and_greed": {
            "value": fear_greed.value,
            "label": fear_greed.label,
        },
        "finbert_news_sentiment": {
            "positive_headlines": news_sent.positive,
            "negative_headlines": news_sent.negative,
            "neutral_headlines": news_sent.neutral,
            "avg_score": news_sent.avg_score,
            "overall_label": news_sent.label,
        },
        "recent_news_headlines": news_text,
    }

    system_prompt = (
        "You are a JSON-only response engine. Do not include markdown formatting "
        "or prose outside the JSON object.\n\n"
        "Role: Hardline Mathematical Quantitative Analyst. You operate on "
        "statistically significant signals, not opinions. Every claim must be "
        "anchored to a numerical threshold or mathematical relationship.\n\n"
        "You will receive a pre-calculated inference payload with weighted scores:\n"
        "1. INFERENCE WEIGHTS (pre-computed): Technical (35%), Sentiment (25%), "
        "Analyst (20%), Volume (20%). The composite_score is your baseline — "
        "adjust ±10 points max based on qualitative headline analysis.\n"
        "2. DENOISED PRICE VELOCITY: Savitzky-Golay polynomial regression "
        "(window=21, order=3) removes market noise. The slope (% per day) "
        "represents true price velocity. If slope is positive but acceleration "
        "is negative, the trend is mathematically decelerating.\n"
        "3. 20-DAY Z-SCORE: Statistical distance from the rolling mean in σ. "
        "|Z| > 2.0 = 95.4% statistical significance for mean reversion. "
        "|Z| > 1.5 = 86.6% probability. This is your primary reversal signal.\n"
        "4. Classical technicals: RSI-14, MACD, SMA crossovers, Bollinger Bands.\n"
        "5. OBV divergence + volume confirmation.\n"
        "6. FinBERT NLP Sentiment: Per-headline sentiment labels (positive/negative/neutral) "
        "from ProsusAI/finbert transformer model. Aggregate avg_score 0-1 (0=bearish, 1=bullish).\n"
        "7. Analyst consensus + Fear & Greed (contrarian-adjusted).\n\n"
        "DECISION FRAMEWORK:\n"
        "- Z-Score > +2.0 AND denoised slope decelerating → SELL "
        "(mean reversion imminent, statistically significant).\n"
        "- Z-Score < -2.0 AND OBV rising → BUY "
        "(oversold with accumulation, 95% reversion probability).\n"
        "- Denoised slope positive + acceleration positive + Z < 1.5 → BUY "
        "(healthy trend with statistical room to run).\n"
        "- Momentum Exhaustion detected → reversal is mathematically likely, "
        "WATCH or counter-trade.\n"
        "- Bollinger Squeeze + Z near 0 + dry volume → WATCH "
        "(breakout imminent, wait for directional confirmation).\n"
        "- OBV divergence overrides price action: Accumulation = stealth buying, "
        "Distribution = smart money exiting.\n\n"
        "Your reasoning MUST reference: 'Denoised Price Velocity', "
        "'Statistical Significance', and specific Z-Score / slope values.\n\n"
        "Respond with ONLY this JSON:\n"
        "{\n"
        '  "actionable_insight": "<Buy|Hold|Sell|Watch> — explanation citing '
        'denoised velocity and Z-Score thresholds",\n'
        '  "confidence_score": <int 0-100>,\n'
        '  "reasoning": "<2-3 sentences. MUST reference denoised slope, Z-Score, '
        'statistical significance, and weight contributions>"\n'
        "}"
    )

    user_message = (
        "Analyze the following market data and produce your recommendation:\n\n"
        + json.dumps(payload, indent=2)
    )

    completion = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=600,
    )

    raw = (completion.choices[0].message.content or "{}").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("GPT-4 returned unparseable JSON: %s", raw)
        parsed = {
            "actionable_insight": raw[:500],
            "confidence_score": 0,
            "reasoning": "Failed to parse structured response from GPT-4.",
        }

    return GPTInsight(
        actionable_insight=parsed.get("actionable_insight", "N/A"),
        confidence_score=max(0, min(100, int(parsed.get("confidence_score", 0)))),
        reasoning=parsed.get("reasoning", ""),
    )


@app.get("/")
async def root():
    return {
        "service": "MSA — Market Sentiment Analyzer",
        "version": "3.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "cache_size": len(_analysis_cache)}


@app.get("/analyze", response_model=AnalysisResponse)
async def analyze(
    ticker: str = Query(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol, e.g. AAPL",
    ),
):
    ticker = ticker.upper().strip()

    if ticker in _analysis_cache:
        logger.info("Cache HIT for %s", ticker)
        cached_data = _analysis_cache[ticker].copy()
        cached_data["cached"] = True
        return AnalysisResponse(**cached_data)

    logger.info("Cache MISS — starting analysis for %s", ticker)
    t0 = time.perf_counter()

    (daily_rows, news), fg, analyst, pt = await asyncio.gather(
        fetch_stock_data(ticker),
        fetch_fear_greed(),
        fetch_analyst_ratings(ticker),
        fetch_price_target(ticker),
    )

    news_sent = await finbert_sentiment(news)

    smas = calculate_smas(daily_rows)
    technicals = calculate_technicals(daily_rows)
    bb = calculate_bollinger(daily_rows)
    obv = calculate_obv(daily_rows)
    vol_profile = calculate_volume_profile(daily_rows)
    denoised = calculate_denoised_trend(daily_rows, technicals.rsi_14)
    zsc = calculate_zscore(daily_rows)
    weights = calculate_inference_weights(
        technicals, denoised, zsc, fg, news_sent, analyst, obv, vol_profile,
    )

    gpt_result: GPTInsight | None = None
    try:
        gpt_result = await gpt_analysis(
            ticker, weights, denoised, zsc,
            smas, technicals, bb, obv, vol_profile, analyst, fg, news_sent, news,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("GPT-4 analysis failed: %s", exc)

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info("Analysis for %s completed in %.2fs", ticker, elapsed)

    # last 30 days of closing prices, oldest first for charting
    recent = daily_rows[:30]
    recent.reverse()
    history = [PricePoint(date=r["date"], close=round(r["close"], 2)) for r in recent]

    result = AnalysisResponse(
        ticker=ticker,
        timestamp=datetime.now(timezone.utc).isoformat(),
        cached=False,
        moving_averages=smas,
        technicals=technicals,
        bollinger_bands=bb,
        obv_analysis=obv,
        volume_profile=vol_profile,
        denoised_trend=denoised,
        zscore_analysis=zsc,
        inference_weights=weights,
        analyst_ratings=analyst,
        price_target=pt,
        fear_greed=fg,
        news=news,
        news_sentiment=news_sent,
        gpt_analysis=gpt_result,
        price_history=history,
        range_6m=calculate_range_6m(daily_rows),
    )

    _analysis_cache[ticker] = result.model_dump()

    return result


@app.get("/sma", response_model=MovingAverages)
async def sma_only(
    ticker: str = Query(..., min_length=1, max_length=10),
):
    daily = await fetch_daily_ohlcv(ticker.upper().strip())
    return calculate_smas(daily)


@app.get("/fear-greed", response_model=FearGreed)
async def fear_greed_only():
    return await fetch_fear_greed()


@app.get("/news", response_model=list[NewsHeadline])
async def news_only(
    ticker: str = Query(..., min_length=1, max_length=10),
):
    return await fetch_news(ticker.upper().strip())


@app.delete("/cache/{ticker}")
async def clear_cache(ticker: str):
    key = ticker.upper().strip()
    removed = _analysis_cache.pop(key, None) is not None
    return {"ticker": key, "removed": removed, "cache_size": len(_analysis_cache)}


@app.delete("/cache")
async def clear_all_cache():
    count = len(_analysis_cache)
    _analysis_cache.clear()
    return {"cleared": count}


# ═════════════════════════════════════════════════════════════════════════
#  v4 — QUANT TERMINAL LAYER
#  Monte Carlo (GBM) · Position Sizing · Shariah Screen (beta) ·
#  Halal Portfolio Optimizer (beta) · Walk-forward Backtest ·
#  Arabic Agentic Copilot (GPT-4o tool-calling, anti-hallucination)
# ═════════════════════════════════════════════════════════════════════════

import hashlib
from datetime import date

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04  # annual, used for Sharpe / frontier


def _is_saudi(ticker: str) -> bool:
    return ticker.upper().endswith(".SR")


def _currency_for(ticker: str) -> str:
    return "SAR" if _is_saudi(ticker) else "USD"


def _closes_oldest_first(daily_rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([r["close"] for r in reversed(daily_rows)], dtype=float)


def _log_returns(closes: np.ndarray) -> np.ndarray:
    return np.diff(np.log(closes))


# ─────────────────────────────────────────────────────────────────────────
#  1. Monte Carlo — seeded GBM, drift/vol estimated from real returns
# ─────────────────────────────────────────────────────────────────────────

class MonteCarloResult(BaseModel):
    ticker: str
    currency: str
    spot: float
    horizon_days: int = Field(..., description="Trading-day horizon")
    n_paths: int
    seed: int = Field(..., description="Deterministic seed (ticker+date) — same result on refresh")
    mu_annual: float = Field(..., description="Annualized drift estimated from 1y of real log returns")
    sigma_annual: float = Field(..., description="Annualized volatility from 1y of real log returns")
    p05: list[float]
    p25: list[float]
    p50: list[float]
    p75: list[float]
    p95: list[float]
    sample_paths: list[list[float]] = Field(default_factory=list, description="Thinned real simulation paths for rendering")
    expected_price: float = Field(..., description="Mean terminal price across all paths")
    var_95_pct: float = Field(..., description="95% Value-at-Risk over horizon, % of spot (loss, positive number)")
    var_95_price: float
    cvar_95_pct: float = Field(..., description="95% Conditional VaR (expected shortfall), % of spot")
    cvar_95_price: float
    target: float
    prob_end_above_target: float = Field(..., description="P(terminal price ≥ target), %")
    prob_touch_target: float = Field(..., description="P(path touches target at any point), %")
    method: str = "GBM: S_t = S₀·exp((μ−σ²/2)t + σ√t·Z) — μ,σ from real 1y daily log returns"


def _run_monte_carlo(
    closes: np.ndarray, spot: float, horizon: int, n_paths: int, seed: int, target: float,
) -> dict[str, Any]:
    rets = _log_returns(closes)
    mu_d = float(np.mean(rets))
    sigma_d = float(np.std(rets, ddof=1))

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, horizon))
    increments = (mu_d - 0.5 * sigma_d**2) + sigma_d * z
    log_paths = np.cumsum(increments, axis=1)
    paths = spot * np.exp(log_paths)
    paths = np.concatenate([np.full((n_paths, 1), spot), paths], axis=1)

    pcts = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    terminal = paths[:, -1]
    term_ret = terminal / spot - 1.0

    q05 = float(np.percentile(term_ret, 5))
    var_pct = max(0.0, -q05)
    tail = term_ret[term_ret <= q05]
    cvar_pct = max(0.0, -float(tail.mean())) if tail.size else var_pct

    prob_end = float((terminal >= target).mean()) * 100
    prob_touch = float((paths.max(axis=1) >= target).mean()) * 100

    # a handful of real paths, thinned, purely for visual texture
    sample = paths[:: max(1, n_paths // 24)][:24]

    return {
        "mu_annual": round(mu_d * TRADING_DAYS, 4),
        "sigma_annual": round(sigma_d * math.sqrt(TRADING_DAYS), 4),
        "p05": [round(float(x), 2) for x in pcts[0]],
        "p25": [round(float(x), 2) for x in pcts[1]],
        "p50": [round(float(x), 2) for x in pcts[2]],
        "p75": [round(float(x), 2) for x in pcts[3]],
        "p95": [round(float(x), 2) for x in pcts[4]],
        "sample_paths": [[round(float(x), 2) for x in p] for p in sample],
        "expected_price": round(float(terminal.mean()), 2),
        "var_95_pct": round(var_pct * 100, 2),
        "var_95_price": round(spot * (1 - var_pct), 2),
        "cvar_95_pct": round(cvar_pct * 100, 2),
        "cvar_95_price": round(spot * (1 - cvar_pct), 2),
        "prob_end_above_target": round(prob_end, 1),
        "prob_touch_target": round(prob_touch, 1),
    }


@app.get("/montecarlo", response_model=MonteCarloResult)
async def monte_carlo(
    ticker: str = Query(..., min_length=1, max_length=12),
    horizon_days: int = Query(63, ge=10, le=252, description="Trading days to simulate (63 ≈ 3 months)"),
    n_paths: int = Query(10_000, ge=1000, le=20_000),
    target: float | None = Query(None, gt=0, description="Target price; default = spot × 1.10"),
    seed: int | None = Query(None, description="Override the deterministic daily seed"),
):
    ticker = ticker.upper().strip()
    daily_rows = await fetch_daily_ohlcv(ticker)
    closes = _closes_oldest_first(daily_rows)
    if closes.size < 60:
        raise HTTPException(status_code=422, detail=f"Not enough history for {ticker} to estimate drift/volatility.")

    spot = float(closes[-1])
    tgt = target if target else round(spot * 1.10, 2)
    if seed is None:
        seed = int(hashlib.sha256(f"{ticker}-{date.today().isoformat()}".encode()).hexdigest()[:8], 16)

    result = await asyncio.to_thread(_run_monte_carlo, closes, spot, horizon_days, n_paths, seed, tgt)

    return MonteCarloResult(
        ticker=ticker,
        currency=_currency_for(ticker),
        spot=round(spot, 2),
        horizon_days=horizon_days,
        n_paths=n_paths,
        seed=seed,
        target=tgt,
        **result,
    )


# ─────────────────────────────────────────────────────────────────────────
#  2. Position sizing — volatility targeting + capped fractional Kelly
# ─────────────────────────────────────────────────────────────────────────

RISK_PROFILES: dict[str, dict[str, float | str]] = {
    "conservative": {"label_ar": "محافظ", "target_vol": 0.10, "multiplier": 0.75, "cap": 0.10},
    "balanced": {"label_ar": "متوازن", "target_vol": 0.15, "multiplier": 1.00, "cap": 0.15},
    "aggressive": {"label_ar": "جريء", "target_vol": 0.20, "multiplier": 1.25, "cap": 0.25},
}


class PositionSizeResult(BaseModel):
    ticker: str
    currency: str
    price: float
    profile: str
    profile_ar: str
    composite_score_used: float
    score_source: str = Field(..., description="'client' if passed by caller, 'cache' from /analyze cache, 'neutral' fallback")
    sigma_annual: float
    atr_14: float
    vol_target_weight: float = Field(..., description="target_vol / realized_vol, capped at 100%")
    p_win: float = Field(..., description="Win probability mapped from composite score (calibration: beta)")
    payoff_ratio: float = Field(..., description="avg 5-day gain / avg 5-day loss from real history")
    kelly_full: float
    kelly_quarter: float = Field(..., description="¼ Kelly (fractional Kelly, industry-standard damping)")
    recommended_pct: float = Field(..., description="Final recommendation, % of capital")
    cap_pct: float
    stop_price: float = Field(..., description="Entry − 2×ATR(14)")
    stop_pct: float
    shares: int | None = None
    position_value: float | None = None
    formulas: list[str]
    notes_ar: str


@app.get("/position-size", response_model=PositionSizeResult)
async def position_size(
    ticker: str = Query(..., min_length=1, max_length=12),
    profile: str = Query("balanced", description="conservative | balanced | aggressive"),
    score: float | None = Query(None, ge=0, le=100, description="Composite score from /analyze (single source of truth)"),
    capital: float | None = Query(None, gt=0, description="Optional capital to convert % into shares"),
):
    ticker = ticker.upper().strip()
    prof = RISK_PROFILES.get(profile.lower())
    if prof is None:
        raise HTTPException(status_code=422, detail="profile must be conservative | balanced | aggressive")

    daily_rows = await fetch_daily_ohlcv(ticker)
    closes = _closes_oldest_first(daily_rows)
    if closes.size < 30:
        raise HTTPException(status_code=422, detail=f"Not enough history for {ticker}.")

    price = float(closes[-1])
    rets = _log_returns(closes)
    sigma_ann = float(np.std(rets, ddof=1)) * math.sqrt(TRADING_DAYS)

    # ATR(14) — true range on real OHLC
    ordered = list(reversed(daily_rows))
    trs = []
    for i in range(1, len(ordered)):
        h, l, pc = ordered[i]["high"], ordered[i]["low"], ordered[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = float(np.mean(trs[-14:])) if len(trs) >= 14 else float(np.mean(trs))

    # composite score: prefer explicit param → /analyze cache → neutral 50
    score_source = "client"
    if score is None:
        cached = _analysis_cache.get(ticker)
        iw = (cached or {}).get("inference_weights") or {}
        if iw.get("composite_score") is not None:
            score = float(iw["composite_score"])
            score_source = "cache"
        else:
            score = 50.0
            score_source = "neutral"

    # volatility targeting
    w_vol = min(1.0, prof["target_vol"] / sigma_ann) if sigma_ann > 0 else 0.0

    # fractional Kelly — p mapped from composite score (calibration in progress → beta)
    p_win = max(0.35, min(0.65, 0.5 + (score - 50) / 100 * 0.30))
    fwd = closes[5:] / closes[:-5] - 1.0
    gains, losses = fwd[fwd > 0], fwd[fwd < 0]
    payoff = float(gains.mean() / abs(losses.mean())) if gains.size and losses.size and losses.mean() != 0 else 1.0
    kelly_full = p_win - (1 - p_win) / payoff if payoff > 0 else 0.0
    kelly_quarter = max(0.0, kelly_full) * 0.25

    if kelly_full <= 0:
        recommended = 0.0
        note = "الإشارة الحالية لا تدعم فتح مركز — كِيلي سالب، القيمة الموصى بها 0%."
    else:
        recommended = min(w_vol, kelly_quarter) * prof["multiplier"]
        recommended = min(recommended, prof["cap"])
        note = "الحجم = الأصغر بين استهداف التقلب و¼ كيلي، مضروبًا بمعامل ملف المخاطرة، بسقف صارم."

    stop = price - 2 * atr
    stop_pct = (price - stop) / price * 100

    shares = pos_val = None
    if capital:
        pos_val = round(capital * recommended, 2)
        shares = int(pos_val / price) if price > 0 else 0

    return PositionSizeResult(
        ticker=ticker,
        currency=_currency_for(ticker),
        price=round(price, 2),
        profile=profile.lower(),
        profile_ar=str(prof["label_ar"]),
        composite_score_used=round(score, 1),
        score_source=score_source,
        sigma_annual=round(sigma_ann, 4),
        atr_14=round(atr, 2),
        vol_target_weight=round(w_vol, 4),
        p_win=round(p_win, 3),
        payoff_ratio=round(payoff, 3),
        kelly_full=round(kelly_full, 4),
        kelly_quarter=round(kelly_quarter, 4),
        recommended_pct=round(recommended * 100, 2),
        cap_pct=round(prof["cap"] * 100, 1),
        stop_price=round(stop, 2),
        stop_pct=round(stop_pct, 2),
        shares=shares,
        position_value=pos_val,
        formulas=[
            f"w_vol = σ_target / σ_realized = {prof['target_vol']:.0%} / {sigma_ann:.1%} = {w_vol:.1%}",
            f"Kelly f* = p − (1−p)/b = {p_win:.2f} − {1-p_win:.2f}/{payoff:.2f} = {kelly_full:.1%}",
            f"¼ Kelly = {kelly_quarter:.1%}",
            f"الحجم النهائي = min(w_vol, ¼Kelly) × {prof['multiplier']} ≤ سقف {prof['cap']:.0%} = {recommended:.1%}",
            f"وقف الخسارة = السعر − 2×ATR(14) = {price:.2f} − 2×{atr:.2f} = {stop:.2f}",
        ],
        notes_ar=note + " احتمال الربح p مشتق من الدرجة المركّبة — المعايرة التاريخية قيد التنفيذ (تجريبي).",
    )


# ─────────────────────────────────────────────────────────────────────────
#  3. Shariah screening — AAOIFI-style ratios (BETA: fundamentals feed WIP)
# ─────────────────────────────────────────────────────────────────────────

# TODO(data): non-compliant income % requires a segment-revenue feed
# (IdealRatings / Refinitiv). Until wired, the income screen and purification
# rate are returned as null and the whole endpoint is flagged beta=true.

AAOIFI_THRESHOLD = 30.0  # % of market cap, per AAOIFI screening standard

_NON_COMPLIANT_INDUSTRIES: dict[str, str] = {
    "banks": "بنوك تقليدية (فوائد ربوية)",
    "insurance": "تأمين تقليدي",
    "credit services": "خدمات ائتمان بفوائد",
    "mortgage": "تمويل عقاري ربوي",
    "brewers": "مشروبات كحولية",
    "wineries": "مشروبات كحولية",
    "distilleries": "مشروبات كحولية",
    "tobacco": "تبغ",
    "casinos": "قمار ومَيسِر",
    "gambling": "قمار ومَيسِر",
    "gaming": "قمار ومَيسِر",
}


class ShariahScreenResult(BaseModel):
    ticker: str
    company: str | None = None
    sector: str | None = None
    industry: str | None = None
    status: str = Field(..., description="compliant | mixed | non_compliant | unknown")
    status_ar: str
    business_activity_pass: bool | None = None
    business_flags: list[str] = Field(default_factory=list)
    market_cap: float | None = None
    debt_ratio_pct: float | None = Field(None, description="Interest-bearing debt / market cap, %")
    debt_pass: bool | None = None
    cash_ratio_pct: float | None = Field(None, description="(Cash + ST investments) / market cap, %")
    cash_pass: bool | None = None
    receivables_ratio_pct: float | None = Field(None, description="Receivables / market cap, %")
    receivables_pass: bool | None = None
    non_compliant_income_pct: float | None = Field(None, description="null = fundamentals feed pending (beta)")
    purification_rate_pct: float | None = Field(None, description="نسبة التطهير — null until income feed is wired")
    threshold_pct: float = AAOIFI_THRESHOLD
    zoya_status: str | None = Field(None, description="Raw Zoya basic-compliance status (COMPLIANT / NON_COMPLIANT / QUESTIONABLE)")
    zoya_report_date: str | None = Field(None, description="Date of the Zoya compliance report")
    beta: bool = True
    data_source: str = "Yahoo Finance balance sheet (annual) — income-segmentation feed pending"
    notes_ar: str


def _yf_shariah_screen(ticker: str) -> ShariahScreenResult:
    stock = yf.Ticker(ticker.upper())
    info: dict[str, Any] = {}
    try:
        info = stock.info or {}
    except Exception as exc:
        logger.warning("Shariah: info fetch failed for %s: %s", ticker, exc)

    sector = info.get("sector")
    industry = info.get("industry")
    company = info.get("shortName") or info.get("longName")
    mcap = info.get("marketCap")

    haystack = f"{sector or ''} {industry or ''}".lower()
    flags = [ar for kw, ar in _NON_COMPLIANT_INDUSTRIES.items() if kw in haystack]
    business_pass: bool | None = (len(flags) == 0) if (sector or industry) else None

    debt = cash = receivables = None
    try:
        bs = stock.balance_sheet
        if bs is not None and not bs.empty:
            latest = bs.iloc[:, 0]

            def row(*names: str) -> float | None:
                for n in names:
                    if n in latest.index:
                        v = latest[n]
                        if v == v:  # not NaN
                            return float(v)
                return None

            debt = row("Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt")
            cash = row(
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Cash Equivalents",
                "Cash Financial",
            )
            receivables = row("Receivables", "Accounts Receivable", "Gross Accounts Receivable")
    except Exception as exc:
        logger.warning("Shariah: balance sheet fetch failed for %s: %s", ticker, exc)

    def ratio(x: float | None) -> float | None:
        if x is None or not mcap:
            return None
        return round(x / mcap * 100, 2)

    debt_r, cash_r, recv_r = ratio(debt), ratio(cash), ratio(receivables)
    debt_ok = (debt_r <= AAOIFI_THRESHOLD) if debt_r is not None else None
    cash_ok = (cash_r <= AAOIFI_THRESHOLD) if cash_r is not None else None
    recv_ok = (recv_r <= AAOIFI_THRESHOLD) if recv_r is not None else None

    ratio_checks = [c for c in (debt_ok, cash_ok, recv_ok) if c is not None]

    if business_pass is False:
        status, status_ar = "non_compliant", "غير متوافق"
        note = "النشاط الأساسي للشركة ضمن الأنشطة غير المتوافقة مع الشريعة."
    elif business_pass is None and not ratio_checks:
        status, status_ar = "unknown", "غير محدد"
        note = "بيانات القوائم المالية غير متوفرة لهذا الرمز حاليًا."
    elif ratio_checks and not all(ratio_checks):
        status, status_ar = "non_compliant", "غير متوافق"
        note = "إحدى النسب المالية تتجاوز حد 30% وفق منهجية AAOIFI."
    elif ratio_checks and all(ratio_checks) and business_pass:
        status, status_ar = "compliant", "حلال (مبدئي)"
        note = "النشاط والنسب المالية ضمن حدود AAOIFI. فحص الدخل غير المتوافق قيد الربط — النتيجة مبدئية."
    else:
        status, status_ar = "mixed", "مختلط / قيد التحقق"
        note = "بعض البيانات غير مكتملة — التصنيف النهائي بانتظار اكتمال مصدر البيانات الأساسية."

    return ShariahScreenResult(
        ticker=ticker.upper(),
        company=company,
        sector=sector,
        industry=industry,
        status=status,
        status_ar=status_ar,
        business_activity_pass=business_pass,
        business_flags=flags,
        market_cap=float(mcap) if mcap else None,
        debt_ratio_pct=debt_r,
        debt_pass=debt_ok,
        cash_ratio_pct=cash_r,
        cash_pass=cash_ok,
        receivables_ratio_pct=recv_r,
        receivables_pass=recv_ok,
        non_compliant_income_pct=None,
        purification_rate_pct=None,
        notes_ar=note + " (تجريبي / قيد المعايرة)",
    )


# Zoya — professional Shariah screening provider (basic compliance API).
# Primary source for the verdict; the AAOIFI ratio detail from Yahoo stays
# as supporting evidence. Falls back to the ratio screen if Zoya is
# unavailable or doesn't cover the symbol.
_zoya_cache: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=512, ttl=3600)
_ZOYA_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,12}$")


async def _zoya_basic_compliance(ticker: str) -> dict[str, Any] | None:
    if not ZOYA_API_KEY or not _ZOYA_TICKER_RE.match(ticker):
        return None
    if ticker in _zoya_cache:
        return _zoya_cache[ticker]

    payload = {
        "query": (
            "query getReport { basicCompliance { report(symbol: \"%s\") "
            "{ exchange name reportDate status symbol } } }" % ticker
        )
    }
    # Zoya's GraphQL gateway accepts the API key via Authorization; try the
    # common header conventions so a key format change doesn't break us.
    header_variants = (
        {"Authorization": ZOYA_API_KEY},
        {"x-api-key": ZOYA_API_KEY},
        {"Authorization": f"Bearer {ZOYA_API_KEY}"},
    )
    for headers in header_variants:
        try:
            async with httpx.AsyncClient(timeout=12) as client:
                resp = await client.post(
                    ZOYA_GRAPHQL_URL,
                    json=payload,
                    headers={**headers, "Content-Type": "application/json"},
                )
            if resp.status_code in (401, 403):
                continue
            resp.raise_for_status()
            data = resp.json()

            errors = data.get("errors") or []
            if errors:
                if any("unauthorized" in str(e).lower() for e in errors):
                    continue  # wrong header convention — try the next one
                logger.warning("Zoya errors for %s: %s", ticker, errors)
                return None

            report = ((data.get("data") or {}).get("basicCompliance") or {}).get("report")
            if report and report.get("status"):
                _zoya_cache[ticker] = report
                return report
            return None
        except Exception as exc:
            logger.warning("Zoya request failed for %s: %s", ticker, exc)
            return None
    logger.warning("Zoya auth failed for all header conventions")
    return None


_ZOYA_STATUS_MAP = {
    "COMPLIANT": ("compliant", "حلال"),
    "NON_COMPLIANT": ("non_compliant", "غير متوافق"),
    "QUESTIONABLE": ("mixed", "مختلط"),
}


@app.get("/shariah", response_model=ShariahScreenResult)
async def shariah_screen(ticker: str = Query(..., min_length=1, max_length=12)):
    t = ticker.upper().strip()
    zoya, result = await asyncio.gather(
        _zoya_basic_compliance(t),
        asyncio.to_thread(_yf_shariah_screen, t),
    )
    if zoya:
        status_key = str(zoya.get("status", "")).upper()
        mapped = _ZOYA_STATUS_MAP.get(status_key)
        if mapped:
            result.status, result.status_ar = mapped
        result.zoya_status = status_key
        result.zoya_report_date = (zoya.get("reportDate") or "")[:10] or None
        result.company = result.company or zoya.get("name")
        result.beta = False  # verdict comes from a professional screening provider
        result.data_source = (
            f"Zoya basic compliance (report {result.zoya_report_date or 'n/a'}) "
            "· ratio detail: Yahoo Finance balance sheet"
        )
        result.notes_ar = (
            "الحكم الشرعي من Zoya (مزوّد فحص شرعي متخصص). "
            "النسب المعروضة من بيانات Yahoo للاطلاع والمقارنة."
        )
    return result


# ─────────────────────────────────────────────────────────────────────────
#  4. Halal portfolio optimizer — Markowitz frontier, no-short (BETA basket)
# ─────────────────────────────────────────────────────────────────────────

# Demo basket: US large-caps commonly held by Shariah-screened ETFs
# (SPUS / HLAL). TODO(data): replace with the live /shariah screen output
# over a Tadawul + US universe once the fundamentals feed is wired.
HALAL_DEMO_BASKET: list[tuple[str, str]] = [
    ("AAPL", "Apple"),
    ("MSFT", "Microsoft"),
    ("NVDA", "NVIDIA"),
    ("GOOGL", "Alphabet"),
    ("TSLA", "Tesla"),
    ("XOM", "Exxon Mobil"),
    ("PG", "Procter & Gamble"),
    ("JNJ", "Johnson & Johnson"),
]


class FrontierPoint(BaseModel):
    risk: float
    ret: float


class PortfolioAllocation(BaseModel):
    weights: dict[str, float]
    expected_return: float
    risk: float
    sharpe: float


class OptimizeResult(BaseModel):
    basket: list[str]
    basket_names: dict[str, str] = Field(..., description="Ticker → company name")
    risk_profile: str
    frontier: list[FrontierPoint]
    cloud: list[FrontierPoint] = Field(default_factory=list, description="Random feasible portfolios (visual)")
    optimal: PortfolioAllocation
    min_vol: PortfolioAllocation
    max_sharpe: PortfolioAllocation
    n_samples: int
    beta: bool = True
    notes_ar: str = (
        "سلة تجريبية من أسهم أمريكية مدرجة في صناديق مؤشرات شرعية (SPUS/HLAL). "
        "التحسين حقيقي (ماركوويتز بدون بيع مكشوف) — ربط السلة بفلتر الشريعة الحي قيد التنفيذ. (تجريبي)"
    )


def _portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple[float, float, float]:
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def _alloc(w: np.ndarray, tickers: list[str], mu: np.ndarray, cov: np.ndarray) -> PortfolioAllocation:
    ret, vol, sh = _portfolio_stats(w, mu, cov)
    return PortfolioAllocation(
        weights={t: round(float(x), 4) for t, x in zip(tickers, w) if x >= 0.005},
        expected_return=round(ret, 4),
        risk=round(vol, 4),
        sharpe=round(sh, 3),
    )


@app.get("/optimize", response_model=OptimizeResult)
async def optimize_portfolio(
    risk_profile: str = Query("balanced", description="conservative | balanced | aggressive"),
    n_samples: int = Query(12_000, ge=2000, le=30_000),
):
    profile = risk_profile.lower()
    if profile not in RISK_PROFILES:
        raise HTTPException(status_code=422, detail="risk_profile must be conservative | balanced | aggressive")

    all_tickers = [t for t, _ in HALAL_DEMO_BASKET]
    results = await asyncio.gather(
        *[fetch_daily_ohlcv(t) for t in all_tickers], return_exceptions=True
    )
    tickers, histories = [], []
    for t, res in zip(all_tickers, results):
        if isinstance(res, BaseException):
            logger.warning("Optimizer: dropping %s (fetch failed: %s)", t, res)
            continue
        tickers.append(t)
        histories.append(res)
    if len(tickers) < 4:
        raise HTTPException(status_code=502, detail="Could not fetch enough basket history to optimize.")

    # align on common dates
    per_ticker: list[dict[str, float]] = [
        {r["date"]: r["close"] for r in rows} for rows in histories
    ]
    common_dates = sorted(set.intersection(*[set(d.keys()) for d in per_ticker]))
    if len(common_dates) < 60:
        raise HTTPException(status_code=502, detail="Not enough overlapping history for the basket.")

    price_matrix = np.array(
        [[per_ticker[i][d] for d in common_dates] for i in range(len(tickers))], dtype=float
    )
    rets = np.diff(np.log(price_matrix), axis=1)
    mu = rets.mean(axis=1) * TRADING_DAYS
    cov = np.cov(rets) * TRADING_DAYS

    def _sample() -> dict[str, Any]:
        rng = np.random.default_rng(42)  # deterministic frontier per basket
        W = rng.dirichlet(np.ones(len(tickers)), size=n_samples)
        p_rets = W @ mu
        p_vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
        sharpes = (p_rets - RISK_FREE_RATE) / p_vols

        # frontier: max return per volatility bucket
        order = np.argsort(p_vols)
        buckets = np.array_split(order, 40)
        frontier_idx, best = [], -np.inf
        for b in buckets:
            if b.size == 0:
                continue
            top = b[np.argmax(p_rets[b])]
            if p_rets[top] > best:  # enforce monotone upper hull
                best = p_rets[top]
                frontier_idx.append(int(top))

        i_minvol = int(np.argmin(p_vols))
        i_sharpe = int(np.argmax(sharpes))
        # aggressive: highest-return frontier point
        i_aggr = frontier_idx[int(np.argmax([p_rets[i] for i in frontier_idx]))]

        cloud_idx = np.linspace(0, n_samples - 1, 350).astype(int)
        return {
            "W": W, "p_rets": p_rets, "p_vols": p_vols,
            "frontier_idx": frontier_idx, "i_minvol": i_minvol,
            "i_sharpe": i_sharpe, "i_aggr": i_aggr, "cloud_idx": cloud_idx,
        }

    s = await asyncio.to_thread(_sample)

    pick = {"conservative": s["i_minvol"], "balanced": s["i_sharpe"], "aggressive": s["i_aggr"]}[profile]

    return OptimizeResult(
        basket=tickers,
        basket_names=dict(HALAL_DEMO_BASKET),
        risk_profile=profile,
        frontier=[
            FrontierPoint(risk=round(float(s["p_vols"][i]), 4), ret=round(float(s["p_rets"][i]), 4))
            for i in s["frontier_idx"]
        ],
        cloud=[
            FrontierPoint(risk=round(float(s["p_vols"][i]), 4), ret=round(float(s["p_rets"][i]), 4))
            for i in s["cloud_idx"]
        ],
        optimal=_alloc(s["W"][pick], tickers, mu, cov),
        min_vol=_alloc(s["W"][s["i_minvol"]], tickers, mu, cov),
        max_sharpe=_alloc(s["W"][s["i_sharpe"]], tickers, mu, cov),
        n_samples=n_samples,
    )


# ─────────────────────────────────────────────────────────────────────────
#  5. Walk-forward backtest — technical pillar vs buy & hold
# ─────────────────────────────────────────────────────────────────────────

class EquityPoint(BaseModel):
    date: str
    strategy: float
    buyhold: float


class BacktestResult(BaseModel):
    ticker: str
    period: str
    n_days: int
    signal_threshold: float = Field(..., description="Long when walk-forward technical score ≥ threshold")
    hit_rate_pct: float | None = Field(None, description="% of long signals with positive 5-day forward return")
    n_signals: int
    brier_score: float | None = Field(None, description="Mean (score/100 − outcome)² on 5-day forward direction")
    sharpe_strategy: float | None = None
    sharpe_buyhold: float | None = None
    max_dd_strategy_pct: float | None = None
    max_dd_buyhold_pct: float | None = None
    total_return_strategy_pct: float | None = None
    total_return_buyhold_pct: float | None = None
    exposure_pct: float | None = Field(None, description="% of days the strategy was in the market")
    equity_curve: list[EquityPoint]
    scope: str = Field(
        "technical_pillar_only",
        description="Walk-forward uses the technical pillar (RSI/MACD/slope/Z) — historical sentiment & analyst feeds pending",
    )
    validation_target: bool = Field(True, description="Full 4-pillar historical validation is a target, not yet achieved")
    notes_ar: str = (
        "اختبار زمني حقيقي للركيزة الفنية من الإشارة المركّبة (RSI/MACD/الميل/Z-Score) على بيانات سنتين فعلية. "
        "أرشفة ركائز المشاعر والمحللين تاريخيًا قيد البناء — التحقق الكامل من الإشارة المركّبة «هدف التحقق»."
    )


def _walkforward_backtest(daily_rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    ordered = list(reversed(daily_rows))  # oldest first
    closes = np.array([r["close"] for r in ordered], dtype=float)
    dates = [r["date"] for r in ordered]
    n = closes.size

    # causal RSI-14 (simple rolling mean of gains/losses, mirrors /analyze)
    delta = np.diff(closes, prepend=closes[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    kern = np.ones(14) / 14
    avg_g = np.convolve(gains, kern, mode="full")[: n]
    avg_l = np.convolve(losses, kern, mode="full")[: n]
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_l > 0, avg_g / avg_l, np.inf)
    rsi = np.where(np.isinf(rs), 100.0, 100 - 100 / (1 + rs))

    # causal MACD histogram
    def ema(x: np.ndarray, span: int) -> np.ndarray:
        k = 2 / (span + 1)
        out = np.empty_like(x)
        out[0] = x[0]
        for i in range(1, x.size):
            out[i] = x[i] * k + out[i - 1] * (1 - k)
        return out

    macd_line = ema(closes, 12) - ema(closes, 26)
    macd_hist = macd_line - ema(macd_line, 9)

    # causal trailing 10-day regression slope (% per day)
    x10 = np.arange(10, dtype=float)
    slope_w = (x10 - x10.mean()) / ((x10 - x10.mean()) ** 2).sum()
    slopes = np.full(n, 0.0)
    conv = np.convolve(closes, slope_w[::-1], mode="valid")  # slope at each trailing window end
    slopes[9:] = conv
    slope_pct = np.divide(slopes, closes, out=np.zeros_like(slopes), where=closes > 0) * 100

    # causal trailing 20-day z-score
    z = np.zeros(n)
    csum = np.cumsum(np.insert(closes, 0, 0))
    csum2 = np.cumsum(np.insert(closes**2, 0, 0))
    for i in range(19, n):
        w = 20
        m = (csum[i + 1] - csum[i + 1 - w]) / w
        var = (csum2[i + 1] - csum2[i + 1 - w]) / w - m * m
        sd = math.sqrt(max(var, 0.0))
        z[i] = (closes[i] - m) / sd if sd > 0 else 0.0

    # technical pillar score (same mapping as calculate_inference_weights)
    rsi_pts = np.zeros(n)
    rsi_pts[rsi < 30] = 1
    rsi_pts[rsi < 20] = 2
    rsi_pts[rsi > 70] = -1
    rsi_pts[rsi > 80] = -2
    macd_pts = np.where(macd_hist > 0, 1.0, -1.0)
    gauge = np.clip(((rsi_pts + macd_pts + 3) / 6) * 100, 0, 100)
    slope_score = np.clip(50 + slope_pct * 25, 0, 100)
    z_score_comp = np.clip(50 - z * 20, 0, 100)
    tech = gauge * 0.40 + slope_score * 0.35 + z_score_comp * 0.25

    warmup = 40
    daily_ret = np.diff(closes) / closes[:-1]  # ret[t] = day t+1 return
    pos = (tech >= threshold).astype(float)

    strat_ret = pos[warmup:-1] * daily_ret[warmup:]
    bh_ret = daily_ret[warmup:]

    eq_strat = np.cumprod(1 + strat_ret)
    eq_bh = np.cumprod(1 + bh_ret)

    def sharpe(r: np.ndarray) -> float | None:
        if r.size < 20 or r.std(ddof=1) == 0:
            return None
        return round(float(r.mean() / r.std(ddof=1) * math.sqrt(TRADING_DAYS)), 2)

    def max_dd(eq: np.ndarray) -> float | None:
        if eq.size == 0:
            return None
        peak = np.maximum.accumulate(eq)
        return round(float(((eq - peak) / peak).min()) * 100, 2)

    # hit rate + Brier on 5-day forward direction
    idx = np.arange(warmup, n - 5)
    fwd_up = (closes[idx + 5] > closes[idx]).astype(float)
    probs = np.clip(tech[idx] / 100, 0.01, 0.99)
    brier = round(float(np.mean((probs - fwd_up) ** 2)), 4) if idx.size else None
    long_mask = tech[idx] >= threshold
    n_signals = int(long_mask.sum())
    hit = round(float(fwd_up[long_mask].mean()) * 100, 1) if n_signals > 0 else None

    # thinned equity curve
    eq_dates = dates[warmup + 1:]
    step = max(1, len(eq_strat) // 200)
    curve = [
        EquityPoint(date=eq_dates[i], strategy=round(float(eq_strat[i]), 4), buyhold=round(float(eq_bh[i]), 4))
        for i in range(0, len(eq_strat), step)
    ]

    return {
        "n_days": int(n),
        "hit_rate_pct": hit,
        "n_signals": n_signals,
        "brier_score": brier,
        "sharpe_strategy": sharpe(strat_ret),
        "sharpe_buyhold": sharpe(bh_ret),
        "max_dd_strategy_pct": max_dd(eq_strat),
        "max_dd_buyhold_pct": max_dd(eq_bh),
        "total_return_strategy_pct": round(float(eq_strat[-1] - 1) * 100, 2) if eq_strat.size else None,
        "total_return_buyhold_pct": round(float(eq_bh[-1] - 1) * 100, 2) if eq_bh.size else None,
        "exposure_pct": round(float(pos[warmup:].mean()) * 100, 1),
        "equity_curve": curve,
    }


@app.get("/backtest", response_model=BacktestResult)
async def backtest(
    ticker: str = Query(..., min_length=1, max_length=12),
    threshold: float = Query(55.0, ge=40, le=80),
):
    ticker = ticker.upper().strip()
    daily_rows = await fetch_daily_ohlcv(ticker, period="2y")
    if len(daily_rows) < 120:
        raise HTTPException(status_code=422, detail=f"Not enough history for {ticker} to backtest.")

    result = await asyncio.to_thread(_walkforward_backtest, daily_rows, threshold)
    return BacktestResult(ticker=ticker, period="2y", signal_threshold=threshold, **result)


# ─────────────────────────────────────────────────────────────────────────
#  6. Arabic agentic copilot — GPT-4o tool-calling over the live endpoints
# ─────────────────────────────────────────────────────────────────────────

ARABIC_TICKER_MAP: dict[str, str] = {
    "أرامكو": "2222.SR", "ارامكو": "2222.SR",
    "الراجحي": "1120.SR", "مصرف الراجحي": "1120.SR",
    "سابك": "2010.SR",
    "اس تي سي": "7010.SR", "إس تي سي": "7010.SR", "الاتصالات السعودية": "7010.SR",
    "أكوا باور": "2082.SR", "اكوا باور": "2082.SR",
    "معادن": "1211.SR",
    "أبل": "AAPL", "آبل": "AAPL", "ابل": "AAPL",
    "مايكروسوفت": "MSFT",
    "تسلا": "TSLA",
    "إنفيديا": "NVDA", "انفيديا": "NVDA", "نفيديا": "NVDA",
    "أمازون": "AMZN", "امازون": "AMZN",
    "جوجل": "GOOGL", "قوقل": "GOOGL", "ألفابت": "GOOGL",
    "ميتا": "META", "فيسبوك": "META",
    "لوسيد": "LCID",
}

COPILOT_SYSTEM_PROMPT = (
    "أنت «مُرشِد» — المحلل الكمّي الذكي لمنصة MSA، تخاطب مستثمرًا عربيًا.\n\n"
    "قواعد صارمة (عدم الالتزام بها يُعد فشلًا):\n"
    "1. لا تذكر أي رقم أو نسبة أو سعر إلا إذا ورد حرفيًا في نتائج الأدوات في هذه المحادثة. "
    "ممنوع منعًا باتًا اختلاق أو تقدير أو «تذكّر» أي أرقام من معرفتك السابقة.\n"
    "2. إذا فشلت أداة أو أعادت خطأ: قل ذلك صراحة («تعذّر جلب البيانات لهذا الجزء») ولا تحاول تعويض النقص بأرقام من عندك.\n"
    "3. اذكر مصدر كل معلومة بين قوسين: (Yahoo Finance)، (FinBERT)، (CNN Fear & Greed)، (محاكاة GBM)، (فلتر AAOIFI — تجريبي).\n"
    "4. أجب بالعربية الفصحى الواضحة. اترك رموز الأسهم والمصطلحات التقنية (RSI, MACD, VaR) بالإنجليزية.\n"
    "5. حوّل أسماء الشركات العربية إلى رموز باستخدام هذه الخريطة قبل استدعاء الأدوات: "
    + json.dumps(ARABIC_TICKER_MAP, ensure_ascii=False)
    + "\n6. الأسهم السعودية تُدعم عبر لاحقة .SR (بيانات Yahoo Finance، قد تكون تغطية الأخبار والمحللين محدودة — قل ذلك عند حدوثه).\n"
    "7. نتيجة الفحص الشرعي تجريبية (قيد المعايرة) — قلها دائمًا عند عرضها.\n"
    "8. اختم دائمًا بسطر: «هذا تحليل كمّي وليس توصية استثمارية.»\n"
    "9. كن موجزًا ومنظمًا: حكم واضح أولًا، ثم الأرقام الداعمة في نقاط قصيرة."
)

CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_stock",
            "description": "التحليل الكامل لسهم: السعر، الدرجة المركّبة (0-100)، الركائز الأربع، RSI/MACD/Z-Score، مشاعر الأخبار FinBERT، إجماع المحللين، حكم GPT-4o.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "رمز السهم، مثل AAPL أو 2222.SR"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "monte_carlo",
            "description": "محاكاة مونت كارلو (GBM، 10,000 مسار) لتوزيع السعر المستقبلي: VaR، CVaR، واحتمال بلوغ سعر مستهدف.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "horizon_days": {"type": "integer", "description": "أيام التداول، افتراضي 63"},
                    "target": {"type": "number", "description": "السعر المستهدف (اختياري)"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shariah_screen",
            "description": "فحص شرعي تجريبي وفق نسب AAOIFI (الدين/القيمة السوقية، النقد، الذمم) وفلتر النشاط. النتيجة: حلال / مختلط / غير متوافق.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "position_size",
            "description": "حجم المركز الموصى به (% من رأس المال) عبر استهداف التقلب و¼ كيلي، مع مستوى وقف الخسارة (2×ATR).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "profile": {"type": "string", "enum": ["conservative", "balanced", "aggressive"]},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "backtest_signal",
            "description": "اختبار زمني (سنتان) للركيزة الفنية من الإشارة مقابل الشراء والاحتفاظ: نسبة الإصابة، Brier، Sharpe، أقصى تراجع.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
]


def _compact_analysis(a: AnalysisResponse) -> dict[str, Any]:
    """Trim the /analyze payload to the numbers the copilot may quote."""
    iw, t, zs, pt = a.inference_weights, a.technicals, a.zscore_analysis, a.price_target
    return {
        "ticker": a.ticker,
        "currency": _currency_for(a.ticker),
        "price": pt.current if pt else None,
        "daily_change_pct": pt.daily_change_pct if pt else None,
        "composite_score_0_100": iw.composite_score if iw else None,
        "composite_signal": iw.composite_signal if iw else None,
        "pillars": {
            "technical_35pct": iw.technical_score if iw else None,
            "sentiment_25pct": iw.sentiment_score if iw else None,
            "analyst_20pct": iw.analyst_score if iw else None,
            "volume_20pct": iw.volume_score if iw else None,
        } if iw else None,
        "rsi_14": t.rsi_14 if t else None,
        "rsi_signal": t.rsi_signal if t else None,
        "macd_trend": t.macd_trend if t else None,
        "sma_signal": a.moving_averages.signal,
        "zscore_20d": zs.zscore if zs else None,
        "zscore_signal": zs.signal if zs else None,
        "reversal_probability_pct": zs.reversal_probability if zs else None,
        "bollinger_position": a.bollinger_bands.position if a.bollinger_bands else None,
        "obv_divergence": a.obv_analysis.divergence if a.obv_analysis else None,
        "news_sentiment": a.news_sentiment.label if a.news_sentiment else None,
        "news_pos_neg_neu": [a.news_sentiment.positive, a.news_sentiment.negative, a.news_sentiment.neutral] if a.news_sentiment else None,
        "fear_greed": {"value": a.fear_greed.value, "label": a.fear_greed.label},
        "analyst_recommendation": a.analyst_ratings.recommendation if a.analyst_ratings else None,
        "analyst_target_mean": pt.target_mean if pt else None,
        "upside_to_target_pct": pt.upside_pct if pt else None,
        "gpt_verdict": a.gpt_analysis.actionable_insight if a.gpt_analysis else None,
        "gpt_confidence": a.gpt_analysis.confidence_score if a.gpt_analysis else None,
        "sources": ["Yahoo Finance (yfinance)", "ProsusAI/FinBERT", "CNN Fear & Greed", "GPT-4o"],
    }


async def _execute_copilot_tool(name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Run a copilot tool against the real endpoints. Returns (json_str, ok)."""
    try:
        ticker = str(args.get("ticker", "")).upper().strip()
        if name == "analyze_stock":
            res = await analyze(ticker=ticker)
            return json.dumps(_compact_analysis(res), ensure_ascii=False), True
        if name == "monte_carlo":
            res = await monte_carlo(
                ticker=ticker,
                horizon_days=int(args.get("horizon_days") or 63),
                n_paths=10_000,
                target=args.get("target"),
                seed=None,
            )
            d = res.model_dump()
            for k in ("p05", "p25", "p50", "p75", "p95", "sample_paths"):
                d.pop(k, None)  # copilot needs the risk numbers, not the curves
            return json.dumps(d, ensure_ascii=False), True
        if name == "shariah_screen":
            res = await shariah_screen(ticker=ticker)
            return json.dumps(res.model_dump(), ensure_ascii=False), True
        if name == "position_size":
            res = await position_size(
                ticker=ticker,
                profile=str(args.get("profile") or "balanced"),
                score=None,
                capital=None,
            )
            return json.dumps(res.model_dump(), ensure_ascii=False), True
        if name == "backtest_signal":
            res = await backtest(ticker=ticker, threshold=55.0)
            d = res.model_dump()
            d.pop("equity_curve", None)
            return json.dumps(d, ensure_ascii=False), True
        return json.dumps({"error": f"unknown tool {name}"}), False
    except HTTPException as exc:
        return json.dumps({"error": f"{exc.status_code}: {exc.detail}"}, ensure_ascii=False), False
    except Exception as exc:
        logger.warning("Copilot tool %s failed: %s", name, exc)
        return json.dumps({"error": str(exc)}, ensure_ascii=False), False


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, description="Previous turns (last 8 kept)")


class ChatResponse(BaseModel):
    reply: str
    tools_used: list[str] = Field(default_factory=list)
    tool_failures: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


@app.post("/chat", response_model=ChatResponse)
async def copilot_chat(req: ChatRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured.")

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    messages: list[dict[str, Any]] = [{"role": "system", "content": COPILOT_SYSTEM_PROMPT}]
    for m in req.history[-8:]:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": req.message})

    tools_used: list[str] = []
    tool_failures: list[str] = []

    for _ in range(5):  # bounded agent loop
        completion = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=CHAT_TOOLS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=900,
        )
        msg = completion.choices[0].message

        if not msg.tool_calls:
            reply = (msg.content or "").strip()
            if not reply:
                reply = "تعذّر توليد إجابة — حاول مرة أخرى."
            sources = sorted(
                {s for s in [
                    "Yahoo Finance" if any(t in tools_used for t in ("analyze_stock", "monte_carlo", "position_size", "backtest_signal", "shariah_screen")) else None,
                    "FinBERT" if "analyze_stock" in tools_used else None,
                    "CNN Fear & Greed" if "analyze_stock" in tools_used else None,
                    "GBM Monte Carlo (10,000 مسار)" if "monte_carlo" in tools_used else None,
                    "AAOIFI screen (تجريبي)" if "shariah_screen" in tools_used else None,
                    "GPT-4o",
                ] if s}
            )
            return ChatResponse(reply=reply, tools_used=tools_used, tool_failures=tool_failures, sources=sources)

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result, ok = await _execute_copilot_tool(tc.function.name, args)
            tools_used.append(tc.function.name)
            if not ok:
                tool_failures.append(tc.function.name)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return ChatResponse(
        reply="تجاوزت المحادثة الحد الأقصى لاستدعاءات الأدوات — جرّب سؤالًا أبسط.",
        tools_used=tools_used,
        tool_failures=tool_failures,
        sources=[],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
