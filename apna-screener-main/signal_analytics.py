"""Isolated historical signal analytics helpers for apna-screener.

This module deliberately does not change the existing scanner/scheduler logic.
Historical signals are reconstructed from Dhan OHLC candles using the scanner's
existing signal detector; they are not a record of alerts actually sent in the past.
"""
from datetime import date, timedelta
import time
import pandas as pd

ANALYTICS_TIMEFRAMES = ["1min", "5min", "15min", "30min", "45min", "1h", "2h", "3h", "4h"]
BASE_INTERVAL_FOR_ANALYTICS = {
    "1min": "1min", "5min": "5min", "15min": "15min",
    "30min": "15min", "45min": "15min",
    "1h": "1h", "2h": "1h", "3h": "1h", "4h": "1h",
}
FETCH_INTERVAL_VALUES = {"1min": 1, "5min": 5, "15min": 15, "1h": 60}


def _date_chunks(start_date, end_date, max_days=30):
    start = date.fromisoformat(str(start_date))
    end = date.fromisoformat(str(end_date))
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=max_days - 1), end)
        yield cursor.isoformat(), chunk_end.isoformat()
        cursor = chunk_end + timedelta(days=1)


def _normalise_response(data_list):
    if not data_list:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(data_list)
    if df.empty:
        return df
    if "timestamp" in df.columns:
        # Dhan commonly returns epoch seconds. Support strings too for safety.
        numeric = pd.to_numeric(df["timestamp"], errors="coerce")
        if numeric.notna().all():
            df["timestamp"] = pd.to_datetime(numeric, unit="s", errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
        else:
            parsed = pd.to_datetime(df["timestamp"], errors="coerce")
            if parsed.dt.tz is None:
                parsed = parsed.dt.tz_localize("Asia/Kolkata", ambiguous="NaT", nonexistent="NaT")
            else:
                parsed = parsed.dt.tz_convert("Asia/Kolkata")
            df["timestamp"] = parsed
    elif "time" in df.columns:
        numeric = pd.to_numeric(df["time"], errors="coerce")
        if numeric.notna().all():
            df["timestamp"] = pd.to_datetime(numeric, unit="s", errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
        else:
            parsed = pd.to_datetime(df["time"], errors="coerce")
            if parsed.dt.tz is None:
                parsed = parsed.dt.tz_localize("Asia/Kolkata", ambiguous="NaT", nonexistent="NaT")
            else:
                parsed = parsed.dt.tz_convert("Asia/Kolkata")
            df["timestamp"] = parsed
    else:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df.get(col, pd.NA), errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    return df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)


def _evaluate_signal(df, signal_idx, direction, entry, sl_points, target_points, fixed_points):
    """Evaluate only candles after the signal candle. Same-candle double hits count as loss."""
    bullish = direction == "BULLISH"
    target_price = entry + target_points if bullish else entry - target_points
    stop_price = entry - sl_points if bullish else entry + sl_points
    fixed_target = entry + fixed_points if bullish else entry - fixed_points
    fixed_stop = entry - fixed_points if bullish else entry + fixed_points

    sl_outcome = "PENDING"
    fixed_outcome = "PENDING"
    sl_exit_time = ""
    fixed_exit_time = ""
    for _, candle in df.iloc[signal_idx + 1:].iterrows():
        ts = candle["timestamp"]
        if sl_outcome == "PENDING":
            if bullish:
                target_hit, stop_hit = candle["high"] >= target_price, candle["low"] <= stop_price
            else:
                target_hit, stop_hit = candle["low"] <= target_price, candle["high"] >= stop_price
            if target_hit and stop_hit:
                sl_outcome, sl_exit_time = "LOSS", str(ts) + " (same-candle ambiguity; counted as loss)"
            elif target_hit:
                sl_outcome, sl_exit_time = "WIN", str(ts)
            elif stop_hit:
                sl_outcome, sl_exit_time = "LOSS", str(ts)
        if fixed_outcome == "PENDING":
            if bullish:
                favourable_hit, adverse_hit = candle["high"] >= fixed_target, candle["low"] <= fixed_stop
            else:
                favourable_hit, adverse_hit = candle["low"] <= fixed_target, candle["high"] >= fixed_stop
            if favourable_hit and adverse_hit:
                fixed_outcome, fixed_exit_time = "LOSS", str(ts) + " (same-candle ambiguity; counted as loss)"
            elif favourable_hit:
                fixed_outcome, fixed_exit_time = "WIN", str(ts)
            elif adverse_hit:
                fixed_outcome, fixed_exit_time = "LOSS", str(ts)
        if sl_outcome != "PENDING" and fixed_outcome != "PENDING":
            break
    return sl_outcome, sl_exit_time, fixed_outcome, fixed_exit_time, target_price, stop_price


def _summary_for(rows, timeframe, index_name):
    selected = [r for r in rows if r["timeframe"] == timeframe and (index_name == "ALL" or r["index"] == index_name)]
    def metric(field):
        wins = sum(r[field] == "WIN" for r in selected)
        losses = sum(r[field] == "LOSS" for r in selected)
        pending = sum(r[field] == "PENDING" for r in selected)
        decided = wins + losses
        return {"wins": wins, "losses": losses, "pending": pending,
                "accuracy": round(100 * wins / decided, 2) if decided else None,
                "decided": decided}
    return {"timeframe": timeframe, "signals": len(selected), "sl_target": metric("sl_outcome"), "fixed_points": metric("fixed_outcome")}


def run_historical_analysis(dhan, start_date, end_date, selected_index, fixed_points,
                            index_ids, timeframes_to_notify, timeframe_map, base_interval,
                            resample_rules, session_start, session_end, detector,
                            resampler, drop_incomplete):
    """Fetch historical candles in small date windows, reconstruct signals and evaluate outcomes."""
    errors = []
    candle_cache = {}
    selected_indices = index_ids.items() if selected_index == "ALL" else [(selected_index, index_ids[selected_index])]
    request_count = 0

    for index_name, security_id in selected_indices:
        for base_tf in sorted(set(BASE_INTERVAL_FOR_ANALYTICS.values())):
            frames = []
            interval_value = FETCH_INTERVAL_VALUES[base_tf]
            for chunk_start, chunk_end in _date_chunks(start_date, end_date, max_days=30):
                try:
                    response = dhan.intraday_minute_data(
                        security_id=security_id,
                        exchange_segment="IDX_I",
                        instrument_type="INDEX",
                        from_date=chunk_start,
                        to_date=chunk_end,
                        interval=interval_value,
                    )
                    request_count += 1
                    data = response
                    if isinstance(response, dict):
                        data = next((response.get(k) for k in ("data", "result", "candles", "items", "rows") if response.get(k)), None)
                    if isinstance(response, pd.DataFrame):
                        data = response.to_dict(orient="records")
                    frame = _normalise_response(data)
                    if not frame.empty:
                        frames.append(frame)
                    time.sleep(0.12)  # be gentle with API rate limits
                except Exception as exc:
                    errors.append(f"{index_name} {base_tf} {chunk_start}–{chunk_end}: {exc}")
            if frames:
                combined = pd.concat(frames, ignore_index=True).sort_values("timestamp")
                combined = combined.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
                start_ts = pd.Timestamp(start_date, tz="Asia/Kolkata")
                end_ts = pd.Timestamp(end_date, tz="Asia/Kolkata") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                combined = combined[(combined["timestamp"] >= start_ts) & (combined["timestamp"] <= end_ts)]
                candle_cache[(index_name, base_tf)] = combined
            else:
                candle_cache[(index_name, base_tf)] = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    rows = []
    for index_name, _ in selected_indices:
        for timeframe in ANALYTICS_TIMEFRAMES:
            base_tf = BASE_INTERVAL_FOR_ANALYTICS[timeframe]
            df = candle_cache.get((index_name, base_tf), pd.DataFrame()).copy()
            if df.empty:
                continue
            if timeframe in resample_rules:
                df = resampler(df, resample_rules[timeframe], offset_minutes=555)
            if df.empty:
                continue
            df = df[(df["timestamp"].dt.time >= session_start) & (df["timestamp"].dt.time <= session_end)].sort_values("timestamp").reset_index(drop=True)
            df = drop_incomplete(df, timeframe)
            if df.empty:
                continue
            bullish_signals, bearish_signals = detector(df, timeframe, index_name)
            signal_items = [(s, "BULLISH") for s in bullish_signals] + [(s, "BEARISH") for s in bearish_signals]
            for sig, direction in signal_items:
                ts = pd.Timestamp(sig["time"])
                matches = df.index[df["timestamp"] == ts].tolist()
                if not matches:
                    continue
                signal_idx = matches[0]
                entry = float(df.iloc[signal_idx]["close"])
                sl_points = abs(float(sig.get("stoploss", 0)))
                target_points = abs(float(sig.get("target", 0)))
                sl_outcome, sl_exit_time, fixed_outcome, fixed_exit_time, target_price, stop_price = _evaluate_signal(
                    df, signal_idx, direction, entry, sl_points, target_points, float(fixed_points)
                )
                rows.append({
                    "time": str(ts), "index": index_name, "timeframe": timeframe,
                    "direction": direction, "signal_type": sig.get("type", ""),
                    "entry": round(entry, 2), "stoploss_points": round(sl_points, 2),
                    "target_points": round(target_points, 2),
                    "stop_price": round(stop_price, 2), "target_price": round(target_price, 2),
                    "sl_outcome": sl_outcome, "sl_exit_time": sl_exit_time,
                    "fixed_outcome": fixed_outcome, "fixed_exit_time": fixed_exit_time,
                })
    rows.sort(key=lambda r: r["time"], reverse=True)
    summaries = [_summary_for(rows, tf, selected_index) for tf in ANALYTICS_TIMEFRAMES]
    return {"rows": rows, "summaries": summaries, "errors": errors, "request_count": request_count}
