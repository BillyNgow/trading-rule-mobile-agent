import os
import sys
import time
import requests
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import screen


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_row(**kwargs):
    """Return a score_setup-compatible Series with sensible A-grade defaults."""
    defaults = {
        "Close":       100.0,
        "SMA20":        98.0,
        "SMA50":        90.0,
        "Volume":    1_500_000,
        "AvgVolume20": 1_000_000,
        "High20":      145.0,
        "Low20":        80.0,
        "RSI14":        55.0,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)
    # Defaults produce: Trend Clean +20, S/R Clear +20, Volume Confirm +15,
    # Chasing Low +20, R:R Pass +15, News +5 → Score 95, Grade A


def make_ohlcv_df(n=200, base_price=100.0):
    """Synthetic OHLCV DataFrame suitable for add_indicators."""
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = pd.Series(base_price + np.arange(n) * 0.1, index=idx)
    return pd.DataFrame({
        "Open":   close * 0.99,
        "High":   close * 1.02,
        "Low":    close * 0.98,
        "Close":  close,
        "Volume": 1_000_000.0,
    }, index=idx)


def make_full_result_row(**overrides):
    """Return a dict with every column make_mobile_html / make_desktop_html expect."""
    row = {
        "Ticker": "AAPL", "Date": "2024-01-02", "Close": 151.0,
        "FinalGrade": "A", "Score": 90, "Status": "Review First",
        "CleanTrend": "Clean", "ClearSR": "Clear",
        "VolumeConfirmation": "Confirm", "NotChasing": "Low",
        "RR_1_to_2_Feasible": "Pass", "NewsRisk": "Manual",
        "VolumeRatio": 1.5, "RR_Estimate": 2.2,
        "DistanceFromSMA20Pct": 2.0, "Notes": "",
        "Volume": 1_500_000, "SMA20": 148.0, "SMA50": 140.0,
        "SMA200": 130.0, "RSI14": 55.0, "ATR14": 2.5,
    }
    row.update(overrides)
    return row


# ── safe_round ────────────────────────────────────────────────────────────────

class TestSafeRound:
    def test_rounds_to_two_decimal_places(self):
        assert screen.safe_round(3.14159) == 3.14

    def test_custom_digits(self):
        assert screen.safe_round(3.14159, 4) == 3.1416

    def test_nan_returns_empty_string(self):
        assert screen.safe_round(float("nan")) == ""

    def test_none_returns_empty_string(self):
        assert screen.safe_round(None) == ""

    def test_non_numeric_string_returns_empty_string(self):
        assert screen.safe_round("abc") == ""

    def test_integer_input(self):
        assert screen.safe_round(5) == 5.0

    def test_zero(self):
        assert screen.safe_round(0) == 0.0


# ── tradingview_url ───────────────────────────────────────────────────────────

class TestTradingviewUrl:
    def test_known_nasdaq_ticker_uses_nasdaq_exchange(self):
        assert "NASDAQ%3AAMZN" in screen.tradingview_url("AMZN")

    def test_unknown_ticker_defaults_to_nyse(self):
        assert "NYSE%3AJPM" in screen.tradingview_url("JPM")

    def test_dot_in_ticker_replaced_with_hyphen(self):
        url = screen.tradingview_url("BRK.B")
        symbol_part = url.split("symbol=")[-1]
        assert "BRK-B" in symbol_part
        assert "." not in symbol_part

    def test_all_nasdaq_set_members_use_nasdaq(self):
        for ticker in screen.NASDAQ_TICKERS:
            assert "NASDAQ" in screen.tradingview_url(ticker)


# ── icon ──────────────────────────────────────────────────────────────────────

class TestIcon:
    def test_value_in_good_returns_checkmark(self):
        assert screen.icon("Clean", ["Clean"]) == "✅"

    def test_value_in_warn_returns_warning(self):
        assert screen.icon("Mixed", ["Clean"], ["Mixed"]) == "⚠️"

    def test_value_in_neither_returns_cross(self):
        assert screen.icon("Weak", ["Clean"], ["Mixed"]) == "❌"

    def test_default_warn_none_does_not_crash(self):
        assert screen.icon("Weak", ["Clean"]) == "❌"

    def test_good_takes_priority_over_warn(self):
        assert screen.icon("X", ["X"], ["X"]) == "✅"


# ── score_setup ───────────────────────────────────────────────────────────────

class TestScoreSetupGrades:
    """Grade boundary tests — verify score arithmetic and grade assignment."""

    def test_grade_a_on_perfect_setup(self):
        # Trend Clean +20, S/R Clear +20, Volume Confirm +15,
        # Chasing Low +20, R:R Pass +15, News +5 → 95
        result = screen.score_setup(make_row())
        assert result["FinalGrade"] == "A"
        assert result["Score"] >= 85
        assert result["Status"] == "Review First"

    def test_grade_b_range(self):
        # Trend Clean +20, S/R Check +10 (pos=0.185 < 0.25),
        # Volume Neutral +8, Chasing Low +20, R:R Pass +15, News +5 → 78
        row = pd.Series({
            "Close": 92.0,  "SMA20": 91.0,  "SMA50": 85.0,
            "High20": 145.0, "Low20": 80.0,
            "Volume": 900_000, "AvgVolume20": 1_000_000,
            "RSI14": 55.0,
        })
        result = screen.score_setup(row)
        assert result["FinalGrade"] == "B"
        assert 70 <= result["Score"] <= 84

    def test_grade_c_range(self):
        # Trend Weak +0 (close < sma50), S/R Clear +20, Volume Neutral +8,
        # Chasing Low +20, R:R Pass +15, News +5 → 68
        row = make_row(SMA50=105.0, Volume=900_000)
        result = screen.score_setup(row)
        assert result["FinalGrade"] == "C"
        assert 50 <= result["Score"] <= 69

    def test_grade_d_on_worst_case_setup(self):
        # Trend Weak +0, S/R Weak +0 (NaN range), Volume Weak +0,
        # Chasing High +0 (dist=20%, rsi=75>70), R:R Fail +0, News +5 → 5
        row = pd.Series({
            "Close": 120.0, "SMA20": 100.0, "SMA50": 130.0,
            "High20": np.nan, "Low20": np.nan,
            "Volume": 100_000, "AvgVolume20": 1_000_000,
            "RSI14": 75.0,
        })
        result = screen.score_setup(row)
        assert result["FinalGrade"] == "D"
        assert result["Score"] < 50


class TestScoreSetupTrend:
    def test_clean_when_close_above_sma20_above_sma50(self):
        result = screen.score_setup(make_row(Close=100, SMA20=98, SMA50=90))
        assert result["CleanTrend"] == "Clean"

    def test_mixed_when_close_above_sma50_only(self):
        result = screen.score_setup(make_row(Close=100, SMA20=105, SMA50=90))
        assert result["CleanTrend"] == "Mixed"

    def test_weak_when_close_below_sma50(self):
        result = screen.score_setup(make_row(Close=85, SMA20=90, SMA50=95))
        assert result["CleanTrend"] == "Weak"

    def test_nan_smas_produce_trend_weak(self):
        result = screen.score_setup(make_row(SMA20=np.nan, SMA50=np.nan))
        assert result["CleanTrend"] == "Weak"


class TestScoreSetupSR:
    def test_clear_when_position_in_middle_range(self):
        # pos = (100-80)/(145-80) = 0.308 → in [0.25, 0.75]
        result = screen.score_setup(make_row(Close=100, High20=145, Low20=80))
        assert result["ClearSR"] == "Clear"

    def test_check_when_position_near_top_edge(self):
        # pos = (98-80)/(100-80) = 0.90 → > 0.75
        result = screen.score_setup(make_row(Close=98, High20=100, Low20=80))
        assert result["ClearSR"] == "Check"

    def test_weak_when_high20_equals_low20(self):
        result = screen.score_setup(make_row(High20=100, Low20=100))
        assert result["ClearSR"] == "Weak"

    def test_weak_when_high20_low20_are_nan(self):
        result = screen.score_setup(make_row(High20=np.nan, Low20=np.nan))
        assert result["ClearSR"] == "Weak"


class TestScoreSetupVolume:
    def test_confirm_when_ratio_at_or_above_1_2(self):
        result = screen.score_setup(make_row(Volume=1_300_000, AvgVolume20=1_000_000))
        assert result["VolumeConfirmation"] == "Confirm"
        assert result["VolumeRatio"] == 1.3

    def test_neutral_when_ratio_between_0_8_and_1_2(self):
        result = screen.score_setup(make_row(Volume=900_000, AvgVolume20=1_000_000))
        assert result["VolumeConfirmation"] == "Neutral"

    def test_weak_when_ratio_below_0_8(self):
        result = screen.score_setup(make_row(Volume=500_000, AvgVolume20=1_000_000))
        assert result["VolumeConfirmation"] == "Weak"

    def test_weak_when_avgvol_is_zero(self):
        result = screen.score_setup(make_row(AvgVolume20=0))
        assert result["VolumeConfirmation"] == "Weak"
        assert result["VolumeRatio"] == 0


class TestScoreSetupRR:
    def test_pass_when_rr_at_or_above_2(self):
        # risk=20, reward=45, rr=2.25
        result = screen.score_setup(make_row(Close=100, High20=145, Low20=80))
        assert result["RR_1_to_2_Feasible"] == "Pass"
        assert result["RR_Estimate"] >= 2.0

    def test_close_when_rr_between_1_5_and_2(self):
        # risk=20, reward=34, rr=1.7
        result = screen.score_setup(make_row(Close=100, High20=134, Low20=80))
        assert result["RR_1_to_2_Feasible"] == "Close"

    def test_fail_when_rr_below_1_5(self):
        # risk=10, reward=10, rr=1.0
        result = screen.score_setup(make_row(Close=100, High20=110, Low20=90))
        assert result["RR_1_to_2_Feasible"] == "Fail"

    def test_fail_when_close_equals_low20(self):
        result = screen.score_setup(make_row(Close=80, Low20=80, High20=145))
        assert result["RR_1_to_2_Feasible"] == "Fail"
        assert result["RR_Estimate"] == 0

    def test_news_risk_is_always_manual(self):
        assert screen.score_setup(make_row())["NewsRisk"] == "Manual"


class TestScoreSetupChasing:
    def test_low_when_dist_le_5_and_rsi_le_65(self):
        # dist=(100-98)/98*100=2.04%, rsi=55
        result = screen.score_setup(make_row(Close=100, SMA20=98, RSI14=55))
        assert result["NotChasing"] == "Low"

    def test_borderline_when_dist_le_8_and_rsi_le_70(self):
        # dist=(107-100)/100*100=7%, rsi=68
        result = screen.score_setup(make_row(Close=107, SMA20=100, RSI14=68))
        assert result["NotChasing"] == "Borderline"

    def test_high_when_dist_above_8(self):
        # dist=(115-100)/100*100=15%, rsi=72
        result = screen.score_setup(make_row(Close=115, SMA20=100, RSI14=72))
        assert result["NotChasing"] == "High"

    def test_high_when_sma20_is_zero(self):
        result = screen.score_setup(make_row(SMA20=0, RSI14=55))
        assert result["NotChasing"] == "High"
        assert result["DistanceFromSMA20Pct"] == 999

    def test_nan_rsi_prevents_chasing_low(self):
        # dist=2% (≤5) but rsi=NaN → pd.notna fails in every branch → always High
        result = screen.score_setup(make_row(Close=100, SMA20=98, RSI14=np.nan))
        assert result["NotChasing"] == "High"


class TestScoreSetupRobustness:
    def test_all_nan_values_does_not_crash(self):
        row = pd.Series({k: np.nan for k in
                         ["Close", "SMA20", "SMA50", "Volume", "AvgVolume20",
                          "High20", "Low20", "RSI14"]})
        result = screen.score_setup(row)
        assert result["FinalGrade"] in ("A", "B", "C", "D")

    def test_score_is_non_negative(self):
        row = pd.Series({k: np.nan for k in
                         ["Close", "SMA20", "SMA50", "Volume", "AvgVolume20",
                          "High20", "Low20", "RSI14"]})
        assert screen.score_setup(row)["Score"] >= 0

    def test_notes_field_is_string(self):
        assert isinstance(screen.score_setup(make_row())["Notes"], str)


# ── add_indicators ────────────────────────────────────────────────────────────

class TestAddIndicators:
    def test_expected_columns_are_added(self):
        df = screen.add_indicators(make_ohlcv_df())
        for col in ["SMA20", "SMA50", "SMA200", "AvgVolume20",
                    "High20", "Low20", "RSI14", "ATR14"]:
            assert col in df.columns

    def test_sma20_is_nan_for_first_19_rows(self):
        df = screen.add_indicators(make_ohlcv_df(n=200))
        assert df["SMA20"].iloc[:19].isna().all()

    def test_sma20_value_correct_at_row_19(self):
        df = make_ohlcv_df(n=200)
        result = screen.add_indicators(df)
        expected = df["Close"].iloc[:20].mean()
        assert abs(result["SMA20"].iloc[19] - expected) < 1e-9

    def test_sma200_is_nan_for_first_199_rows(self):
        df = screen.add_indicators(make_ohlcv_df(n=200))
        assert df["SMA200"].iloc[:199].isna().all()

    def test_rsi_is_100_for_monotonically_rising_prices(self):
        n = 50
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        close = pd.Series(range(1, n + 1), index=idx, dtype=float)
        df = pd.DataFrame({"Open": close, "High": close,
                           "Low": close, "Close": close, "Volume": 1_000_000.0})
        result = screen.add_indicators(df)
        # Warmup: first 14 rows have NaN RSI; from row 14 onward all gains, zero losses → RSI=100
        assert result["RSI14"].iloc[14:].eq(100).all()

    def test_rsi_is_0_for_monotonically_falling_prices(self):
        n = 50
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        close = pd.Series(range(n, 0, -1), index=idx, dtype=float)
        df = pd.DataFrame({"Open": close, "High": close,
                           "Low": close, "Close": close, "Volume": 1_000_000.0})
        result = screen.add_indicators(df)
        assert result["RSI14"].iloc[14:].eq(0).all()

    def test_atr_equals_true_range_for_constant_ohlc(self):
        # H=105, L=95, C=100 each day → TR = max(10, |105-100|, |95-100|) = 10
        n = 50
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        close = pd.Series([100.0] * n, index=idx)
        df = pd.DataFrame({
            "Open": close, "High": close + 5,
            "Low": close - 5, "Close": close, "Volume": 1_000_000.0,
        })
        result = screen.add_indicators(df)
        assert (result["ATR14"].dropna() == 10.0).all()

    def test_high20_is_rolling_max_of_high(self):
        df = make_ohlcv_df(n=200)
        result = screen.add_indicators(df)
        expected = df["High"].rolling(20).max()
        pd.testing.assert_series_equal(result["High20"], expected, check_names=False)

    def test_low20_is_rolling_min_of_low(self):
        df = make_ohlcv_df(n=200)
        result = screen.add_indicators(df)
        expected = df["Low"].rolling(20).min()
        pd.testing.assert_series_equal(result["Low20"], expected, check_names=False)


# ── fetch_daily ───────────────────────────────────────────────────────────────

SAMPLE_API_RESPONSE = {
    "Time Series (Daily)": {
        "2024-01-03": {"1. open": "151.0", "2. high": "153.0",
                       "3. low": "150.0", "4. close": "152.0", "5. volume": "1100000"},
        "2024-01-02": {"1. open": "150.0", "2. high": "152.0",
                       "3. low": "149.0", "4. close": "151.0", "5. volume": "1000000"},
    }
}


class TestFetchDaily:
    def _mock_get(self, payload):
        r = MagicMock()
        r.json.return_value = payload
        return r

    def test_returns_dataframe_on_success(self):
        with patch("requests.get", return_value=self._mock_get(SAMPLE_API_RESPONSE)):
            df = screen.fetch_daily("AAPL")
        assert isinstance(df, pd.DataFrame)

    def test_ohlcv_columns_are_present(self):
        with patch("requests.get", return_value=self._mock_get(SAMPLE_API_RESPONSE)):
            df = screen.fetch_daily("AAPL")
        assert {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns)

    def test_price_columns_are_numeric(self):
        with patch("requests.get", return_value=self._mock_get(SAMPLE_API_RESPONSE)):
            df = screen.fetch_daily("AAPL")
        for col in ["Open", "High", "Low", "Close"]:
            assert pd.api.types.is_float_dtype(df[col]), f"{col} is not float"

    def test_index_is_sorted_ascending(self):
        with patch("requests.get", return_value=self._mock_get(SAMPLE_API_RESPONSE)):
            df = screen.fetch_daily("AAPL")
        assert df.index.is_monotonic_increasing

    def test_returns_none_when_time_series_key_missing(self):
        payload = {"Note": "API rate limit reached."}
        with patch("requests.get", return_value=self._mock_get(payload)):
            result = screen.fetch_daily("AAPL")
        assert result is None

    def test_passes_symbol_and_function_to_api(self):
        with patch("requests.get", return_value=self._mock_get(SAMPLE_API_RESPONSE)) as mock_get:
            screen.fetch_daily("TSLA")
        params = mock_get.call_args[1]["params"]
        assert params["symbol"] == "TSLA"
        assert params["function"] == "TIME_SERIES_DAILY"


# ── send_telegram_text ────────────────────────────────────────────────────────

class TestSendTelegramText:
    def test_no_http_call_when_token_is_none(self):
        with patch.object(screen, "TELEGRAM_BOT_TOKEN", None), \
             patch("requests.post") as mock_post:
            screen.send_telegram_text("hello")
        mock_post.assert_not_called()

    def test_no_http_call_when_chat_id_is_none(self):
        with patch.object(screen, "TELEGRAM_BOT_TOKEN", "tok"), \
             patch.object(screen, "TELEGRAM_CHAT_ID", None), \
             patch("requests.post") as mock_post:
            screen.send_telegram_text("hello")
        mock_post.assert_not_called()

    def test_posts_to_sendmessage_endpoint(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch.object(screen, "TELEGRAM_BOT_TOKEN", "mytoken"), \
             patch.object(screen, "TELEGRAM_CHAT_ID", "123"), \
             patch("requests.post", return_value=mock_response) as mock_post:
            screen.send_telegram_text("hello")
        url = mock_post.call_args[0][0]
        assert "mytoken" in url
        assert "sendMessage" in url

    def test_non_200_response_does_not_raise(self):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        with patch.object(screen, "TELEGRAM_BOT_TOKEN", "tok"), \
             patch.object(screen, "TELEGRAM_CHAT_ID", "123"), \
             patch("requests.post", return_value=mock_response):
            screen.send_telegram_text("hello")  # must not raise


# ── send_telegram_document ────────────────────────────────────────────────────

class TestSendTelegramDocument:
    def test_no_http_call_when_unconfigured(self):
        with patch.object(screen, "TELEGRAM_BOT_TOKEN", None), \
             patch("requests.post") as mock_post:
            screen.send_telegram_document("/dev/null", "caption")
        mock_post.assert_not_called()

    def test_posts_to_senddocument_endpoint(self, tmp_path):
        tmp_file = tmp_path / "report.html"
        tmp_file.write_text("<html/>")
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch.object(screen, "TELEGRAM_BOT_TOKEN", "tok"), \
             patch.object(screen, "TELEGRAM_CHAT_ID", "123"), \
             patch("requests.post", return_value=mock_response) as mock_post:
            screen.send_telegram_document(str(tmp_file), "My caption")
        url = mock_post.call_args[0][0]
        assert "sendDocument" in url


# ── telegram_summary ─────────────────────────────────────────────────────────

class TestTelegramSummary:
    def _results(self):
        return pd.DataFrame([
            {"Ticker": "AAPL", "FinalGrade": "A", "Score": 90, "RR_Estimate": 2.5},
            {"Ticker": "MSFT", "FinalGrade": "B", "Score": 75, "RR_Estimate": 1.8},
            {"Ticker": "JPM",  "FinalGrade": "C", "Score": 55, "RR_Estimate": 1.2},
        ])

    def test_ab_tickers_appear_in_candidates_section(self):
        summary = screen.telegram_summary(self._results())
        ab_section = summary.split("Top 5:")[0]
        assert "AAPL" in ab_section
        assert "MSFT" in ab_section

    def test_rejected_ticker_not_in_ab_section(self):
        summary = screen.telegram_summary(self._results())
        ab_section = summary.split("Top 5:")[0]
        assert "JPM" not in ab_section

    def test_no_ab_candidates_shows_placeholder(self):
        results = pd.DataFrame([
            {"Ticker": "JPM", "FinalGrade": "C", "Score": 55, "RR_Estimate": 1.2},
        ])
        assert "No A/B candidates today." in screen.telegram_summary(results)

    def test_public_report_url_included_when_set(self):
        with patch.object(screen, "PUBLIC_REPORT_URL", "https://example.com/report"):
            summary = screen.telegram_summary(self._results())
        assert "https://example.com/report" in summary

    def test_public_report_url_omitted_when_empty(self):
        with patch.object(screen, "PUBLIC_REPORT_URL", ""):
            summary = screen.telegram_summary(self._results())
        assert "Open mobile dashboard" not in summary

    def test_top_5_section_is_present(self):
        assert "Top 5:" in screen.telegram_summary(self._results())


# ── HTML report generation ────────────────────────────────────────────────────

class TestMakeDesktopHtml:
    def _results(self):
        return pd.DataFrame([make_full_result_row()])

    def test_creates_file_at_given_path(self, tmp_path):
        path = str(tmp_path / "report.html")
        screen.make_desktop_html(self._results(), path)
        assert os.path.exists(path)

    def test_output_contains_ticker(self, tmp_path):
        path = str(tmp_path / "report.html")
        screen.make_desktop_html(self._results(), path)
        assert "AAPL" in open(path, encoding="utf-8").read()

    def test_output_contains_generated_timestamp(self, tmp_path):
        path = str(tmp_path / "report.html")
        screen.make_desktop_html(self._results(), path)
        assert "Generated" in open(path, encoding="utf-8").read()


class TestMakeMobileHtml:
    def _ab_result(self):
        return pd.DataFrame([make_full_result_row(Ticker="AAPL", FinalGrade="A")])

    def _rejected_result(self):
        return pd.DataFrame([make_full_result_row(
            Ticker="JPM", FinalGrade="C", Score=55, Status="Reject",
            CleanTrend="Weak", ClearSR="Check", VolumeConfirmation="Neutral",
            NotChasing="High", RR_1_to_2_Feasible="Fail", RR_Estimate=1.0,
        )])

    def test_creates_file_at_given_path(self, tmp_path):
        path = str(tmp_path / "index.html")
        screen.make_mobile_html(self._ab_result(), path)
        assert os.path.exists(path)

    def test_ab_ticker_appears_in_output(self, tmp_path):
        path = str(tmp_path / "index.html")
        screen.make_mobile_html(self._ab_result(), path)
        assert "AAPL" in open(path, encoding="utf-8").read()

    def test_no_ab_candidates_shows_placeholder(self, tmp_path):
        path = str(tmp_path / "index.html")
        screen.make_mobile_html(self._rejected_result(), path)
        assert "No A/B candidates today." in open(path, encoding="utf-8").read()

    def test_rejected_section_capped_at_10_rows(self, tmp_path):
        path = str(tmp_path / "index.html")
        rows = [make_full_result_row(
            Ticker=f"ZZ{i:02d}", FinalGrade="C", Score=50 + i,
            Status="Reject", CleanTrend="Weak", ClearSR="Check",
            VolumeConfirmation="Neutral", NotChasing="High",
            RR_1_to_2_Feasible="Fail", RR_Estimate=1.0,
        ) for i in range(15)]
        results = pd.DataFrame(rows)
        screen.make_mobile_html(results, path)
        content = open(path, encoding="utf-8").read()
        rendered = sum(1 for i in range(15) if f"ZZ{i:02d}" in content)
        assert rendered == 10

    def test_verdict_manual_review_for_grade_a_pass_rr(self, tmp_path):
        path = str(tmp_path / "index.html")
        screen.make_mobile_html(pd.DataFrame([make_full_result_row(FinalGrade="A", RR_1_to_2_Feasible="Pass")]), path)
        assert "Manual review only." in open(path, encoding="utf-8").read()

    def test_verdict_watch_only_for_grade_b_non_pass_rr(self, tmp_path):
        path = str(tmp_path / "index.html")
        screen.make_mobile_html(pd.DataFrame([make_full_result_row(FinalGrade="B", RR_1_to_2_Feasible="Close")]), path)
        assert "Watch only. R:R not fully 1:2." in open(path, encoding="utf-8").read()

    def test_verdict_reject_for_grade_c(self, tmp_path):
        path = str(tmp_path / "index.html")
        row = make_full_result_row(FinalGrade="C", Score=60, Status="Reject",
                                   CleanTrend="Weak", RR_1_to_2_Feasible="Pass")
        screen.make_mobile_html(pd.DataFrame([row]), path)
        assert "Reject under current rules." in open(path, encoding="utf-8").read()

    def test_summary_scanned_count(self, tmp_path):
        path = str(tmp_path / "index.html")
        rows = [
            make_full_result_row(Ticker="AAPL", FinalGrade="A"),
            make_full_result_row(Ticker="MSFT", FinalGrade="B"),
            make_full_result_row(Ticker="JPM",  FinalGrade="C", Score=60, Status="Reject",
                                 CleanTrend="Weak", RR_1_to_2_Feasible="Fail"),
        ]
        screen.make_mobile_html(pd.DataFrame(rows), path)
        content = open(path, encoding="utf-8").read()
        assert "<b>3</b><span>Scanned</span>" in content
        assert "<b>2</b><span>A/B</span>" in content
        assert "<b>1</b><span>A</span>" in content

    def test_tradingview_url_present_in_card(self, tmp_path):
        path = str(tmp_path / "index.html")
        screen.make_mobile_html(pd.DataFrame([make_full_result_row(Ticker="AAPL", FinalGrade="A")]), path)
        assert "tradingview.com" in open(path, encoding="utf-8").read()


class TestMakeDesktopHtmlContent:
    def _results(self, **overrides):
        return pd.DataFrame([make_full_result_row(**overrides)])

    def test_output_contains_final_grade(self, tmp_path):
        path = str(tmp_path / "report.html")
        screen.make_desktop_html(self._results(FinalGrade="A"), path)
        assert "A" in open(path, encoding="utf-8").read()

    def test_output_contains_score(self, tmp_path):
        path = str(tmp_path / "report.html")
        screen.make_desktop_html(self._results(Score=90), path)
        assert "90" in open(path, encoding="utf-8").read()


# ── send_telegram_document (extended) ────────────────────────────────────────

class TestSendTelegramDocumentExtended:
    def test_non_200_response_does_not_raise(self, tmp_path):
        tmp_file = tmp_path / "report.html"
        tmp_file.write_text("<html/>")
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        with patch.object(screen, "TELEGRAM_BOT_TOKEN", "tok"), \
             patch.object(screen, "TELEGRAM_CHAT_ID", "123"), \
             patch("requests.post", return_value=mock_response):
            screen.send_telegram_document(str(tmp_file), "caption")  # must not raise

    def test_caption_is_included_in_request(self, tmp_path):
        tmp_file = tmp_path / "report.html"
        tmp_file.write_text("<html/>")
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch.object(screen, "TELEGRAM_BOT_TOKEN", "tok"), \
             patch.object(screen, "TELEGRAM_CHAT_ID", "123"), \
             patch("requests.post", return_value=mock_response) as mock_post:
            screen.send_telegram_document(str(tmp_file), "My Caption")
        assert mock_post.call_args[1]["data"]["caption"] == "My Caption"


# ── add_indicators (extended) ─────────────────────────────────────────────────

class TestAddIndicatorsShortDf:
    def test_sma200_all_nan_for_df_shorter_than_200_rows(self):
        df = screen.add_indicators(make_ohlcv_df(n=50))
        assert df["SMA200"].isna().all()


# ── fetch_daily (extended) ────────────────────────────────────────────────────

class TestFetchDailyNetworkErrors:
    def test_raises_on_requests_timeout(self):
        with patch("requests.get", side_effect=requests.Timeout):
            with pytest.raises(requests.Timeout):
                screen.fetch_daily("AAPL")


# ── score_setup grade boundaries ─────────────────────────────────────────────

class TestScoreSetupGradeBoundaries:
    """Pin exact achievable scores at each grade boundary."""

    def test_score_85_is_grade_a(self):
        # Clean+20, Clear+20, Confirm+15, Borderline+10, Pass+15, News+5 = 85
        row = make_row(Close=106, SMA20=100, SMA50=90,
                       High20=160, Low20=80, RSI14=68,
                       Volume=1_200_000, AvgVolume20=1_000_000)
        result = screen.score_setup(row)
        assert result["Score"] == 85
        assert result["FinalGrade"] == "A"

    def test_score_81_is_grade_b(self):
        # Clean+20, Clear+20, Neutral+8, Low+20, Close+8, News+5 = 81
        row = make_row(Close=100, SMA20=98, SMA50=90,
                       High20=134, Low20=80, RSI14=55,
                       Volume=900_000, AvgVolume20=1_000_000)
        result = screen.score_setup(row)
        assert result["Score"] == 81
        assert result["FinalGrade"] == "B"

    def test_score_70_is_grade_b(self):
        # Clean+20, Clear+20, Confirm+15, Borderline+10, Fail+0, News+5 = 70
        row = make_row(Close=106, SMA20=100, SMA50=90,
                       High20=120, Low20=80, RSI14=68,
                       Volume=1_200_000, AvgVolume20=1_000_000)
        result = screen.score_setup(row)
        assert result["Score"] == 70
        assert result["FinalGrade"] == "B"

    def test_score_68_is_grade_c(self):
        # Clean+20, Clear+20, Neutral+8, High+0, Pass+15, News+5 = 68
        row = make_row(Close=120, SMA20=100, SMA50=90,
                       High20=200, Low20=80, RSI14=72,
                       Volume=900_000, AvgVolume20=1_000_000)
        result = screen.score_setup(row)
        assert result["Score"] == 68
        assert result["FinalGrade"] == "C"

    def test_score_50_is_grade_c(self):
        # Clean+20, Check+10, Confirm+15, High+0, Fail+0, News+5 = 50
        row = make_row(Close=120, SMA20=100, SMA50=90,
                       High20=125, Low20=80, RSI14=75,
                       Volume=1_200_000, AvgVolume20=1_000_000)
        result = screen.score_setup(row)
        assert result["Score"] == 50
        assert result["FinalGrade"] == "C"

    def test_score_48_is_grade_d(self):
        # Mixed+10, Check+10, Neutral+8, High+0, Pass+15, News+5 = 48
        # SMA20 < SMA50 → Mixed; close well above SMA20 → High chasing
        row = make_row(Close=100, SMA20=85, SMA50=90,
                       High20=200, Low20=80, RSI14=68,
                       Volume=900_000, AvgVolume20=1_000_000)
        result = screen.score_setup(row)
        assert result["Score"] == 48
        assert result["FinalGrade"] == "D"


# ── generate_sample_results ───────────────────────────────────────────────────

class TestGenerateSampleResults:
    def test_returns_dataframe(self):
        assert isinstance(screen.generate_sample_results(), pd.DataFrame)

    def test_contains_all_four_grades(self):
        grades = set(screen.generate_sample_results()["FinalGrade"].unique())
        assert grades == {"A", "B", "C", "D"}

    def test_sorted_by_grade_then_score_descending(self):
        df = screen.generate_sample_results()
        order = {"A": 1, "B": 2, "C": 3, "D": 4}
        grade_vals = df["FinalGrade"].map(order).tolist()
        assert grade_vals == sorted(grade_vals), "grades not in A→D order"
        for grade in ["A", "B", "C", "D"]:
            scores = df.loc[df["FinalGrade"] == grade, "Score"].tolist()
            assert scores == sorted(scores, reverse=True), f"scores within {grade} not descending"

    def test_grade_order_column_is_dropped(self):
        assert "GradeOrder" not in screen.generate_sample_results().columns


# ── main ──────────────────────────────────────────────────────────────────────

class TestMain:
    def test_sample_mode_writes_three_output_files(self, tmp_path):
        with patch("sys.argv", ["screen.py", "--sample"]), \
             patch.object(screen, "OUTPUT_DIR", str(tmp_path)), \
             patch("screen.send_telegram_text"), \
             patch("screen.send_telegram_document"):
            screen.main()
        assert (tmp_path / "daily_rule_report.csv").exists()
        assert (tmp_path / "daily_rule_report.html").exists()
        assert (tmp_path / "index.html").exists()

    def test_missing_api_key_raises_value_error(self):
        with patch("sys.argv", ["screen.py"]), \
             patch.object(screen, "API_KEY", None):
            with pytest.raises(ValueError, match="ALPHA_VANTAGE_API_KEY"):
                screen.main()

    def test_empty_results_sends_telegram_and_returns_early(self, tmp_path):
        tickers_file = tmp_path / "tickers.txt"
        tickers_file.write_text("AAPL\nMSFT\n")
        with patch("sys.argv", ["screen.py"]), \
             patch.object(screen, "API_KEY", "dummy"), \
             patch.object(screen, "TICKERS_FILE", str(tickers_file)), \
             patch("screen.fetch_daily", return_value=None), \
             patch("time.sleep"), \
             patch("screen.send_telegram_text") as mock_tg, \
             patch("screen.send_telegram_document") as mock_doc:
            screen.main()
        mock_tg.assert_called_once()
        assert "no results" in mock_tg.call_args[0][0].lower()
        mock_doc.assert_not_called()

    def test_rate_limit_sleep_called_per_ticker(self, tmp_path):
        tickers_file = tmp_path / "tickers.txt"
        tickers_file.write_text("AAPL\nMSFT\n")
        with patch("sys.argv", ["screen.py"]), \
             patch.object(screen, "API_KEY", "dummy"), \
             patch.object(screen, "TICKERS_FILE", str(tickers_file)), \
             patch.object(screen, "OUTPUT_DIR", str(tmp_path)), \
             patch("screen.fetch_daily", return_value=make_ohlcv_df(n=200)), \
             patch("time.sleep") as mock_sleep, \
             patch("screen.send_telegram_text"), \
             patch("screen.send_telegram_document"):
            screen.main()
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(15)
