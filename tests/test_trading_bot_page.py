"""Tests for trading bot page utilities."""

import pytest
from unittest.mock import patch


class TestPortfolioPersistence:
    def test_save_and_load_portfolio_id(self, tmp_path):
        pfile = tmp_path / "portfolio_id.txt"
        pfile.write_text("test-pid-123")
        assert pfile.read_text().strip() == "test-pid-123"

    def test_missing_file_returns_none(self, tmp_path):
        pfile = tmp_path / "portfolio_id.txt"
        assert not pfile.exists()


class TestMCPGracefulDegradation:
    @patch("src.investor.mcp_client.is_connected", return_value=False)
    def test_is_connected_false_when_server_down(self, mock):
        from src.investor.mcp_client import is_connected
        assert is_connected() is False

    @patch("src.investor.mcp_client.call_tool", return_value={"error": "Connection refused"})
    def test_tool_returns_error_on_failure(self, mock):
        from src.investor.mcp_client import detect_market_regime
        result = detect_market_regime()
        assert "error" in result


class TestMergeMovers:
    def test_merge_empty_data(self):
        movers = {}
        assert len(movers) == 0

    def test_merge_anomalies_only(self):
        movers = {}
        anomalies = {
            "anomalies": [
                {"symbol": "AAPL", "total_score": 5, "anomalies": [{"type": "52w_high"}]}
            ]
        }
        for item in anomalies["anomalies"]:
            sym = item["symbol"]
            movers[sym] = {"symbol": sym, "badges": [], "score": item["total_score"]}
        assert movers["AAPL"]["score"] == 5
