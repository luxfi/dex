"""Tests for LX Trading SDK risk module."""

import pytest
from decimal import Decimal

from lx_trading.risk import RiskManager, RiskError
from lx_trading.config import RiskConfig
from lx_trading.types import OrderRequest, Side, OrderType, TimeInForce


def create_risk_config(**kwargs) -> RiskConfig:
    defaults = {
        "enabled": True,
        "max_position_size": Decimal("100"),
        "max_order_size": Decimal("10"),
        "max_daily_loss": Decimal("1000"),
        "max_open_orders": 5,
        "kill_switch_enabled": False,
        "position_limits": {},
    }
    defaults.update(kwargs)
    return RiskConfig(**defaults)


def create_order_request(**kwargs) -> OrderRequest:
    defaults = {
        "symbol": "BTC-USDC",
        "side": Side.BUY,
        "order_type": OrderType.LIMIT,
        "quantity": Decimal("1"),
        "price": Decimal("50000"),
        "time_in_force": TimeInForce.GTC,
    }
    defaults.update(kwargs)
    return OrderRequest(**defaults)


class TestRiskManagerConfiguration:
    def test_enabled_flag(self):
        enabled_manager = RiskManager(create_risk_config(enabled=True))
        assert enabled_manager.is_enabled is True

        disabled_manager = RiskManager(create_risk_config(enabled=False))
        assert disabled_manager.is_enabled is False

    def test_skip_validation_when_disabled(self):
        manager = RiskManager(create_risk_config(enabled=False))
        # Oversized order should pass when disabled
        order = create_order_request(quantity=Decimal("1000"))
        manager.validate_order(order)  # Should not raise


class TestKillSwitch:
    def test_starts_inactive(self):
        manager = RiskManager(create_risk_config())
        assert manager.is_killed is False

    def test_activate_kill(self):
        manager = RiskManager(create_risk_config())
        manager.kill()
        assert manager.is_killed is True

    def test_deactivate_reset(self):
        manager = RiskManager(create_risk_config())
        manager.kill()
        manager.reset()
        assert manager.is_killed is False

    def test_blocks_all_orders_when_active(self):
        manager = RiskManager(create_risk_config())
        manager.kill()

        order = create_order_request()
        with pytest.raises(RiskError, match="Kill switch"):
            manager.validate_order(order)


class TestOrderSizeLimits:
    def test_reject_oversized_order(self):
        manager = RiskManager(create_risk_config(max_order_size=Decimal("5")))

        order = create_order_request(quantity=Decimal("10"))
        with pytest.raises(RiskError, match="Order size"):
            manager.validate_order(order)

    def test_allow_order_within_limit(self):
        manager = RiskManager(create_risk_config(max_order_size=Decimal("10")))

        order = create_order_request(quantity=Decimal("5"))
        manager.validate_order(order)  # Should not raise

    def test_allow_any_size_when_no_limit(self):
        manager = RiskManager(create_risk_config(
            max_order_size=Decimal("0"),
            max_position_size=Decimal("0"),
        ))

        order = create_order_request(quantity=Decimal("1000"))
        manager.validate_order(order)  # Should not raise


class TestPositionLimits:
    def test_reject_order_exceeding_max_position(self):
        manager = RiskManager(create_risk_config(max_position_size=Decimal("10")))

        # Simulate existing position
        manager.update_position("BTC", Decimal("8"), Side.BUY)

        # Trying to buy 5 more would exceed limit (8 + 5 = 13 > 10)
        order = create_order_request(quantity=Decimal("5"))
        with pytest.raises(RiskError, match="position"):
            manager.validate_order(order)

    def test_allow_order_within_position_limit(self):
        manager = RiskManager(create_risk_config(max_position_size=Decimal("10")))

        manager.update_position("BTC", Decimal("5"), Side.BUY)

        order = create_order_request(quantity=Decimal("3"))
        manager.validate_order(order)  # Should not raise

    def test_asset_specific_limits(self):
        manager = RiskManager(create_risk_config(
            position_limits={"BTC": Decimal("5")},
        ))

        manager.update_position("BTC", Decimal("4"), Side.BUY)

        # Trying to buy 2 more exceeds BTC-specific limit
        order = create_order_request(quantity=Decimal("2"))
        with pytest.raises(RiskError, match="Position limit"):
            manager.validate_order(order)


class TestOpenOrdersLimit:
    def test_reject_when_max_reached(self):
        manager = RiskManager(create_risk_config(max_open_orders=3))

        manager.order_opened("BTC-USDC")
        manager.order_opened("BTC-USDC")
        manager.order_opened("BTC-USDC")

        order = create_order_request()
        with pytest.raises(RiskError, match="open orders"):
            manager.validate_order(order)

    def test_allow_when_under_limit(self):
        manager = RiskManager(create_risk_config(max_open_orders=5))

        manager.order_opened("BTC-USDC")
        manager.order_opened("BTC-USDC")

        order = create_order_request()
        manager.validate_order(order)  # Should not raise

    def test_track_per_symbol(self):
        manager = RiskManager(create_risk_config(max_open_orders=2))

        manager.order_opened("BTC-USDC")
        manager.order_opened("BTC-USDC")

        # Different symbol should still be allowed
        order = create_order_request(symbol="ETH-USDC")
        manager.validate_order(order)  # Should not raise

    def test_decrement_on_close(self):
        manager = RiskManager(create_risk_config(max_open_orders=2))

        manager.order_opened("BTC-USDC")
        manager.order_opened("BTC-USDC")
        manager.order_closed("BTC-USDC")

        order = create_order_request()
        manager.validate_order(order)  # Should not raise


class TestDailyLossLimit:
    def test_reject_when_exceeded(self):
        manager = RiskManager(create_risk_config(max_daily_loss=Decimal("100")))

        manager.update_pnl(Decimal("-150"))

        order = create_order_request()
        with pytest.raises(RiskError, match="Daily loss"):
            manager.validate_order(order)

    def test_allow_when_within_limit(self):
        manager = RiskManager(create_risk_config(max_daily_loss=Decimal("100")))

        manager.update_pnl(Decimal("-50"))

        order = create_order_request()
        manager.validate_order(order)  # Should not raise

    def test_auto_trigger_kill_switch(self):
        manager = RiskManager(create_risk_config(
            max_daily_loss=Decimal("100"),
            kill_switch_enabled=True,
        ))

        manager.update_pnl(Decimal("-150"))
        assert manager.is_killed is True

    def test_no_auto_trigger_when_disabled(self):
        manager = RiskManager(create_risk_config(
            max_daily_loss=Decimal("100"),
            kill_switch_enabled=False,
        ))

        manager.update_pnl(Decimal("-150"))
        assert manager.is_killed is False


class TestPositionTracking:
    def test_track_buys(self):
        manager = RiskManager(create_risk_config())

        manager.update_position("BTC", Decimal("5"), Side.BUY)
        assert manager.position("BTC") == Decimal("5")

        manager.update_position("BTC", Decimal("3"), Side.BUY)
        assert manager.position("BTC") == Decimal("8")

    def test_track_sells(self):
        manager = RiskManager(create_risk_config())

        manager.update_position("BTC", Decimal("10"), Side.BUY)
        manager.update_position("BTC", Decimal("3"), Side.SELL)
        assert manager.position("BTC") == Decimal("7")

    def test_allow_negative_positions(self):
        manager = RiskManager(create_risk_config())

        manager.update_position("BTC", Decimal("5"), Side.SELL)
        assert manager.position("BTC") == Decimal("-5")

    def test_unknown_asset_returns_zero(self):
        manager = RiskManager(create_risk_config())
        assert manager.position("UNKNOWN") == Decimal("0")

    def test_all_positions(self):
        manager = RiskManager(create_risk_config())

        manager.update_position("BTC", Decimal("5"), Side.BUY)
        manager.update_position("ETH", Decimal("10"), Side.BUY)

        positions = manager.positions()
        assert len(positions) == 2
        assert positions["BTC"] == Decimal("5")
        assert positions["ETH"] == Decimal("10")


class TestPnlTracking:
    def test_track_daily_pnl(self):
        manager = RiskManager(create_risk_config())

        manager.update_pnl(Decimal("50"))
        assert manager.daily_pnl == Decimal("50")

        manager.update_pnl(Decimal("-20"))
        assert manager.daily_pnl == Decimal("30")

    def test_reset_daily_pnl(self):
        manager = RiskManager(create_risk_config())

        manager.update_pnl(Decimal("100"))
        manager.reset_daily_pnl()

        assert manager.daily_pnl == Decimal("0")


class TestOpenOrdersTracking:
    def test_track_count(self):
        manager = RiskManager(create_risk_config())

        assert manager.open_orders("BTC-USDC") == 0

        manager.order_opened("BTC-USDC")
        manager.order_opened("BTC-USDC")
        assert manager.open_orders("BTC-USDC") == 2

        manager.order_closed("BTC-USDC")
        assert manager.open_orders("BTC-USDC") == 1

    def test_not_go_below_zero(self):
        manager = RiskManager(create_risk_config())

        manager.order_closed("BTC-USDC")
        manager.order_closed("BTC-USDC")

        assert manager.open_orders("BTC-USDC") == 0


class TestRiskError:
    def test_error_message(self):
        error = RiskError("test message")
        assert str(error) == "test message"
