package lx

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

// Test Chainlink price source functionality
func TestChainlinkSource(t *testing.T) {
	t.Run("NewChainlinkPriceSource", func(t *testing.T) {
		source := NewChainlinkPriceSource()
		assert.NotNil(t, source)
		assert.NotNil(t, source.feedAddresses)
	})

	source := NewChainlinkPriceSource()

	t.Run("initChainlinkFeeds", func(t *testing.T) {
		// Test package-level function
		feeds := initChainlinkFeeds()
		assert.NotNil(t, feeds)
	})

	t.Run("Start", func(t *testing.T) {
		err := source.Start()
		assert.NoError(t, err)
	})

	t.Run("pollLoop", func(t *testing.T) {
		// Should not panic
		go source.pollLoop()
	})

	t.Run("pollAllFeeds", func(t *testing.T) {
		// Should not panic
		source.pollAllFeeds()
	})

	t.Run("pollFeed", func(t *testing.T) {
		// Should not panic - simplified test
		ctx := context.Background()
		source.pollFeed(ctx, "ETH/USD", "0x123")
	})

	t.Run("simulateChainlinkPrice", func(t *testing.T) {
		price := source.simulateChainlinkPrice("ETH/USD")
		assert.GreaterOrEqual(t, price, 0.0) // Can be 0 in simulation
	})

	t.Run("GetPrice", func(t *testing.T) {
		priceData, err := source.GetPrice("ETH/USD")
		assert.NoError(t, err)
		assert.NotNil(t, priceData)
	})

	t.Run("GetPrices", func(t *testing.T) {
		pairs := []string{"ETH/USD", "BTC/USD"}
		prices, err := source.GetPrices(pairs)
		assert.NoError(t, err)
		assert.NotNil(t, prices)
	})

	t.Run("Subscribe", func(t *testing.T) {
		source.Subscribe("ETH/USD")
	})

	t.Run("Unsubscribe", func(t *testing.T) {
		source.Unsubscribe("ETH/USD")
	})

	t.Run("IsHealthy", func(t *testing.T) {
		healthy := source.IsHealthy()
		assert.True(t, healthy)
	})

	t.Run("GetName", func(t *testing.T) {
		name := source.GetName()
		assert.Equal(t, "chainlink", name)
	})

	t.Run("GetWeight", func(t *testing.T) {
		weight := source.GetWeight()
		assert.Greater(t, weight, 0.0) // Weight can vary
	})

	t.Run("Close", func(t *testing.T) {
		err := source.Close()
		assert.NoError(t, err)
	})

	t.Run("GetLatestRoundData", func(t *testing.T) {
		data, err := source.GetLatestRoundData("ETH/USD")
		assert.NoError(t, err)
		assert.NotNil(t, data)
	})
}