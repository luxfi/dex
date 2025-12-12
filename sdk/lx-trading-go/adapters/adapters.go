// Package adapters provides venue adapter implementations.
package adapters

import (
	"github.com/luxfi/trading"
)

func init() {
	trading.RegisterAdapterFactory(&factory{})
}

type factory struct{}

func (f *factory) CreateDexAdapter(name string, cfg *trading.NativeVenueConfig) trading.VenueAdapter {
	return NewLxDexAdapter(name, cfg)
}

func (f *factory) CreateAmmAdapter(name string, cfg *trading.NativeVenueConfig) trading.VenueAdapter {
	return NewLxAmmAdapter(name, cfg)
}

func (f *factory) CreateCcxtAdapter(name string, cfg *trading.CcxtConfig) trading.VenueAdapter {
	return NewCcxtAdapter(name, cfg)
}

func (f *factory) CreateHummingbotAdapter(name string, cfg *trading.HummingbotConfig) trading.VenueAdapter {
	return NewHummingbotAdapter(name, cfg)
}
