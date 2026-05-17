//go:build grpc

package client

import "testing"

func TestTimeInForceConversion(t *testing.T) {
	tests := []struct {
		name string
		tif  TimeInForce
	}{
		{"GTC", TimeInForceGTC},
		{"IOC", TimeInForceIOC},
		{"FOK", TimeInForceFOK},
		{"DAY", TimeInForceDAY},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Just ensure it doesn't panic
			result := timeInForceToProto(tt.tif)
			if result < 0 || result > 3 {
				t.Errorf("timeInForceToProto(%s) returned invalid value %d", tt.tif, result)
			}
		})
	}
}

func TestTimeInForceToProtoDefault(t *testing.T) {
	// Unknown value should default to GTC (0)
	result := timeInForceToProto(TimeInForce("UNKNOWN"))
	if result != 0 {
		t.Errorf("Expected default to be 0 (GTC), got %d", result)
	}
}
