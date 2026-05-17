package client

import (
	"testing"
)

func TestNewClient(t *testing.T) {
	t.Run("default options", func(t *testing.T) {
		client, err := NewClient()
		if err != nil {
			t.Fatalf("NewClient() error = %v", err)
		}
		if client == nil {
			t.Fatal("NewClient() returned nil")
		}
		if client.jsonRPCURL != "http://localhost:8080" {
			t.Errorf("Expected jsonRPCURL http://localhost:8080, got %s", client.jsonRPCURL)
		}
		if client.wsURL != "ws://localhost:8081" {
			t.Errorf("Expected wsURL ws://localhost:8081, got %s", client.wsURL)
		}
	})

	t.Run("with options", func(t *testing.T) {
		client, err := NewClient(
			WithJSONRPCURL("http://custom:9090"),
			WithWebSocketURL("ws://custom:9091"),
			WithGRPCURL("custom:50052"),
			WithAPIKey("test-api-key"),
		)
		if err != nil {
			t.Fatalf("NewClient() error = %v", err)
		}
		if client.jsonRPCURL != "http://custom:9090" {
			t.Errorf("Expected jsonRPCURL http://custom:9090, got %s", client.jsonRPCURL)
		}
		if client.wsURL != "ws://custom:9091" {
			t.Errorf("Expected wsURL ws://custom:9091, got %s", client.wsURL)
		}
		if client.grpcURL != "custom:50052" {
			t.Errorf("Expected grpcURL custom:50052, got %s", client.grpcURL)
		}
		if client.apiKey != "test-api-key" {
			t.Errorf("Expected apiKey test-api-key, got %s", client.apiKey)
		}
	})
}

func TestClientDisconnect(t *testing.T) {
	client, err := NewClient()
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}

	// Should not panic when not connected
	err = client.Disconnect()
	if err != nil {
		t.Errorf("Disconnect() error = %v", err)
	}
}
