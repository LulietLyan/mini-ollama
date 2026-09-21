package runner

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestWaitForHTTPReady(t *testing.T) {
	var requests atomic.Int32

	server := httptest.NewServer(
		http.HandlerFunc(func(
			writer http.ResponseWriter,
			request *http.Request,
		) {
			if requests.Add(1) < 3 {
				writer.WriteHeader(http.StatusServiceUnavailable)
				return
			}

			writer.WriteHeader(http.StatusOK)
		}),
	)
	defer server.Close()

	err := WaitForHTTPReady(
		context.Background(),
		ReadinessConfig{
			URL:      server.URL,
			Timeout:  1 * time.Second,
			Interval: 5 * time.Millisecond,
		},
	)
	if err != nil {
		t.Fatalf("WaitForHTTPReady returned error: %v", err)
	}

	if requests.Load() < 3 {
		t.Fatalf(
			"request count = %d, want at least 3",
			requests.Load(),
		)
	}
}

func TestWaitForHTTPReadyTimeout(t *testing.T) {
	server := httptest.NewServer(
		http.HandlerFunc(func(
			writer http.ResponseWriter,
			request *http.Request,
		) {
			writer.WriteHeader(http.StatusServiceUnavailable)
		}),
	)
	defer server.Close()

	err := WaitForHTTPReady(
		context.Background(),
		ReadinessConfig{
			URL:      server.URL,
			Timeout:  50 * time.Millisecond,
			Interval: 5 * time.Millisecond,
		},
	)
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
}
