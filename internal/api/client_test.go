package api

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestChatStreamRequiresDoneEvent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = writer.Write([]byte("event: token\ndata: {\"delta\":\"hello\"}\n\n"))
	}))
	defer server.Close()

	err := (&Client{BaseURL: server.URL}).ChatStream(context.Background(), "model", "conversation", "hi", func(string) error {
		return nil
	})
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("error = %v, want io.ErrUnexpectedEOF", err)
	}
}

func TestChatStreamReportsErrorEvent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = writer.Write([]byte("event: error\ndata: {\"error\":\"backend failed\"}\n\n"))
	}))
	defer server.Close()

	err := (&Client{BaseURL: server.URL}).ChatStream(context.Background(), "model", "conversation", "hi", func(string) error {
		return nil
	})
	if err == nil || !strings.Contains(err.Error(), "backend failed") {
		t.Fatalf("error = %v, want backend error", err)
	}
}

func TestRefreshModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost || request.URL.Path != "/api/v1/models/refresh" {
			t.Fatalf("request = %s %s", request.Method, request.URL.Path)
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"models":[{"name":"demo"}]}`))
	}))
	defer server.Close()

	models, err := (&Client{BaseURL: server.URL}).RefreshModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0].Name != "demo" {
		t.Fatalf("models = %#v", models)
	}
}
