package llama

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestChatReadsSSE(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = writer.Write([]byte(
			": ping\n\n" +
				"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
				"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n" +
				"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n" +
				"data: [DONE]\n\n",
		))
	}))
	defer server.Close()

	var callback string
	response, err := (&Client{}).Chat(context.Background(), server.URL, ChatRequest{Stream: true}, func(delta string) error {
		callback += delta
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if response != "hello world" || callback != response {
		t.Fatalf("response = %q, callback = %q", response, callback)
	}
}

func TestChatReadsUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`))
	}))
	defer server.Close()

	result, err := (&Client{}).ChatResult(context.Background(), server.URL, ChatRequest{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != "ok" || result.Usage == nil || result.Usage.TotalTokens != 5 {
		t.Fatalf("result = %#v", result)
	}
}

func TestChatSSEReadsUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = writer.Write([]byte(
			"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n" +
				"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":1,\"total_tokens\":5}}\n\n" +
				"data: [DONE]\n\n",
		))
	}))
	defer server.Close()

	result, err := (&Client{}).ChatResult(context.Background(), server.URL, ChatRequest{Stream: true}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != "ok" || result.Usage == nil || result.Usage.TotalTokens != 5 {
		t.Fatalf("result = %#v", result)
	}
}
