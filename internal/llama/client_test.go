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
