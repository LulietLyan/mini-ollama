package api

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"mini-ollama/internal/catalog"
	"mini-ollama/internal/history"
	modelservice "mini-ollama/internal/service"
)

func newOpenAITestHTTP(t *testing.T, backend http.Handler, lifecycle modelservice.Lifecycle) *httptest.Server {
	t.Helper()

	modelsDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelsDir, "demo.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}

	modelCatalog, err := catalog.New(modelsDir)
	if err != nil {
		t.Fatal(err)
	}

	store, err := history.Open(filepath.Join(t.TempDir(), "mini-ollama.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = store.Close()
	})

	backendServer := httptest.NewServer(backend)
	t.Cleanup(backendServer.Close)

	server := New(Config{
		Catalog: modelCatalog,
		History: store,
		BackendURL: func() (string, error) {
			return backendServer.URL, nil
		},
		Status: func() modelservice.Status {
			return modelservice.Status{
				Lifecycle: lifecycle,
				Model:     "demo",
			}
		},
	})

	return httptest.NewServer(server.Handler())
}

func TestOpenAIModels(t *testing.T) {
	server := newOpenAITestHTTP(
		t,
		http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}),
		modelservice.LifecycleStopped,
	)
	defer server.Close()

	response, err := http.Get(server.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", response.StatusCode)
	}

	var body struct {
		Object string        `json:"object"`
		Data   []openAIModel `json:"data"`
	}

	if err := json.NewDecoder(response.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}

	if body.Object != "list" ||
		len(body.Data) != 1 ||
		body.Data[0].ID != "demo" {
		t.Fatalf("body = %#v", body)
	}
}

func TestOpenAIChatCompletion(t *testing.T) {
	backend := http.HandlerFunc(func(
		writer http.ResponseWriter,
		request *http.Request,
	) {
		if request.URL.Path != "/v1/chat/completions" {
			t.Fatalf("backend path = %s", request.URL.Path)
		}

		var payload struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}

		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}

		if len(payload.Messages) != 2 ||
			payload.Messages[0].Role != "system" ||
			payload.Messages[1].Content != "hello" {
			t.Fatalf("messages = %#v", payload.Messages)
		}

		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(
			writer,
			`{"choices":[{"message":{"role":"assistant","content":"world"}}]}`,
		)
	})

	server := newOpenAITestHTTP(t, backend, modelservice.LifecycleReady)
	defer server.Close()

	requestBody := strings.NewReader(
		`{"model":"demo","messages":[{"role":"developer","content":"be concise"},{"role":"user","content":"hello"}]}`,
	)

	request, err := http.NewRequestWithContext(
		context.Background(),
		http.MethodPost,
		server.URL+"/v1/chat/completions",
		requestBody,
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")

	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(response.Body)
		t.Fatalf("status = %d body = %s", response.StatusCode, data)
	}

	var body openAICompletion
	if err := json.NewDecoder(response.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}

	if body.Object != "chat.completion" ||
		body.Model != "demo" ||
		body.Choices[0].Message.Content != "world" {
		t.Fatalf("body = %#v", body)
	}
}

func TestOpenAIStreamingCompletion(t *testing.T) {
	backend := http.HandlerFunc(func(
		writer http.ResponseWriter,
		request *http.Request,
	) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(
			writer,
			"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"+
				"data: [DONE]\n\n",
		)
	})

	server := newOpenAITestHTTP(t, backend, modelservice.LifecycleReady)
	defer server.Close()

	requestBody := strings.NewReader(
		`{"model":"demo","messages":[{"role":"user","content":"hello"}],"stream":true}`,
	)

	request, err := http.NewRequest(
		http.MethodPost,
		server.URL+"/v1/chat/completions",
		requestBody,
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")

	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()

	data, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}

	body := string(data)
	if response.StatusCode != http.StatusOK ||
		!strings.Contains(body, `"chat.completion.chunk"`) ||
		!strings.Contains(body, `"content":"hello"`) ||
		!strings.Contains(body, `"finish_reason":"stop"`) ||
		!strings.HasSuffix(strings.TrimSpace(body), "data: [DONE]") {
		t.Fatalf("status = %d body = %s", response.StatusCode, body)
	}
}

func TestOpenAIRejectsUnavailableService(t *testing.T) {
	server := newOpenAITestHTTP(
		t,
		http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}),
		modelservice.LifecycleStopped,
	)
	defer server.Close()

	requestBody := strings.NewReader(
		`{"model":"demo","messages":[{"role":"user","content":"hello"}]}`,
	)

	request, err := http.NewRequest(
		http.MethodPost,
		server.URL+"/v1/chat/completions",
		requestBody,
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")

	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d", response.StatusCode)
	}
}
