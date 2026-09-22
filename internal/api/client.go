package api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

type Conversation struct {
	ID        string `json:"id"`
	Model     string `json:"model,omitempty"`
	CreatedAt string `json:"created_at,omitempty"`
}

type Model struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	SizeBytes   int64  `json:"size_bytes"`
}

type Status struct {
	Lifecycle      string  `json:"lifecycle"`
	Model          string  `json:"model"`
	Error          string  `json:"error"`
	StartedAt      *string `json:"started_at"`
	ReadyAt        *string `json:"ready_at"`
	LoadDurationMs int64   `json:"load_duration_ms"`
	RequestCount   uint64  `json:"request_count"`
}

func (c *Client) ListModels(ctx context.Context) ([]Model, error) {
	var result struct {
		Models []Model `json:"models"`
	}
	if err := c.doJSON(ctx, http.MethodGet, "/api/v1/models", nil, &result); err != nil {
		return nil, err
	}
	return result.Models, nil
}

func (c *Client) RefreshModels(ctx context.Context) ([]Model, error) {
	var result struct {
		Models []Model `json:"models"`
	}
	if err := c.doJSON(ctx, http.MethodPost, "/api/v1/models/refresh", nil, &result); err != nil {
		return nil, err
	}
	return result.Models, nil
}

func (c *Client) Status(ctx context.Context) (Status, error) {
	var result Status
	if err := c.doJSON(ctx, http.MethodGet, "/api/v1/status", nil, &result); err != nil {
		return Status{}, err
	}
	return result, nil
}

func (c *Client) Stop(ctx context.Context) error {
	return c.doJSON(ctx, http.MethodPost, "/api/v1/service/stop", nil, &struct{}{})
}

func (c *Client) Switch(ctx context.Context, model string) error {
	payload, err := json.Marshal(map[string]string{"model": model})
	if err != nil {
		return err
	}
	return c.doJSON(ctx, http.MethodPost, "/api/v1/service/switch", payload, &struct{}{})
}

func (c *Client) CreateConversation(ctx context.Context, model string) (Conversation, error) {
	payload, err := json.Marshal(map[string]string{"model": model})
	if err != nil {
		return Conversation{}, err
	}
	var result Conversation
	if err := c.doJSON(ctx, http.MethodPost, "/api/v1/conversations", payload, &result); err != nil {
		return Conversation{}, err
	}
	return result, nil
}

func (c *Client) ChatStream(ctx context.Context, model, conversationID, message string, onDelta func(string) error) error {
	if onDelta == nil {
		onDelta = func(string) error { return nil }
	}
	payload, err := json.Marshal(map[string]any{
		"model":           model,
		"conversation_id": conversationID,
		"message":         message,
		"stream":          true,
	})
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint("/api/v1/chat"), bytes.NewReader(payload))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	client := c.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	response, err := client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		return fmt.Errorf("mini-ollama returned HTTP %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}

	scanner := bufio.NewScanner(response.Body)
	scanner.Buffer(make([]byte, 4096), 4<<20)
	event := ""
	data := make([]string, 0, 1)
	doneSeen := false
	flush := func() error {
		if event == "" && len(data) == 0 {
			return nil
		}
		if event == "done" {
			doneSeen = true
			event = ""
			data = data[:0]
			return nil
		}
		if event == "error" {
			var value struct {
				Error string `json:"error"`
			}
			if err := json.Unmarshal([]byte(strings.Join(data, "\n")), &value); err != nil {
				return fmt.Errorf("decode SSE error: %w", err)
			}
			if value.Error == "" {
				value.Error = "server returned an SSE error"
			}
			return fmt.Errorf("chat stream: %s", value.Error)
		}
		if event != "token" || len(data) == 0 {
			event = ""
			data = data[:0]
			return nil
		}
		var value struct {
			Delta string `json:"delta"`
		}
		if err := json.Unmarshal([]byte(strings.Join(data, "\n")), &value); err != nil {
			return err
		}
		if value.Delta != "" {
			if err := onDelta(value.Delta); err != nil {
				return err
			}
		}
		event = ""
		data = data[:0]
		return nil
	}

	for scanner.Scan() {
		line := strings.TrimSuffix(scanner.Text(), "\r")
		if line == "" {
			if err := flush(); err != nil {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		if strings.HasPrefix(line, "event:") {
			event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}
		if strings.HasPrefix(line, "data:") {
			data = append(data, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	if err := flush(); err != nil {
		return err
	}
	if !doneSeen {
		return io.ErrUnexpectedEOF
	}
	return nil
}

func (c *Client) doJSON(ctx context.Context, method, path string, body []byte, output any) error {
	request, err := http.NewRequestWithContext(ctx, method, c.endpoint(path), bytes.NewReader(body))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	client := c.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	response, err := client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		return fmt.Errorf("mini-ollama returned HTTP %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}
	return json.NewDecoder(response.Body).Decode(output)
}

func (c *Client) endpoint(path string) string {
	return strings.TrimRight(c.BaseURL, "/") + path
}
