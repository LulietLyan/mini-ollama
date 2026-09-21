package llama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Stream      bool      `json:"stream"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	Stop        []string  `json:"stop,omitempty"`
}

type Client struct {
	HTTPClient *http.Client
}

func (c *Client) Chat(ctx context.Context, backendURL string, request ChatRequest, onDelta func(string) error) (string, error) {
	if onDelta == nil {
		onDelta = func(string) error { return nil }
	}
	payload, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("encode chat request: %w", err)
	}

	endpoint := strings.TrimRight(backendURL, "/") + "/v1/chat/completions"
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return "", fmt.Errorf("create chat request: %w", err)
	}
	httpRequest.Header.Set("Content-Type", "application/json")

	client := c.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	response, err := client.Do(httpRequest)
	if err != nil {
		return "", fmt.Errorf("call llama-server: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		return "", fmt.Errorf("llama-server returned HTTP %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}

	if !request.Stream {
		return readComplete(response.Body)
	}
	return readSSE(response.Body, onDelta)
}

func readComplete(reader io.Reader) (string, error) {
	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(reader).Decode(&response); err != nil {
		return "", fmt.Errorf("decode chat response: %w", err)
	}
	if len(response.Choices) == 0 {
		return "", errors.New("chat response contains no choices")
	}
	return response.Choices[0].Message.Content, nil
}

func readSSE(reader io.Reader, onDelta func(string) error) (string, error) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 4096), 4<<20)
	var data []string
	var response strings.Builder
	finished := false

	flush := func() error {
		if len(data) == 0 {
			return nil
		}
		payload := strings.Join(data, "\n")
		data = data[:0]
		if payload == "[DONE]" {
			finished = true
			return nil
		}

		var chunk struct {
			Error *struct {
				Message string `json:"message"`
			} `json:"error,omitempty"`
			Choices []struct {
				Delta struct {
					Content *string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return fmt.Errorf("decode llama SSE event: %w", err)
		}
		if chunk.Error != nil {
			return errors.New(chunk.Error.Message)
		}
		if len(chunk.Choices) == 0 || chunk.Choices[0].Delta.Content == nil {
			return nil
		}
		delta := *chunk.Choices[0].Delta.Content
		if delta == "" {
			return nil
		}
		if err := onDelta(delta); err != nil {
			return err
		}
		response.WriteString(delta)
		return nil
	}

	for scanner.Scan() {
		line := strings.TrimSuffix(scanner.Text(), "\r")
		if line == "" {
			if err := flush(); err != nil {
				return "", err
			}
			if finished {
				break
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		if strings.HasPrefix(line, "data:") {
			data = append(data, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("read llama SSE stream: %w", err)
	}
	if !finished {
		if err := flush(); err != nil {
			return "", err
		}
	}
	return response.String(), nil
}
