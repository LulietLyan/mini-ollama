package api

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"mini-ollama/internal/llama"
	modelservice "mini-ollama/internal/service"
	"net/http"
	"os"
	"strings"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
)

const maxOpenAIContent = 64 << 10

var completionSequence atomic.Uint64

type openAIChatRequest struct {
	Model               string            `json:"model"`
	Messages            []openAIMessage   `json:"messages"`
	Stream              bool              `json:"stream,omitempty"`
	StreamOptions       *openAIStreamOpts `json:"stream_options,omitempty"`
	MaxTokens           *int              `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int              `json:"max_completion_tokens,omitempty"`
	Temperature         *float64          `json:"temperature,omitempty"`
	Stop                json.RawMessage   `json:"stop,omitempty"`
	N                   int               `json:"n,omitempty"`
	TopP                *float64          `json:"top_p,omitempty"`
	PresencePenalty     *float64          `json:"presence_penalty,omitempty"`
	FrequencyPenalty    *float64          `json:"frequency_penalty,omitempty"`
	ResponseFormat      json.RawMessage   `json:"response_format,omitempty"`
	Tools               json.RawMessage   `json:"tools,omitempty"`
	ToolChoice          json.RawMessage   `json:"tool_choice,omitempty"`
	ParallelToolCalls   *bool             `json:"parallel_tool_calls,omitempty"`
}

type openAIStreamOpts struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

type openAIMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
	Name    string          `json:"name,omitempty"`
}

type openAIModel struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type openAICompletion struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []openAIChoice `json:"choices"`
	Usage   *openAIUsage   `json:"usage,omitempty"`
}

type openAIChoice struct {
	Index        int                   `json:"index"`
	Message      openAIResponseMessage `json:"message"`
	FinishReason string                `json:"finish_reason"`
}

type openAIResponseMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type openAIChunk struct {
	ID      string              `json:"id"`
	Object  string              `json:"object"`
	Created int64               `json:"created"`
	Model   string              `json:"model"`
	Choices []openAIChunkChoice `json:"choices"`
	Usage   *openAIUsage        `json:"usage,omitempty"`
}

type openAIChunkChoice struct {
	Index        int         `json:"index"`
	Delta        openAIDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

type openAIDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

func (s *Server) openAIModels(c *gin.Context) {
	models := make([]openAIModel, 0, len(s.catalog.List()))
	for _, entry := range s.catalog.List() {
		created := int64(0)
		if info, err := os.Stat(entry.Path); err == nil {
			created = info.ModTime().Unix()
		}
		models = append(models, openAIModel{
			ID:      entry.Name,
			Object:  "model",
			Created: created,
			OwnedBy: "mini-ollama",
		})
	}
	c.JSON(http.StatusOK, gin.H{"object": "list", "data": models})
}

func (s *Server) openAIModel(c *gin.Context) {
	name := strings.TrimSpace(c.Param("model"))
	entry, ok := s.catalog.Find(name)
	if !ok {
		openAIError(c, http.StatusNotFound, "model not found", "invalid_request_error", "model", "model_not_found")
		return
	}

	created := int64(0)
	if info, err := os.Stat(entry.Path); err == nil {
		created = info.ModTime().Unix()
	}

	c.JSON(http.StatusOK, openAIModel{
		ID:      entry.Name,
		Object:  "model",
		Created: created,
		OwnedBy: "mini-ollama",
	})
}

func (s *Server) openAIChat(c *gin.Context) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxJSONBody)

	var request openAIChatRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		openAIError(c, http.StatusBadRequest, err.Error(), "invalid_request_error", "request", "invalid_request")
		return
	}

	request.Model = strings.TrimSpace(request.Model)
	if request.Model == "" {
		openAIError(c, http.StatusBadRequest, "model is required", "invalid_request_error", "model", "missing_required_parameter")
		return
	}

	if len(request.Messages) == 0 {
		openAIError(c, http.StatusBadRequest, "messages must contain at least one item", "invalid_request_error", "messages", "invalid_value")
		return
	}

	if request.N == 0 {
		request.N = 1
	}
	if request.N != 1 {
		openAIError(c, http.StatusBadRequest, "only n=1 is supported", "invalid_request_error", "n", "unsupported_parameter")
		return
	}

	if request.MaxTokens != nil && request.MaxCompletionTokens != nil {
		openAIError(c, http.StatusBadRequest, "max_tokens and max_completion_tokens cannot both be set", "invalid_request_error", "max_tokens", "invalid_request")
		return
	}

	maxTokens := 0
	if request.MaxTokens != nil {
		maxTokens = *request.MaxTokens
	}
	if request.MaxCompletionTokens != nil {
		maxTokens = *request.MaxCompletionTokens
	}
	if maxTokens < 0 {
		openAIError(c, http.StatusBadRequest, "max token count must be zero or greater", "invalid_request_error", "max_tokens", "invalid_value")
		return
	}

	if request.Temperature != nil && (*request.Temperature < 0 || *request.Temperature > 2) {
		openAIError(c, http.StatusBadRequest, "temperature must be between 0 and 2", "invalid_request_error", "temperature", "invalid_value")
		return
	}

	if request.StreamOptions != nil && request.StreamOptions.IncludeUsage {
		openAIError(c, http.StatusBadRequest, "stream_options.include_usage is not supported", "invalid_request_error", "stream_options.include_usage", "unsupported_parameter")
		return
	}

	if request.TopP != nil || request.PresencePenalty != nil || request.FrequencyPenalty != nil {
		openAIError(c, http.StatusBadRequest, "top_p and presence/frequency penalties are not supported", "invalid_request_error", "sampling", "unsupported_parameter")
		return
	}

	if len(request.ResponseFormat) != 0 && string(request.ResponseFormat) != "null" {
		openAIError(c, http.StatusBadRequest, "response_format is not supported", "invalid_request_error", "response_format", "unsupported_parameter")
		return
	}

	if len(request.Tools) != 0 && string(request.Tools) != "null" {
		openAIError(c, http.StatusBadRequest, "tools are not supported", "invalid_request_error", "tools", "unsupported_parameter")
		return
	}

	if len(request.ToolChoice) != 0 && string(request.ToolChoice) != "null" {
		openAIError(c, http.StatusBadRequest, "tool_choice is not supported", "invalid_request_error", "tool_choice", "unsupported_parameter")
		return
	}

	if request.ParallelToolCalls != nil {
		openAIError(c, http.StatusBadRequest, "parallel_tool_calls is not supported", "invalid_request_error", "parallel_tool_calls", "unsupported_parameter")
		return
	}

	messages := make([]llama.Message, 0, len(request.Messages))
	for index, item := range request.Messages {
		role := strings.TrimSpace(item.Role)
		if role == "developer" {
			role = "system"
		}

		if role != "system" && role != "user" && role != "assistant" {
			openAIError(c, http.StatusBadRequest,
				fmt.Sprintf("unsupported message role at index %d: %s", index, item.Role),
				"invalid_request_error", "messages", "invalid_value")
			return
		}

		content, err := openAITextContent(item.Content)
		if err != nil {
			openAIError(c, http.StatusBadRequest,
				fmt.Sprintf("invalid message content at index %d: %v", index, err),
				"invalid_request_error", "messages", "invalid_value")
			return
		}

		messages = append(messages, llama.Message{
			Role:    role,
			Content: content,
		})
	}

	stop, err := openAIStop(request.Stop)
	if err != nil {
		openAIError(c, http.StatusBadRequest, err.Error(), "invalid_request_error", "stop", "invalid_value")
		return
	}

	entry, ok := s.catalog.Find(request.Model)
	if !ok {
		openAIError(c, http.StatusNotFound, "model not found", "invalid_request_error", "model", "model_not_found")
		return
	}

	status := s.status()
	if status.Lifecycle != modelservice.LifecycleReady {
		openAIError(c, http.StatusServiceUnavailable, "model service is not ready", "server_error", "model", "service_unavailable")
		return
	}

	if status.Model != entry.Name {
		openAIError(c, http.StatusConflict, "requested model is not loaded", "invalid_request_error", "model", "model_not_loaded")
		return
	}

	backendURL, err := s.backendURL()
	if err != nil {
		openAIError(c, http.StatusServiceUnavailable, err.Error(), "server_error", "model", "service_unavailable")
		return
	}

	s.requests.Add(1)

	upstream := llama.ChatRequest{
		Model:       request.Model,
		Messages:    messages,
		Stream:      request.Stream,
		MaxTokens:   maxTokens,
		Temperature: request.Temperature,
		Stop:        stop,
	}

	completionID := newCompletionID()
	created := time.Now().Unix()

	if !request.Stream {
		response, err := s.llama.Chat(c.Request.Context(), backendURL, upstream, nil)
		if err != nil {
			openAIError(c, http.StatusBadGateway, err.Error(), "upstream_error", "", "backend_error")
			return
		}

		c.JSON(http.StatusOK, openAICompletion{
			ID:      completionID,
			Object:  "chat.completion",
			Created: created,
			Model:   request.Model,
			Choices: []openAIChoice{{
				Index: 0,
				Message: openAIResponseMessage{
					Role:    "assistant",
					Content: response,
				},
				FinishReason: "stop",
			}},
		})
		return
	}

	c.Header("Content-Type", "text/event-stream; charset=utf-8")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")
	c.Writer.WriteHeader(http.StatusOK)

	if err := writeOpenAIChunk(
		c,
		completionID,
		request.Model,
		created,
		openAIChunkChoice{
			Index: 0,
			Delta: openAIDelta{Role: "assistant"},
		},
	); err != nil {
		return
	}

	_, err = s.llama.Chat(c.Request.Context(), backendURL, upstream, func(delta string) error {
		return writeOpenAIChunk(
			c,
			completionID,
			request.Model,
			created,
			openAIChunkChoice{
				Index: 0,
				Delta: openAIDelta{Content: delta},
			},
		)
	})
	if err != nil {
		_ = writeOpenAIStreamError(c, err.Error())
		return
	}

	finishReason := "stop"
	if err := writeOpenAIChunk(
		c,
		completionID,
		request.Model,
		created,
		openAIChunkChoice{
			Index:        0,
			FinishReason: &finishReason,
		},
	); err != nil {
		return
	}

	_, _ = io.WriteString(c.Writer, "data: [DONE]\n\n")
	if flusher, ok := c.Writer.(http.Flusher); ok {
		flusher.Flush()
	}
}

func openAITextContent(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", fmt.Errorf("content must be text")
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		if text == "" || len(text) > maxOpenAIContent {
			return "", fmt.Errorf("content is empty or too large")
		}
		return text, nil
	}

	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}

	if err := json.Unmarshal(raw, &parts); err != nil || len(parts) == 0 {
		return "", fmt.Errorf("content must be a string or text parts")
	}

	var builder strings.Builder
	for _, part := range parts {
		if part.Type != "text" {
			return "", fmt.Errorf("content part type %q is not supported", part.Type)
		}

		builder.WriteString(part.Text)
		if builder.Len() > maxOpenAIContent {
			return "", fmt.Errorf("content is too large")
		}
	}

	if builder.Len() == 0 {
		return "", fmt.Errorf("content is empty")
	}

	return builder.String(), nil
}

func openAIStop(raw json.RawMessage) ([]string, error) {
	raw = json.RawMessage(strings.TrimSpace(string(raw)))
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}

	var value string
	if json.Unmarshal(raw, &value) == nil {
		if value == "" {
			return nil, fmt.Errorf("stop cannot be empty")
		}
		return []string{value}, nil
	}

	var values []string
	if err := json.Unmarshal(raw, &values); err != nil || len(values) == 0 || len(values) > 4 {
		return nil, fmt.Errorf("stop must be a string or an array of at most four strings")
	}

	for _, item := range values {
		if item == "" {
			return nil, fmt.Errorf("stop values cannot be empty")
		}
	}

	return values, nil
}

func writeOpenAIChunk(c *gin.Context, id, model string, created int64, choice openAIChunkChoice) error {
	payload, err := json.Marshal(openAIChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []openAIChunkChoice{choice},
	})
	if err != nil {
		return err
	}

	if _, err := fmt.Fprintf(c.Writer, "data: %s\n\n", payload); err != nil {
		return err
	}

	if flusher, ok := c.Writer.(http.Flusher); ok {
		flusher.Flush()
	}

	return nil
}

func writeOpenAIStreamError(c *gin.Context, message string) error {
	payload, err := json.Marshal(gin.H{
		"error": gin.H{
			"message": message,
			"type":    "upstream_error",
			"code":    "backend_error",
		},
	})
	if err != nil {
		return err
	}

	if _, err := fmt.Fprintf(c.Writer, "data: %s\n\n", payload); err != nil {
		return err
	}

	if flusher, ok := c.Writer.(http.Flusher); ok {
		flusher.Flush()
	}

	return nil
}

func openAIError(c *gin.Context, status int, message, typ, param, code string) {
	value := gin.H{
		"message": message,
		"type":    typ,
	}

	if param != "" {
		value["param"] = param
	}
	if code != "" {
		value["code"] = code
	}

	c.JSON(status, gin.H{"error": value})
}

func newCompletionID() string {
	buffer := make([]byte, 12)
	if _, err := rand.Read(buffer); err == nil {
		return "chatcmpl-" + hex.EncodeToString(buffer)
	}

	return fmt.Sprintf("chatcmpl-%d", completionSequence.Add(1))
}
