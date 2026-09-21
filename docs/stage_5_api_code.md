# 阶段 5 API 完整代码

这些文件组成自定义 Gin API 和 CLI/TUI 共用的 HTTP 客户端。`server.go` 包含请求体限制、模型刷新、SSE、会话锁和停止/切换路由。

## 文件：`internal/api/server.go`

```go
package api

import (
	"context"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"

	"mini-ollama/internal/catalog"
	"mini-ollama/internal/history"
	"mini-ollama/internal/llama"
	modelservice "mini-ollama/internal/service"

	"github.com/gin-gonic/gin"
)

type Controller interface {
	Stop(context.Context) error
	Switch(context.Context, string) error
}

type Config struct {
	Control    Controller
	Catalog    *catalog.Catalog
	History    *history.Store
	BackendURL func() (string, error)
	Status     func() modelservice.Status
}

type Server struct {
	control    Controller
	catalog    *catalog.Catalog
	history    *history.Store
	backendURL func() (string, error)
	status     func() modelservice.Status
	llama      llama.Client
	requests   atomic.Uint64
	locks      *conversationLocks
}

const (
	maxJSONBody       = 1 << 20
	maxModelName      = 256
	maxConversationID = 128
	maxMessage        = 64 << 10
)

type conversationLocks struct {
	mu      sync.Mutex
	entries map[string]*conversationLock
}

type conversationLock struct {
	mu   sync.Mutex
	refs int
}

func newConversationLocks() *conversationLocks {
	return &conversationLocks{entries: make(map[string]*conversationLock)}
}

func (locks *conversationLocks) acquire(id string) func() {
	locks.mu.Lock()
	entry := locks.entries[id]
	if entry == nil {
		entry = &conversationLock{}
		locks.entries[id] = entry
	}
	entry.refs++
	locks.mu.Unlock()

	entry.mu.Lock()
	return func() {
		entry.mu.Unlock()
		locks.mu.Lock()
		entry.refs--
		if entry.refs == 0 {
			delete(locks.entries, id)
		}
		locks.mu.Unlock()
	}
}

func New(config Config) *Server {
	return &Server{
		control:    config.Control,
		catalog:    config.Catalog,
		history:    config.History,
		backendURL: config.BackendURL,
		status:     config.Status,
		locks:      newConversationLocks(),
	}
}

func (s *Server) Handler() http.Handler {
	router := gin.New()
	router.Use(gin.Recovery())

	group := router.Group("/api/v1")

	group.GET("/health", s.health)

	group.GET("/models", s.models)
	group.POST("/models/refresh", s.refreshModels)

	group.GET("/status", s.statusHandler)

	group.POST("/chat", s.chat)

	group.POST("/conversations", s.createConversation)
	group.GET("/conversations", s.listConversations)
	group.GET("/conversations/:id", s.getConversation)
	group.DELETE("/conversations/:id", s.deleteConversation)

	group.POST("/service/stop", s.stopService)
	group.POST("/service/switch", s.switchService)

	return router
}

func (s *Server) health(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"status": "ok"}) }

func (s *Server) models(c *gin.Context) {
	result := make([]gin.H, 0)
	for _, entry := range s.catalog.List() {
		result = append(result, gin.H{
			"name":        entry.Name,
			"description": entry.Description,
			"size_bytes":  entry.SizeBytes,
		})
	}
	c.JSON(http.StatusOK, gin.H{"models": result})
}

func (s *Server) refreshModels(c *gin.Context) {
	if err := s.catalog.Refresh(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	s.models(c)
}

func (s *Server) statusHandler(c *gin.Context) {
	status := s.status()
	c.JSON(http.StatusOK, gin.H{
		"lifecycle":     status.Lifecycle,
		"model":         status.Model,
		"error":         status.Error,
		"started_at":    status.StartedAt,
		"request_count": s.requests.Load(),
	})
}

type chatRequest struct {
	Model          string   `json:"model"`
	ConversationID string   `json:"conversation_id"`
	Message        string   `json:"message"`
	Stream         bool     `json:"stream"`
	MaxTokens      int      `json:"max_tokens,omitempty"`
	Temperature    *float64 `json:"temperature,omitempty"`
	Stop           []string `json:"stop,omitempty"`
}

func (s *Server) chat(c *gin.Context) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxJSONBody)
	var request chatRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	request.Model = strings.TrimSpace(request.Model)
	request.ConversationID = strings.TrimSpace(request.ConversationID)
	request.Message = strings.TrimSpace(request.Message)
	if request.Model == "" || request.ConversationID == "" || request.Message == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model, conversation_id and message are required"})
		return
	}
	if len(request.Model) > maxModelName || len(request.ConversationID) > maxConversationID || len(request.Message) > maxMessage {
		c.JSON(http.StatusRequestEntityTooLarge, gin.H{"error": "chat input is too large"})
		return
	}

	entry, ok := s.catalog.Find(request.Model)
	if !ok {
		c.JSON(http.StatusNotFound, gin.H{"error": "model not found"})
		return
	}

	status := s.status()
	if status.Lifecycle != modelservice.LifecycleReady {
		c.JSON(http.StatusConflict, gin.H{"error": "model service is not ready", "lifecycle": status.Lifecycle})
		return
	}
	if status.Model != entry.Name {
		c.JSON(http.StatusConflict, gin.H{"error": "requested model is not loaded"})
		return
	}

	release := s.locks.acquire(request.ConversationID)
	defer release()

	conversation, err := s.history.GetConversation(c.Request.Context(), request.ConversationID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}
	if conversation.Model != request.Model {
		c.JSON(http.StatusConflict, gin.H{"error": "conversation model mismatch"})
		return
	}
	if _, err := s.history.AppendMessage(c.Request.Context(), conversation.ID, "user", request.Message); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	messages, err := s.history.Messages(c.Request.Context(), conversation.ID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	backendURL, err := s.backendURL()
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
		return
	}

	backendMessages := make([]llama.Message, 0, len(messages))
	for _, message := range messages {
		backendMessages = append(backendMessages, llama.Message{Role: message.Role, Content: message.Content})
	}

	s.requests.Add(1)
	upstream := llama.ChatRequest{Model: request.Model, Messages: backendMessages, Stream: request.Stream, MaxTokens: request.MaxTokens, Temperature: request.Temperature, Stop: request.Stop}

	if !request.Stream {
		response, err := s.llama.Chat(c.Request.Context(), backendURL, upstream, nil)
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
			return
		}

		if _, err := s.history.AppendMessage(c.Request.Context(), conversation.ID, "assistant", response); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": err.Error(),
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"conversation_id": conversation.ID,
			"message":         gin.H{"role": "assistant", "content": response},
		})
		return
	}

	c.Header("Content-Type", "text/event-stream; charset=utf-8")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")

	c.Writer.WriteHeader(http.StatusOK)
	c.Writer.Flush()

	var response strings.Builder
	_, err = s.llama.Chat(c.Request.Context(), backendURL, upstream, func(delta string) error {
		response.WriteString(delta)
		c.SSEvent("token", gin.H{"conversation_id": conversation.ID, "delta": delta})
		c.Writer.Flush()
		return nil
	})
	if err != nil {
		c.SSEvent("error", gin.H{"error": err.Error()})
		c.Writer.Flush()
		return
	}
	if _, err := s.history.AppendMessage(c.Request.Context(), conversation.ID, "assistant", response.String()); err != nil {
		c.SSEvent("error", gin.H{"error": err.Error()})
		c.Writer.Flush()
		return
	}
	c.SSEvent("done", gin.H{"conversation_id": conversation.ID})
	c.Writer.Flush()
}

type conversationRequest struct {
	Model string `json:"model"`
}

func (s *Server) createConversation(c *gin.Context) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxJSONBody)
	var request conversationRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	entry, ok := s.catalog.Find(strings.TrimSpace(request.Model))
	if !ok {
		c.JSON(http.StatusNotFound, gin.H{"error": "model not found"})
		return
	}
	if len(strings.TrimSpace(request.Model)) > maxModelName {
		c.JSON(http.StatusRequestEntityTooLarge, gin.H{"error": "model name is too long"})
		return
	}

	conversation, err := s.history.CreateConversation(c.Request.Context(), entry.Name)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, conversation)
}

func (s *Server) listConversations(c *gin.Context) {
	conversations, err := s.history.ListConversations(c.Request.Context())
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"conversations": conversations})
}

func (s *Server) getConversation(c *gin.Context) {
	conversation, err := s.history.GetConversation(c.Request.Context(), c.Param("id"))
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, conversation)
}

func (s *Server) deleteConversation(c *gin.Context) {
	release := s.locks.acquire(c.Param("id"))
	defer release()
	if err := s.history.DeleteConversation(c.Request.Context(), c.Param("id")); err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	c.Status(http.StatusNoContent)
}

type switchRequest struct {
	Model string `json:"model"`
}

func (s *Server) stopService(c *gin.Context) {
	if s.control == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "service control is unavailable"})
		return
	}
	if err := s.control.Stop(c.Request.Context()); err != nil {
		c.JSON(http.StatusConflict, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "stopped"})
}

func (s *Server) switchService(c *gin.Context) {
	if s.control == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "service control is unavailable"})
		return
	}

	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxJSONBody)
	var request switchRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	request.Model = strings.TrimSpace(request.Model)
	if request.Model == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	if _, ok := s.catalog.Find(request.Model); !ok {
		c.JSON(http.StatusNotFound, gin.H{"error": "model not found"})
		return
	}
	if err := s.control.Switch(c.Request.Context(), request.Model); err != nil {
		c.JSON(http.StatusConflict, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "ready", "model": request.Model})
}
```

## 文件：`internal/api/client.go`

```go
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
	Lifecycle    string  `json:"lifecycle"`
	Model        string  `json:"model"`
	Error        string  `json:"error"`
	StartedAt    *string `json:"started_at"`
	RequestCount uint64  `json:"request_count"`
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
```

## 文件：`internal/api/client_test.go`

```go
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
```

## 验证

```bash
gofmt -w internal/api/server.go internal/api/client.go internal/api/client_test.go
go test ./internal/api
go test -race ./internal/api
```
