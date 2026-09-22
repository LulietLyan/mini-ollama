package api

import (
	"context"
	"fmt"
	"io"
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

type Controller interface {
	Stop(context.Context) error
	Switch(context.Context, string) error
}

type Config struct {
	Control        Controller
	Catalog        *catalog.Catalog
	History        *history.Store
	BackendURL     func() (string, error)
	AcquireBackend func(context.Context, string) (string, func(), error)
	Status         func() modelservice.Status
	Statuses       func() map[string]modelservice.Status
}

type Server struct {
	control        Controller
	catalog        *catalog.Catalog
	history        *history.Store
	backendURL     func() (string, error)
	acquireBackend func(context.Context, string) (string, func(), error)
	status         func() modelservice.Status
	statuses       func() map[string]modelservice.Status
	llama          llama.Client
	requests       atomic.Uint64
	locks          *conversationLocks
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
		control:        config.Control,
		catalog:        config.Catalog,
		history:        config.History,
		backendURL:     config.BackendURL,
		acquireBackend: config.AcquireBackend,
		status:         config.Status,
		statuses:       config.Statuses,
		locks:          newConversationLocks(),
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
	group.GET("/metrics", s.metricsHandler)

	group.POST("/chat", s.chat)

	group.POST("/conversations", s.createConversation)
	group.GET("/conversations", s.listConversations)
	group.GET("/conversations/:id", s.getConversation)
	group.DELETE("/conversations/:id", s.deleteConversation)

	group.POST("/service/stop", s.stopService)
	group.POST("/service/switch", s.switchService)

	openAI := router.Group("/v1")
	openAI.GET("/models", s.openAIModels)
	openAI.GET("/models/:model", s.openAIModel)
	openAI.POST("/chat/completions", s.openAIChat)

	return router
}

func (s *Server) health(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"status": "ok"}) }

func (s *Server) metricsHandler(c *gin.Context) {
	backendURL, err := s.backendURL()
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
		return
	}
	request, err := http.NewRequestWithContext(c.Request.Context(), http.MethodGet, strings.TrimRight(backendURL, "/")+"/metrics", nil)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 4<<20))
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	contentType := response.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "text/plain; charset=utf-8"
	}
	c.Data(response.StatusCode, contentType, body)
}

func (s *Server) acquire(model string, ctx context.Context) (string, func(), error) {
	if s.acquireBackend != nil {
		return s.acquireBackend(ctx, model)
	}
	status := s.status()
	if status.Lifecycle != modelservice.LifecycleReady {
		return "", nil, fmt.Errorf("model service is not ready: %s", status.Lifecycle)
	}
	entry, ok := s.catalog.Find(model)
	if !ok || status.Model != entry.Name {
		return "", nil, fmt.Errorf("requested model is not loaded")
	}
	url, err := s.backendURL()
	if err != nil {
		return "", nil, err
	}
	return url, func() {}, nil
}

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
	value := gin.H{
		"lifecycle":           status.Lifecycle,
		"model":               status.Model,
		"error":               status.Error,
		"started_at":          status.StartedAt,
		"ready_at":            status.ReadyAt,
		"load_duration_ms":    status.LoadDurationMs,
		"request_count":       s.requests.Load(),
		"device":              status.Device,
		"tensor_split":        status.TensorSplit,
		"auto_gpu":            status.AutoGPU,
		"required_memory_mib": status.RequiredMemoryMiB,
	}
	if s.statuses != nil {
		value["loaded_models"] = s.statuses()
	}
	c.JSON(http.StatusOK, value)
}

type chatRequest struct {
	Model            string   `json:"model"`
	ConversationID   string   `json:"conversation_id"`
	Message          string   `json:"message"`
	Stream           bool     `json:"stream"`
	MaxTokens        int      `json:"max_tokens,omitempty"`
	Temperature      *float64 `json:"temperature,omitempty"`
	TopP             *float64 `json:"top_p,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	Seed             *int     `json:"seed,omitempty"`
	Stop             []string `json:"stop,omitempty"`
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

	_, ok := s.catalog.Find(request.Model)
	if !ok {
		c.JSON(http.StatusNotFound, gin.H{"error": "model not found"})
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

	backendURL, releaseBackend, err := s.acquire(request.Model, c.Request.Context())
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
		return
	}
	defer releaseBackend()

	backendMessages := make([]llama.Message, 0, len(messages))
	for _, message := range messages {
		backendMessages = append(backendMessages, llama.Message{Role: message.Role, Content: message.Content})
	}

	s.requests.Add(1)
	upstream := llama.ChatRequest{
		Model:            request.Model,
		Messages:         backendMessages,
		Stream:           request.Stream,
		MaxTokens:        request.MaxTokens,
		Temperature:      request.Temperature,
		TopP:             request.TopP,
		PresencePenalty:  request.PresencePenalty,
		FrequencyPenalty: request.FrequencyPenalty,
		Seed:             request.Seed,
		Stop:             request.Stop,
	}

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
