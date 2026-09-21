# mini-ollama 阶段 2：服务 API、聊天和会话

本文档给出阶段 2 的完整输入材料。Go 源文件仍由你自己手动输入；本次只新增这份 Markdown 文档。

阶段 2 的范围：

- `serve MODEL` 同时拥有 llama-server 和 Gin HTTP API；
- 模型目录扫描，API 只暴露模型名；
- SQLite 持久化会话和消息；
- 调用 llama-server `/v1/chat/completions`；
- 解析上游 SSE，并向外输出自定义 SSE；
- `chat MODEL` 连接已经运行的 serve；
- health、models、status、chat、conversations API。

本阶段不实现 TUI、Hugging Face 下载、多模型并行、远程认证和 OpenAI 兼容入口。

文档中的标题编号是输入顺序。由于代码按文件职责分组，目录扫描、SQLite 和 llama 客户端的基础代码位于本文后半部分；输入时请按 `2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8` 的标题编号执行。

输入顺序：

1. 依赖：`go get` 和 `go mod tidy`。
2. 模型目录：`internal/catalog`。
3. SQLite：`internal/history`。
4. llama SSE：`internal/llama`。
5. API 客户端和 CLI：`internal/api/client.go`、`cmd/chat.go`。
6. Gin API 和 serve：`internal/api/server.go`、`cmd/serve.go`。
7. 注册命令，最后执行格式化、测试和构建。

## 1. 依赖

在仓库根目录执行：

```bash
go get github.com/gin-gonic/gin@v1.10.0 modernc.org/sqlite@latest
go mod tidy
```

## 5. CLI 使用的 API 客户端

### `internal/api/client.go`

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
	ID string `json:"id"`
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
	payload, err := json.Marshal(map[string]any{
		"model": model,
		"conversation_id": conversationID,
		"message": message,
		"stream": true,
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
	flush := func() error {
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
	return flush()
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

### `cmd/paths.go`

```go
package cmd

import (
	"fmt"
	"os"
	"path/filepath"
)

func defaultModelsDir() (string, error) {
	if value := os.Getenv("MINI_OLLAMA_MODELS_DIR"); value != "" {
		return value, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("find user home directory: %w", err)
	}
	return filepath.Join(home, ".mini-ollama", "models"), nil
}

func defaultDataDir() (string, error) {
	if value := os.Getenv("MINI_OLLAMA_DATA_DIR"); value != "" {
		return value, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("find user home directory: %w", err)
	}
	return filepath.Join(home, ".mini-ollama"), nil
}
```

### `cmd/chat.go`

```go
package cmd

import (
	"bufio"
	"fmt"
	"os"
	"os/signal"
	"strings"

	"mini-ollama/internal/api"

	"github.com/spf13/cobra"
)

func newChatCommand() *cobra.Command {
	var serverURL string
	command := &cobra.Command{
		Use: "chat MODEL",
		Short: "Chat with a model managed by mini-ollama serve",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runChat(cmd, args[0], serverURL)
		},
	}
	command.Flags().StringVar(&serverURL, "server", "http://127.0.0.1:11434", "mini-ollama HTTP server URL")
	return command
}

func runChat(cmd *cobra.Command, model, serverURL string) error {
	ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt)
	defer stop()
	client := &api.Client{BaseURL: serverURL}
	conversation, err := client.CreateConversation(ctx, model)
	if err != nil {
		return fmt.Errorf("create conversation: %w", err)
	}
	fmt.Fprintf(cmd.OutOrStdout(), "conversation %s\n", conversation.ID)

	scanner := bufio.NewScanner(cmd.InOrStdin())
	for {
		if _, err := fmt.Fprint(cmd.OutOrStdout(), "> "); err != nil {
			return err
		}
		if !scanner.Scan() {
			break
		}
		message := strings.TrimSpace(scanner.Text())
		if message == "" {
			continue
		}
		if message == "/exit" || message == "/quit" {
			break
		}
		if err := client.ChatStream(ctx, model, conversation.ID, message, func(delta string) error {
			_, err := fmt.Fprint(cmd.OutOrStdout(), delta)
			return err
		}); err != nil {
			return fmt.Errorf("chat request: %w", err)
		}
		fmt.Fprintln(cmd.OutOrStdout())
	}
	return scanner.Err()
}
```

## 6. serve 接入 API

在现有 `internal/service/service.go` 的 `Config` 中加入：

```go
ModelName string
```

并将 `Start` 中的模型名赋值替换为：

```go
s.model = config.ModelName
if s.model == "" {
s.model = modelName(config.ModelPath)
}
```

将 `cmd/serve.go` 完整替换为：

```go
package cmd

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"mini-ollama/internal/api"
	"mini-ollama/internal/catalog"
	"mini-ollama/internal/history"
	modelservice "mini-ollama/internal/service"

	"github.com/spf13/cobra"
)

type serveOptions struct {
	contextSize int
	gpuLayers string
	device string
	backendHost string
	backendPort int
	apiHost string
	apiPort int
	modelsDir string
	dataDir string
}

func newServeCommand() *cobra.Command {
	options := serveOptions{apiHost: "127.0.0.1", apiPort: 11434}
	command := &cobra.Command{
		Use: "serve MODEL",
		Short: "Start the local model service and HTTP API",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return serveModel(cmd, args[0], options)
		},
	}
	flags := command.Flags()
	flags.IntVarP(&options.contextSize, "context-size", "c", 0, "context size")
	flags.StringVar(&options.gpuLayers, "gpu-layers", "", "GPU layers")
	flags.StringVar(&options.device, "device", "", "one CUDA device")
	flags.StringVar(&options.backendHost, "backend-host", "127.0.0.1", "llama-server host")
	flags.IntVar(&options.backendPort, "backend-port", 0, "llama-server port; zero selects a free port")
	flags.StringVar(&options.apiHost, "host", options.apiHost, "mini-ollama API host")
	flags.IntVar(&options.apiPort, "port", options.apiPort, "mini-ollama API port")
	flags.StringVar(&options.modelsDir, "models-dir", "", "model directory")
	flags.StringVar(&options.dataDir, "data-dir", "", "data directory")
	return command
}

func serveModel(cmd *cobra.Command, modelArg string, options serveOptions) error {
	modelPath, err := resolveGGUFPath(modelArg)
	if err != nil { return err }
	if options.modelsDir == "" {
		options.modelsDir, err = defaultModelsDir()
		if err != nil { return err }
	}
	if options.dataDir == "" {
		options.dataDir, err = defaultDataDir()
		if err != nil { return err }
	}
	if options.apiPort < 1 || options.apiPort > 65535 {
		return fmt.Errorf("API port must be between 1 and 65535")
	}
	serverPath, err := llamaServerPath()
	if err != nil { return err }

	modelCatalog, err := catalog.New(options.modelsDir)
	if err != nil { return err }
	entry, err := modelCatalog.RegisterPath(modelPath)
	if err != nil { return err }
	store, err := history.Open(filepath.Join(options.dataDir, "mini-ollama.db"))
	if err != nil { return err }
	defer store.Close()

	runContext, stopSignals := signal.NotifyContext(cmd.Context(), os.Interrupt, syscall.SIGTERM)
	defer stopSignals()
	modelService := modelservice.New(modelservice.Config{
		ServerPath: serverPath,
		ModelPath: modelPath,
		ModelName: entry.Name,
		Device: options.device,
		ContextSize: options.contextSize,
		GPULayers: options.gpuLayers,
		Host: options.backendHost,
		Port: options.backendPort,
		Stdout: cmd.OutOrStdout(),
		Stderr: cmd.ErrOrStderr(),
	})
	if err := modelService.Start(runContext); err != nil { return err }

	apiServer := api.New(api.Config{
		Catalog: modelCatalog,
		History: store,
		BackendURL: modelService.BackendURL,
		Status: modelService.Status,
	})
	httpServer := &http.Server{
		Addr: net.JoinHostPort(options.apiHost, strconv.Itoa(options.apiPort)),
		Handler: apiServer.Handler(),
	}
	serverErrors := make(chan error, 1)
	go func() {
		err := httpServer.ListenAndServe()
		if err != nil && err != http.ErrServerClosed { serverErrors <- err }
	}()
	fmt.Fprintf(cmd.OutOrStdout(), "mini-ollama API is ready: http://%s\n", httpServer.Addr)

	waitErrors := make(chan error, 1)
	go func() { waitErrors <- modelService.Wait(runContext) }()
	var result error
	select {
	case <-runContext.Done():
	case err := <-serverErrors:
		result = fmt.Errorf("HTTP server: %w", err)
	case err := <-waitErrors:
		result = err
	}

	shutdownContext, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = httpServer.Shutdown(shutdownContext)
	stopErr := modelService.Stop(shutdownContext)
	if result != nil { return result }
	if stopErr != nil && runContext.Err() == nil { return stopErr }
	return nil
}
```

## 7. 注册命令

在 `cmd/cmd.go` 的 `rootCmd.AddCommand` 中加入 `newChatCommand()`：

```go
rootCmd.AddCommand(
	newVersionCommand(),
	newServeCommand(),
	newChatCommand(),
	newRunCommand(),
)
```

## 8. 格式化、测试、构建

```bash
gofmt -w \
  cmd/chat.go cmd/cmd.go cmd/paths.go cmd/serve.go \
  internal/api/client.go internal/api/server.go \
  internal/catalog/catalog.go internal/catalog/catalog_test.go \
  internal/history/store.go internal/llama/client.go \
  internal/llama/client_test.go internal/service/service.go
```

```bash
go mod tidy
go test ./...
go vet ./...
go build -o /tmp/mini-ollama-stage2 .
```

## 9. Linux + CUDA 验收

本阶段使用已经下载好的模型：

```text
/rtai_cephfs/liangjm/models/GGUF/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

启动服务：

```bash
CUDA_VISIBLE_DEVICES=0 /tmp/mini-ollama-stage2 serve \
  /rtai_cephfs/liangjm/models/GGUF/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --models-dir /rtai_cephfs/liangjm/models/GGUF \
  --data-dir /tmp/mini-ollama-data-liangym \
  --device 0 \
  --gpu-layers all
```

另一个终端检查 API：

```bash
curl --fail http://127.0.0.1:11434/api/v1/health
curl --fail http://127.0.0.1:11434/api/v1/models
curl --fail http://127.0.0.1:11434/api/v1/status
```

创建会话：

```bash
curl --fail -X POST \
  -H 'Content-Type: application/json' \
  -d '{"model":"Llama-3.2-1B-Instruct-GGUF"}' \
  http://127.0.0.1:11434/api/v1/conversations
```

将返回 JSON 中的 `id` 填入聊天请求：

```bash
curl -N -X POST \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"Llama-3.2-1B-Instruct-GGUF",
    "conversation_id":"替换为返回的会话ID",
    "message":"用一句话介绍你自己",
    "stream":true
  }' \
  http://127.0.0.1:11434/api/v1/chat
```

最后运行 CLI：

```bash
/tmp/mini-ollama-stage2 chat Llama-3.2-1B-Instruct-GGUF
```

## 10. 阶段 2 完成标准

```bash
go test ./...
go vet ./...
go build -o /tmp/mini-ollama-stage2 .
```

并且以下行为成立：

- `serve` 只启动一个 llama-server；
- Gin API 默认只监听 `127.0.0.1:11434`；
- `/api/v1/models` 只返回模型名；
- 会话和消息在 SQLite 中可查询；
- `/api/v1/chat` 输出 `token`、`done`、`error` SSE 事件；
- 上游 `[DONE]`、注释行、空 chunk 和非 2xx 错误都能处理；
- `chat` 连接已有 serve，不会另起 llama-server；
- `Ctrl-C` 能关闭 HTTP 服务并停止后端进程。

## 2. 新增文件

### `internal/catalog/catalog.go`

模型根目录可以直接放 GGUF，也可以使用“模型目录/一个 GGUF”的布局。模型目录名作为 API 模型名。

```go
package catalog

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

type Entry struct {
	Name string
	Path string
}

type Catalog struct {
	mu      sync.RWMutex
	rootDir string
	entries map[string]Entry
}

func New(rootDir string) (*Catalog, error) {
	if strings.TrimSpace(rootDir) == "" {
		return nil, errors.New("models directory is empty")
	}
	info, err := os.Stat(rootDir)
	if err != nil {
		return nil, fmt.Errorf("check models directory: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("models path is not a directory: %s", rootDir)
	}

	catalog := &Catalog{
		rootDir: rootDir,
		entries: make(map[string]Entry),
	}
	if err := catalog.Refresh(); err != nil {
		return nil, err
	}
	return catalog, nil
}

func (c *Catalog) Refresh() error {
	entries, err := scan(c.rootDir)
	if err != nil {
		return err
	}
	c.mu.Lock()
	c.entries = entries
	c.mu.Unlock()
	return nil
}

func (c *Catalog) RegisterPath(path string) (Entry, error) {
	path, err := filepath.Abs(path)
	if err != nil {
		return Entry{}, fmt.Errorf("resolve model path: %w", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		return Entry{}, fmt.Errorf("check model path: %w", err)
	}
	if !info.Mode().IsRegular() ||
		!strings.EqualFold(filepath.Ext(path), ".gguf") {
		return Entry{}, fmt.Errorf("model path is not a GGUF file: %s", path)
	}

	entry := Entry{Name: nameForPath(c.rootDir, path), Path: path}
	c.mu.Lock()
	c.entries[entry.Name] = entry
	c.mu.Unlock()
	return entry, nil
}

func (c *Catalog) List() []Entry {
	c.mu.RLock()
	defer c.mu.RUnlock()
	entries := make([]Entry, 0, len(c.entries))
	for _, entry := range c.entries {
		entries = append(entries, entry)
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Name < entries[j].Name
	})
	return entries
}

func (c *Catalog) Find(name string) (Entry, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	entry, ok := c.entries[name]
	return entry, ok
}

func scan(rootDir string) (map[string]Entry, error) {
	entries := make(map[string]Entry)
	items, err := os.ReadDir(rootDir)
	if err != nil {
		return nil, fmt.Errorf("read models directory: %w", err)
	}

	for _, item := range items {
		path := filepath.Join(rootDir, item.Name())
		if item.IsDir() {
			files, err := ggufFiles(path)
			if err != nil {
				return nil, err
			}
			if len(files) == 0 {
				continue
			}
			if len(files) > 1 {
				return nil, fmt.Errorf(
					"model directory contains multiple GGUF files: %s",
					path,
				)
			}
			if _, exists := entries[item.Name()]; exists {
				return nil, fmt.Errorf("duplicate model name: %s", item.Name())
			}
			entries[item.Name()] = Entry{
				Name: item.Name(),
				Path: files[0],
			}
			continue
		}

		if item.Type().IsRegular() &&
			strings.EqualFold(filepath.Ext(item.Name()), ".gguf") {
				name := strings.TrimSuffix(item.Name(), filepath.Ext(item.Name()))
				entries[name] = Entry{Name: name, Path: path}
		}
	}
	return entries, nil
}

func ggufFiles(directory string) ([]string, error) {
	items, err := os.ReadDir(directory)
	if err != nil {
		return nil, fmt.Errorf("read model directory: %w", err)
	}
	paths := make([]string, 0, 1)
	for _, item := range items {
		if item.IsDir() || !item.Type().IsRegular() {
			continue
		}
		if strings.EqualFold(filepath.Ext(item.Name()), ".gguf") {
			paths = append(paths, filepath.Join(directory, item.Name()))
		}
	}
	sort.Strings(paths)
	return paths, nil
}

func nameForPath(rootDir, path string) string {
	parent := filepath.Dir(path)
	if filepath.Clean(parent) != filepath.Clean(rootDir) {
		return filepath.Base(parent)
	}
	name := filepath.Base(path)
	return strings.TrimSuffix(name, filepath.Ext(name))
}
```

### `internal/catalog/catalog_test.go`

```go
package catalog

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCatalogUsesDirectoryName(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "demo-model")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	model := filepath.Join(directory, "weights.Q4_K_M.gguf")
	if err := os.WriteFile(model, []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}

	catalog, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	entry, ok := catalog.Find("demo-model")
	if !ok || entry.Path != model {
		t.Fatalf("entry = %#v, found = %v", entry, ok)
	}
}
```

### `internal/history/store.go`

```go
package history

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	_ "modernc.org/sqlite"
)

type Message struct {
	ID             int64     `json:"id"`
	ConversationID string    `json:"conversation_id"`
	Role           string    `json:"role"`
	Content        string    `json:"content"`
	CreatedAt      time.Time `json:"created_at"`
}

type Conversation struct {
	ID        string    `json:"id"`
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	Messages  []Message `json:"messages,omitempty"`
}

type Store struct {
	db *sql.DB
}

func Open(path string) (*Store, error) {
	if path == "" {
		return nil, errors.New("database path is empty")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("create data directory: %w", err)
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open sqlite database: %w", err)
	}
	store := &Store{db: db}
	if err := store.migrate(context.Background()); err != nil {
		_ = db.Close()
		return nil, err
	}
	return store, nil
}

func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

func (s *Store) CreateConversation(ctx context.Context, model string) (Conversation, error) {
	id, err := newID()
	if err != nil {
		return Conversation{}, err
	}
	now := time.Now().UTC()
	stamp := now.Format(time.RFC3339Nano)
	_, err = s.db.ExecContext(ctx,
		`INSERT INTO conversations(id, model, created_at, updated_at)
		 VALUES (?, ?, ?, ?)`, id, model, stamp, stamp)
	if err != nil {
		return Conversation{}, fmt.Errorf("create conversation: %w", err)
	}
	return Conversation{ID: id, Model: model, CreatedAt: now, UpdatedAt: now}, nil
}

func (s *Store) ListConversations(ctx context.Context) ([]Conversation, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, model, created_at, updated_at
		 FROM conversations ORDER BY updated_at DESC`)
	if err != nil {
		return nil, fmt.Errorf("list conversations: %w", err)
	}
	defer rows.Close()
	result := make([]Conversation, 0)
	for rows.Next() {
		conversation, err := scanConversation(rows)
		if err != nil {
			return nil, err
		}
		result = append(result, conversation)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read conversations: %w", err)
	}
	return result, nil
}

func (s *Store) GetConversation(ctx context.Context, id string) (Conversation, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, model, created_at, updated_at
		 FROM conversations WHERE id = ?`, id)
	conversation, err := scanConversation(row)
	if errors.Is(err, sql.ErrNoRows) {
		return Conversation{}, fmt.Errorf("conversation not found: %s", id)
	}
	if err != nil {
		return Conversation{}, fmt.Errorf("get conversation: %w", err)
	}
	conversation.Messages, err = s.Messages(ctx, id)
	if err != nil {
		return Conversation{}, err
	}
	return conversation, nil
}

func (s *Store) DeleteConversation(ctx context.Context, id string) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin deletion: %w", err)
	}
	defer tx.Rollback()
	if _, err := tx.ExecContext(ctx,
		`DELETE FROM messages WHERE conversation_id = ?`, id); err != nil {
		return fmt.Errorf("delete messages: %w", err)
	}
	result, err := tx.ExecContext(ctx,
		`DELETE FROM conversations WHERE id = ?`, id)
	if err != nil {
		return fmt.Errorf("delete conversation: %w", err)
	}
	count, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("check deleted conversation: %w", err)
	}
	if count == 0 {
		return fmt.Errorf("conversation not found: %s", id)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit deletion: %w", err)
	}
	return nil
}

func (s *Store) AppendMessage(ctx context.Context, conversationID, role, content string) (Message, error) {
	if role != "system" && role != "user" && role != "assistant" {
		return Message{}, fmt.Errorf("unsupported message role: %s", role)
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return Message{}, fmt.Errorf("begin message append: %w", err)
	}
	defer tx.Rollback()
	now := time.Now().UTC()
	result, err := tx.ExecContext(ctx,
		`INSERT INTO messages(conversation_id, role, content, created_at)
		 VALUES (?, ?, ?, ?)`, conversationID, role, content, now.Format(time.RFC3339Nano))
	if err != nil {
		return Message{}, fmt.Errorf("append message: %w", err)
	}
	id, err := result.LastInsertId()
	if err != nil {
		return Message{}, fmt.Errorf("read message id: %w", err)
	}
	if _, err := tx.ExecContext(ctx,
		`UPDATE conversations SET updated_at = ? WHERE id = ?`, now.Format(time.RFC3339Nano), conversationID); err != nil {
		return Message{}, fmt.Errorf("update conversation: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return Message{}, fmt.Errorf("commit message: %w", err)
	}
	return Message{ID: id, ConversationID: conversationID, Role: role, Content: content, CreatedAt: now}, nil
}

func (s *Store) Messages(ctx context.Context, conversationID string) ([]Message, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, conversation_id, role, content, created_at
		 FROM messages WHERE conversation_id = ? ORDER BY id`, conversationID)
	if err != nil {
		return nil, fmt.Errorf("list messages: %w", err)
	}
	defer rows.Close()
	result := make([]Message, 0)
	for rows.Next() {
		var message Message
		var createdAt string
		if err := rows.Scan(&message.ID, &message.ConversationID, &message.Role, &message.Content, &createdAt); err != nil {
			return nil, fmt.Errorf("scan message: %w", err)
		}
		message.CreatedAt, err = time.Parse(time.RFC3339Nano, createdAt)
		if err != nil {
			return nil, fmt.Errorf("parse message time: %w", err)
		}
		result = append(result, message)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read messages: %w", err)
	}
	return result, nil
}

func (s *Store) migrate(ctx context.Context) error {
	_, err := s.db.ExecContext(ctx, `
		PRAGMA foreign_keys = ON;
		PRAGMA journal_mode = DELETE;
		CREATE TABLE IF NOT EXISTS conversations(
			id TEXT PRIMARY KEY,
			model TEXT NOT NULL,
			created_at TEXT NOT NULL,
			updated_at TEXT NOT NULL
		);
		CREATE TABLE IF NOT EXISTS messages(
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			conversation_id TEXT NOT NULL,
			role TEXT NOT NULL,
			content TEXT NOT NULL,
			created_at TEXT NOT NULL,
			FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
		);
		CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
		ON messages(conversation_id, id);
	`)
	if err != nil {
		return fmt.Errorf("migrate database: %w", err)
	}
	return nil
}

type rowScanner interface { Scan(dest ...any) error }

func scanConversation(row rowScanner) (Conversation, error) {
	var conversation Conversation
	var createdAt, updatedAt string
	if err := row.Scan(&conversation.ID, &conversation.Model, &createdAt, &updatedAt); err != nil {
		return Conversation{}, err
	}
	var err error
	conversation.CreatedAt, err = time.Parse(time.RFC3339Nano, createdAt)
	if err != nil { return Conversation{}, err }
	conversation.UpdatedAt, err = time.Parse(time.RFC3339Nano, updatedAt)
	if err != nil { return Conversation{}, err }
	return conversation, nil
}

func newID() (string, error) {
	buffer := make([]byte, 16)
	if _, err := rand.Read(buffer); err != nil { return "", err }
	return hex.EncodeToString(buffer), nil
}
```

## 3. llama-server SSE 客户端

### `internal/llama/client.go`

这个客户端固定调用 llama-server 的 `/v1/chat/completions`。它会忽略 SSE 注释行和没有 content 的首尾 chunk，识别 `[DONE]`，并把上游错误转换为 Go error。

```go
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
	if onDelta == nil { onDelta = func(string) error { return nil } }
	payload, err := json.Marshal(request)
	if err != nil { return "", fmt.Errorf("encode chat request: %w", err) }

	endpoint := strings.TrimRight(backendURL, "/") + "/v1/chat/completions"
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil { return "", fmt.Errorf("create chat request: %w", err) }
	httpRequest.Header.Set("Content-Type", "application/json")

	client := c.HTTPClient
	if client == nil { client = http.DefaultClient }
	response, err := client.Do(httpRequest)
	if err != nil { return "", fmt.Errorf("call llama-server: %w", err) }
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		return "", fmt.Errorf("llama-server returned HTTP %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}

	if !request.Stream { return readComplete(response.Body) }
	return readSSE(response.Body, onDelta)
}

func readComplete(reader io.Reader) (string, error) {
	var response struct {
		Choices []struct {
			Message struct { Content string `json:"content"` } `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(reader).Decode(&response); err != nil {
		return "", fmt.Errorf("decode chat response: %w", err)
	}
	if len(response.Choices) == 0 { return "", errors.New("chat response contains no choices") }
	return response.Choices[0].Message.Content, nil
}

func readSSE(reader io.Reader, onDelta func(string) error) (string, error) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 4096), 4<<20)
	var data []string
	var response strings.Builder
	finished := false

	flush := func() error {
		if len(data) == 0 { return nil }
		payload := strings.Join(data, "\n")
		data = data[:0]
		if payload == "[DONE]" { finished = true; return nil }

		var chunk struct {
			Error *struct { Message string `json:"message"` } `json:"error,omitempty"`
			Choices []struct {
				Delta struct { Content *string `json:"content"` } `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return fmt.Errorf("decode llama SSE event: %w", err)
		}
		if chunk.Error != nil { return errors.New(chunk.Error.Message) }
		if len(chunk.Choices) == 0 || chunk.Choices[0].Delta.Content == nil { return nil }
		delta := *chunk.Choices[0].Delta.Content
		if delta == "" { return nil }
		if err := onDelta(delta); err != nil { return err }
		response.WriteString(delta)
		return nil
	}

	for scanner.Scan() {
		line := strings.TrimSuffix(scanner.Text(), "\r")
		if line == "" {
			if err := flush(); err != nil { return "", err }
			if finished { break }
			continue
		}
		if strings.HasPrefix(line, ":") { continue }
		if strings.HasPrefix(line, "data:") {
			data = append(data, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if err := scanner.Err(); err != nil { return "", fmt.Errorf("read llama SSE stream: %w", err) }
	if !finished {
		if err := flush(); err != nil { return "", err }
	}
	return response.String(), nil
}
```

### `internal/llama/client_test.go`

```go
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
			": ping\n\n"+
			"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n"+
			"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"+
			"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n"+
			"data: [DONE]\n\n",
		))
	}))
	defer server.Close()

	var callback string
	response, err := (Client{}).Chat(context.Background(), server.URL, ChatRequest{Stream: true}, func(delta string) error {
		callback += delta
		return nil
	})
	if err != nil { t.Fatal(err) }
	if response != "hello world" || callback != response {
		t.Fatalf("response = %q, callback = %q", response, callback)
	}
}
```

## 4. API 服务

### `internal/api/server.go`

```go
package api

import (
	"net/http"
	"strings"
	"sync/atomic"

	"mini-ollama/internal/catalog"
	"mini-ollama/internal/history"
	"mini-ollama/internal/llama"
	modelservice "mini-ollama/internal/service"

	"github.com/gin-gonic/gin"
)

type Config struct {
	Catalog    *catalog.Catalog
	History    *history.Store
	BackendURL func() (string, error)
	Status     func() modelservice.Status
}

type Server struct {
	catalog    *catalog.Catalog
	history    *history.Store
	backendURL func() (string, error)
	status     func() modelservice.Status
	llama      llama.Client
	requests   atomic.Uint64
}

func New(config Config) *Server {
	return &Server{catalog: config.Catalog, history: config.History, backendURL: config.BackendURL, status: config.Status}
}

func (s *Server) Handler() http.Handler {
	router := gin.New()
	router.Use(gin.Recovery())
	group := router.Group("/api/v1")
	group.GET("/health", s.health)
	group.GET("/models", s.models)
	group.GET("/status", s.statusHandler)
	group.POST("/chat", s.chat)
	group.POST("/conversations", s.createConversation)
	group.GET("/conversations", s.listConversations)
	group.GET("/conversations/:id", s.getConversation)
	group.DELETE("/conversations/:id", s.deleteConversation)
	return router
}

func (s *Server) health(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"status": "ok"}) }

func (s *Server) models(c *gin.Context) {
	result := make([]gin.H, 0)
	for _, entry := range s.catalog.List() { result = append(result, gin.H{"name": entry.Name}) }
	c.JSON(http.StatusOK, gin.H{"models": result})
}

func (s *Server) statusHandler(c *gin.Context) {
	status := s.status()
	c.JSON(http.StatusOK, gin.H{
		"lifecycle": status.Lifecycle,
		"model": status.Model,
		"error": status.Error,
		"started_at": status.StartedAt,
		"request_count": s.requests.Load(),
	})
}

type chatRequest struct {
	Model string `json:"model"`
	ConversationID string `json:"conversation_id"`
	Message string `json:"message"`
	Stream bool `json:"stream"`
	MaxTokens int `json:"max_tokens,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
	Stop []string `json:"stop,omitempty"`
}

func (s *Server) chat(c *gin.Context) {
	var request chatRequest
	if err := c.ShouldBindJSON(&request); err != nil { c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()}); return }
	request.Model = strings.TrimSpace(request.Model)
	request.ConversationID = strings.TrimSpace(request.ConversationID)
	request.Message = strings.TrimSpace(request.Message)
	if request.Model == "" || request.ConversationID == "" || request.Message == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model, conversation_id and message are required"})
		return
	}

	entry, ok := s.catalog.Find(request.Model)
	if !ok { c.JSON(http.StatusNotFound, gin.H{"error": "model not found"}); return }
	status := s.status()
	if status.Lifecycle != modelservice.LifecycleReady {
		c.JSON(http.StatusConflict, gin.H{"error": "model service is not ready", "lifecycle": status.Lifecycle})
		return
	}
	if status.Model != entry.Name { c.JSON(http.StatusConflict, gin.H{"error": "requested model is not loaded"}); return }

	conversation, err := s.history.GetConversation(c.Request.Context(), request.ConversationID)
	if err != nil { c.JSON(http.StatusNotFound, gin.H{"error": err.Error()}); return }
	if conversation.Model != request.Model { c.JSON(http.StatusConflict, gin.H{"error": "conversation model mismatch"}); return }
	if _, err := s.history.AppendMessage(c.Request.Context(), conversation.ID, "user", request.Message); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()}); return
	}
	messages, err := s.history.Messages(c.Request.Context(), conversation.ID)
	if err != nil { c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()}); return }
	backendURL, err := s.backendURL()
	if err != nil { c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()}); return }

	backendMessages := make([]llama.Message, 0, len(messages))
	for _, message := range messages { backendMessages = append(backendMessages, llama.Message{Role: message.Role, Content: message.Content}) }
	s.requests.Add(1)
	upstream := llama.ChatRequest{Model: request.Model, Messages: backendMessages, Stream: request.Stream, MaxTokens: request.MaxTokens, Temperature: request.Temperature, Stop: request.Stop}

	if !request.Stream {
		response, err := s.llama.Chat(c.Request.Context(), backendURL, upstream, nil)
		if err != nil { c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()}); return }
		if _, err := s.history.AppendMessage(c.Request.Context(), conversation.ID, "assistant", response); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()}); return
		}
		c.JSON(http.StatusOK, gin.H{"conversation_id": conversation.ID, "message": gin.H{"role": "assistant", "content": response}})
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
	if err != nil { c.SSEvent("error", gin.H{"error": err.Error()}); c.Writer.Flush(); return }
	if _, err := s.history.AppendMessage(c.Request.Context(), conversation.ID, "assistant", response.String()); err != nil {
		c.SSEvent("error", gin.H{"error": err.Error()}); c.Writer.Flush(); return
	}
	c.SSEvent("done", gin.H{"conversation_id": conversation.ID})
	c.Writer.Flush()
}

type conversationRequest struct { Model string `json:"model"` }

func (s *Server) createConversation(c *gin.Context) {
	var request conversationRequest
	if err := c.ShouldBindJSON(&request); err != nil { c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()}); return }
	entry, ok := s.catalog.Find(strings.TrimSpace(request.Model))
	if !ok { c.JSON(http.StatusNotFound, gin.H{"error": "model not found"}); return }
	conversation, err := s.history.CreateConversation(c.Request.Context(), entry.Name)
	if err != nil { c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()}); return }
	c.JSON(http.StatusCreated, conversation)
}

func (s *Server) listConversations(c *gin.Context) {
	conversations, err := s.history.ListConversations(c.Request.Context())
	if err != nil { c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()}); return }
	c.JSON(http.StatusOK, gin.H{"conversations": conversations})
}

func (s *Server) getConversation(c *gin.Context) {
	conversation, err := s.history.GetConversation(c.Request.Context(), c.Param("id"))
	if err != nil { c.JSON(http.StatusNotFound, gin.H{"error": err.Error()}); return }
	c.JSON(http.StatusOK, conversation)
}

func (s *Server) deleteConversation(c *gin.Context) {
	if err := s.history.DeleteConversation(c.Request.Context(), c.Param("id")); err != nil { c.JSON(http.StatusNotFound, gin.H{"error": err.Error()}); return }
	c.Status(http.StatusNoContent)
}
```
