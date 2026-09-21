# mini-ollama 阶段 3：模型目录和模型管理

阶段 3 把模型管理从“传入 GGUF 路径”推进为“使用模型名”。GGUF 路径只在服务端目录映射中存在，CLI 和 HTTP 客户端只使用模型名。

本阶段完成：

- 模型目录扫描；
- 模型目录名作为默认模型名；
- `manifest.json` 覆盖模型名和元数据；
- GGUF 文件存在性、大小和 SHA-256 校验；
- `mini-ollama models` 命令；
- `serve MODEL` 同时接受模型名和兼容的 GGUF 路径；
- `mini-ollama stop`；
- `mini-ollama switch MODEL`；
- HTTP `POST /api/v1/service/stop`；
- HTTP `POST /api/v1/service/switch`；
- 单模型约束和显式切换。

本阶段不实现多模型并行、自动显存分配、下载和转换；项目也不实现终端 TUI。

## 1. 固定模型目录布局

默认目录：

```text
~/.mini-ollama/models/
├── Llama-3.2-1B-Instruct-GGUF/
│   ├── manifest.json
│   └── Llama-3.2-1B-Instruct-Q4_K_M.gguf
└── Qwen3-8B-GGUF/
    ├── manifest.json
    └── Qwen3-8B-Q4_K_M.gguf
```

没有 `manifest.json` 时：

- 子目录名就是模型名；
- 子目录必须包含一个 GGUF 文件；
- 根目录直接放置的 GGUF 使用去掉扩展名的文件名。

manifest 文件格式：

```json
{
  "name": "llama3.2:1b",
  "description": "Llama 3.2 1B Instruct Q4_K_M",
  "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  "size_bytes": 807694464,
  "sha256": "optional-lowercase-sha256"
}
```

`file`、`size_bytes` 和 `sha256` 都是可选字段。填写后，`models --verify` 会验证它们；验证失败的模型不会被启动。

## 2. 替换模型目录实现

文件：`internal/catalog/catalog.go`

```go
package catalog

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

type Manifest struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	File        string `json:"file"`
	SizeBytes   int64  `json:"size_bytes"`
	SHA256      string `json:"sha256"`
}

type Entry struct {
	Name        string
	Path        string
	Description string
	SizeBytes   int64
	SHA256      string
	Manifest    string
}

type Catalog struct {
	mu      sync.RWMutex
	rootDir string
	entries map[string]Entry
}

func New(rootDir string) (*Catalog, error) {
	rootDir = strings.TrimSpace(rootDir)
	if rootDir == "" {
		return nil, errors.New("models directory is empty")
	}
	info, err := os.Stat(rootDir)
	if err != nil {
		return nil, fmt.Errorf("check models directory: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("models path is not a directory: %s", rootDir)
	}

	catalog := &Catalog{rootDir: rootDir, entries: make(map[string]Entry)}
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

func (c *Catalog) List() []Entry {
	c.mu.RLock()
	defer c.mu.RUnlock()
	result := make([]Entry, 0, len(c.entries))
	for _, entry := range c.entries {
		result = append(result, entry)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	return result
}

func (c *Catalog) Find(name string) (Entry, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	entry, ok := c.entries[name]
	return entry, ok
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
	if !info.Mode().IsRegular() || !isGGUF(path) {
		return Entry{}, fmt.Errorf("model path is not a GGUF file: %s", path)
	}

	entry := Entry{
		Name:      nameForPath(c.rootDir, path),
		Path:      path,
		SizeBytes: info.Size(),
	}
	c.mu.Lock()
	c.entries[entry.Name] = entry
	c.mu.Unlock()
	return entry, nil
}

func (c *Catalog) Verify(entry Entry) error {
	info, err := os.Stat(entry.Path)
	if err != nil {
		return fmt.Errorf("check model file: %w", err)
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("model path is not a regular file: %s", entry.Path)
	}
	if entry.SizeBytes > 0 && info.Size() != entry.SizeBytes {
		return fmt.Errorf("model size mismatch for %s", entry.Name)
	}
	if entry.SHA256 == "" {
		return nil
	}

	file, err := os.Open(entry.Path)
	if err != nil {
		return fmt.Errorf("open model for checksum: %w", err)
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return fmt.Errorf("hash model: %w", err)
	}
	got := hex.EncodeToString(hash.Sum(nil))
	if !strings.EqualFold(got, entry.SHA256) {
		return fmt.Errorf("model SHA-256 mismatch for %s", entry.Name)
	}
	return nil
}

func scan(root string) (map[string]Entry, error) {
	items, err := os.ReadDir(root)
	if err != nil {
		return nil, fmt.Errorf("read models directory: %w", err)
	}
	entries := make(map[string]Entry)

	for _, item := range items {
		path := filepath.Join(root, item.Name())
		if item.IsDir() {
			entry, ok, err := scanModelDirectory(path)
			if err != nil { return nil, err }
			if !ok { continue }
			if _, exists := entries[entry.Name]; exists { return nil, fmt.Errorf("duplicate model name: %s", entry.Name) }
			entries[entry.Name] = entry
			continue
		}
		if !item.Type().IsRegular() || !isGGUF(path) { continue }
		info, err := item.Info()
		if err != nil { return nil, err }
		name := strings.TrimSuffix(item.Name(), filepath.Ext(item.Name()))
		entries[name] = Entry{Name: name, Path: path, SizeBytes: info.Size()}
	}
	return entries, nil
}

func scanModelDirectory(directory string) (Entry, bool, error) {
	manifestPath := filepath.Join(directory, "manifest.json")
	manifest, hasManifest, err := readManifest(manifestPath)
	if err != nil { return Entry{}, false, err }

	files, err := os.ReadDir(directory)
	if err != nil { return Entry{}, false, fmt.Errorf("read model directory: %w", err) }
	gguf := make([]string, 0, 1)
	for _, item := range files {
		if !item.IsDir() && item.Type().IsRegular() && isGGUF(item.Name()) {
			gguf = append(gguf, filepath.Join(directory, item.Name()))
		}
	}
	if len(gguf) == 0 { return Entry{}, false, nil }
	if len(gguf) > 1 && (!hasManifest || manifest.File == "") {
		return Entry{}, false, fmt.Errorf("multiple GGUF files without manifest file field: %s", directory)
	}

	file := gguf[0]
	if hasManifest && manifest.File != "" {
		candidate := filepath.Join(directory, manifest.File)
		if filepath.Dir(candidate) != filepath.Clean(directory) || !isGGUF(candidate) {
			return Entry{}, false, fmt.Errorf("manifest file must be a GGUF directly inside %s", directory)
		}
		file = candidate
	}
	info, err := os.Stat(file)
	if err != nil { return Entry{}, false, fmt.Errorf("check model file: %w", err) }

	name := filepath.Base(directory)
	if hasManifest && strings.TrimSpace(manifest.Name) != "" { name = strings.TrimSpace(manifest.Name) }
	entry := Entry{Name: name, Path: file, SizeBytes: info.Size()}
	if hasManifest {
		entry.Description = manifest.Description
		entry.SHA256 = strings.ToLower(strings.TrimSpace(manifest.SHA256))
		entry.Manifest = manifestPath
		if manifest.SizeBytes > 0 { entry.SizeBytes = manifest.SizeBytes }
	}
	return entry, true, nil
}

func readManifest(path string) (Manifest, bool, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) { return Manifest{}, false, nil }
	if err != nil { return Manifest{}, false, fmt.Errorf("read manifest: %w", err) }
	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil { return Manifest{}, false, fmt.Errorf("decode manifest %s: %w", path, err) }
	return manifest, true, nil
}

func isGGUF(path string) bool { return strings.EqualFold(filepath.Ext(path), ".gguf") }

func nameForPath(root, path string) string {
	parent := filepath.Dir(path)
	if filepath.Clean(parent) != filepath.Clean(root) { return filepath.Base(parent) }
	return strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
}
```

文件：`internal/catalog/catalog_test.go`

```go
package catalog

import (
	"os"
	"path/filepath"
	"testing"
)

func TestManifestOverridesModelName(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "directory-name")
	if err := os.Mkdir(directory, 0o755); err != nil { t.Fatal(err) }
	if err := os.WriteFile(filepath.Join(directory, "model.gguf"), []byte("gguf"), 0o644); err != nil { t.Fatal(err) }
	manifest := `{"name":"custom:name","file":"model.gguf","size_bytes":4}`
	if err := os.WriteFile(filepath.Join(directory, "manifest.json"), []byte(manifest), 0o644); err != nil { t.Fatal(err) }

	catalog, err := New(root)
	if err != nil { t.Fatal(err) }
	if _, ok := catalog.Find("custom:name"); !ok { t.Fatal("manifest name was not used") }
}

func TestVerifySize(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "model.gguf")
	if err := os.WriteFile(path, []byte("gguf"), 0o644); err != nil { t.Fatal(err) }
	catalog, err := New(root)
	if err != nil { t.Fatal(err) }
	entry, ok := catalog.Find("model")
	if !ok { t.Fatal("model was not found") }
	entry.SizeBytes++
	if err := catalog.Verify(entry); err == nil { t.Fatal("expected size mismatch") }
}
```

## 3. 模型解析器

文件：`internal/catalog/resolve.go`

这个解析器让 `serve`、`switch`、CLI 和 API 共用同一套模型名解析逻辑。显式传入 `.gguf` 路径仍然保留，便于兼容阶段 1 的命令。

```go
package catalog

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func (c *Catalog) Resolve(value string) (Entry, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return Entry{}, fmt.Errorf("model name is empty")
	}
	if entry, ok := c.Find(value); ok {
		return entry, nil
	}

	path, err := filepath.Abs(value)
	if err != nil {
		return Entry{}, fmt.Errorf("resolve model argument: %w", err)
	}
	root, err := filepath.Abs(c.rootDir)
	if err != nil {
		return Entry{}, fmt.Errorf("resolve models directory: %w", err)
	}
	relative, err := filepath.Rel(root, path)
	if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(os.PathSeparator)) {
		return Entry{}, fmt.Errorf("model path is outside models directory: %s", path)
	}
	if !isGGUF(path) {
		return Entry{}, fmt.Errorf("model must be a GGUF file: %s", path)
	}

	entry, err := c.RegisterPath(path)
	if err != nil {
		return Entry{}, err
	}
	return entry, nil
}

func (c *Catalog) RootDir() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.rootDir
}
```

## 4. 单模型控制器

阶段 2 的 `Service` 管理一个后端进程。阶段 3 增加 `Controller`，负责把模型名称解析为目录条目，并保证显式 `switch` 的顺序是：停止旧模型、确认停止、再启动新模型。

文件：`internal/control/controller.go`

```go
package control

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"

	"mini-ollama/internal/catalog"
	modelservice "mini-ollama/internal/service"
)

type Config struct {
	Catalog     *catalog.Catalog
	ServerPath  string
	Device      string
	ContextSize int
	GPULayers   string
	BackendHost string
	BackendPort int
	Stdout      io.Writer
	Stderr      io.Writer
}

type Controller struct {
	mu      sync.Mutex
	config  Config
	service *modelservice.Service
	entry   catalog.Entry
}

func New(config Config) *Controller {
	return &Controller{config: config}
}

func (c *Controller) Start(ctx context.Context, modelName string) error {
	c.mu.Lock()
	if c.service != nil {
		status := c.service.Status()
		c.mu.Unlock()
		return fmt.Errorf("model service is already %s", status.Lifecycle)
	}
	entry, ok := c.config.Catalog.Find(modelName)
	if !ok {
		c.mu.Unlock()
		return fmt.Errorf("model not found: %s", modelName)
	}
	service := modelservice.New(modelservice.Config{
		ServerPath: c.config.ServerPath,
		ModelPath: entry.Path,
		ModelName: entry.Name,
		Device: c.config.Device,
		ContextSize: c.config.ContextSize,
		GPULayers: c.config.GPULayers,
		Host: c.config.BackendHost,
		Port: c.config.BackendPort,
		Stdout: c.config.Stdout,
		Stderr: c.config.Stderr,
	})
	c.service = service
	c.entry = entry
	c.mu.Unlock()

	if err := service.Start(ctx); err != nil {
		c.mu.Lock()
		c.service = nil
		c.entry = catalog.Entry{}
		c.mu.Unlock()
		return err
	}
	return nil
}

func (c *Controller) Stop(ctx context.Context) error {
	c.mu.Lock()
	service := c.service
	c.mu.Unlock()
	if service == nil {
		return nil
	}
	if err := service.Stop(ctx); err != nil {
		return err
	}
	c.mu.Lock()
	c.service = nil
	c.entry = catalog.Entry{}
	c.mu.Unlock()
	return nil
}

func (c *Controller) Switch(ctx context.Context, modelName string) error {
	c.mu.Lock()
	service := c.service
	current := c.entry.Name
	c.mu.Unlock()
	if current == modelName && service != nil {
		status := service.Status()
		if status.Lifecycle == modelservice.LifecycleStarting || status.Lifecycle == modelservice.LifecycleReady {
			return fmt.Errorf("model is already loaded: %s", modelName)
		}
	}
	if service != nil {
		if status := service.Status(); status.Lifecycle == modelservice.LifecycleStarting || status.Lifecycle == modelservice.LifecycleStopping {
			return fmt.Errorf("model service is busy: %s", status.Lifecycle)
		}
		if err := c.Stop(ctx); err != nil {
			return err
		}
	}
	return c.Start(ctx, modelName)
}

func (c *Controller) Status() modelservice.Status {
	c.mu.Lock()
	service := c.service
	c.mu.Unlock()
	if service == nil {
		return modelservice.Status{Lifecycle: modelservice.LifecycleStopped}
	}
	return service.Status()
}

func (c *Controller) BackendURL() (string, error) {
	c.mu.Lock()
	service := c.service
	c.mu.Unlock()
	if service == nil {
		return "", errors.New("model service is stopped")
	}
	return service.BackendURL()
}

func (c *Controller) CurrentModel() string {
	return c.Status().Model
}
```

`Controller` 的 `Stop` 是幂等的。聊天请求只检查 `Status` 和模型名，不调用 `Switch`，所以不会发生隐式切换。

为防止两个 HTTP 请求同时执行切换，在 `Controller` 中增加独立的操作锁：

```go
type Controller struct {
	mu      sync.Mutex
	operationMu sync.Mutex
	config  Config
	service *modelservice.Service
	entry   catalog.Entry
}
```

`Start`、`Stop` 和 `Switch` 的函数体最外层都必须使用：

```go
c.operationMu.Lock()
defer c.operationMu.Unlock()
```

不要在 `Status` 或 `BackendURL` 中获取 `operationMu`，否则状态查询会被模型加载阻塞。

为了避免 `Switch` 调用公开方法时发生重复加锁，控制器的三个公开方法应替换为下面的形式；原来的方法体分别改名为 `startLocked` 和 `stopLocked`：

```go
func (c *Controller) Start(ctx context.Context, modelName string) error {
	c.operationMu.Lock()
	defer c.operationMu.Unlock()
	return c.startLocked(ctx, modelName)
}

func (c *Controller) Stop(ctx context.Context) error {
	c.operationMu.Lock()
	defer c.operationMu.Unlock()
	return c.stopLocked(ctx)
}

func (c *Controller) Switch(ctx context.Context, modelName string) error {
	c.operationMu.Lock()
	defer c.operationMu.Unlock()
	if _, ok := c.config.Catalog.Find(modelName); !ok {
		return fmt.Errorf("model not found: %s", modelName)
	}

	c.mu.Lock()
	current := c.entry.Name
	service := c.service
	c.mu.Unlock()
	if current == modelName && service != nil {
		return fmt.Errorf("model is already loaded: %s", modelName)
	}
	if service != nil {
		status := service.Status()
		if status.Lifecycle == modelservice.LifecycleStarting || status.Lifecycle == modelservice.LifecycleStopping {
			return fmt.Errorf("model service is busy: %s", status.Lifecycle)
		}
		if err := c.stopLocked(ctx); err != nil {
			return err
		}
	}
	return c.startLocked(ctx, modelName)
}
```

`startLocked` 和 `stopLocked` 使用原来 `Start`、`Stop` 的主体代码，但不再获取 `operationMu`。这样同一时刻只有一个模型转换操作，状态查询仍然可以并发执行。

## 5. models CLI

文件：`cmd/models.go`

```go
package cmd

import (
	"encoding/json"
	"fmt"
	"text/tabwriter"

	"mini-ollama/internal/catalog"

	"github.com/spf13/cobra"
)

func newModelsCommand() *cobra.Command {
	var modelsDir string
	var asJSON bool
	var verify bool

	command := &cobra.Command{
		Use:   "models",
		Short: "List local models",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			if modelsDir == "" {
				var err error
				modelsDir, err = defaultModelsDir()
				if err != nil { return err }
			}
			modelCatalog, err := catalog.New(modelsDir)
			if err != nil { return err }
			entries := modelCatalog.List()
			if verify {
				for _, entry := range entries {
					if err := modelCatalog.Verify(entry); err != nil {
						return err
					}
				}
			}
			if asJSON { return json.NewEncoder(cmd.OutOrStdout()).Encode(entries) }
			writer := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 4, 2, ' ', 0)
			fmt.Fprintln(writer, "NAME\tSIZE_BYTES\tDESCRIPTION")
			for _, entry := range entries {
				fmt.Fprintf(writer, "%s\t%d\t%s\n", entry.Name, entry.SizeBytes, entry.Description)
			}
			return writer.Flush()
		},
	}
	command.Flags().StringVar(&modelsDir, "models-dir", "", "model directory")
	command.Flags().BoolVar(&asJSON, "json", false, "print JSON")
	command.Flags().BoolVar(&verify, "verify", false, "verify manifest size and SHA-256")
	return command
}
```

## 6. stop 和 switch CLI

文件：`cmd/service_control.go`

```go
package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/spf13/cobra"
)

func newStopCommand() *cobra.Command {
	var serverURL string
	command := &cobra.Command{
		Use: "stop",
		Short: "Stop the loaded model",
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return postServiceControl(cmd.Context(), serverURL, "/api/v1/service/stop", nil)
		},
	}
	command.Flags().StringVar(&serverURL, "server", "http://127.0.0.1:11434", "mini-ollama HTTP server URL")
	return command
}

func newSwitchCommand() *cobra.Command {
	var serverURL string
	command := &cobra.Command{
		Use: "switch MODEL",
		Short: "Explicitly switch the loaded model",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			payload, err := json.Marshal(map[string]string{"model": args[0]})
			if err != nil { return err }
			return postServiceControl(cmd.Context(), serverURL, "/api/v1/service/switch", payload)
		},
	}
	command.Flags().StringVar(&serverURL, "server", "http://127.0.0.1:11434", "mini-ollama HTTP server URL")
	return command
}

func postServiceControl(ctx context.Context, baseURL, path string, payload []byte) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(baseURL, "/")+path, bytes.NewReader(payload))
	if err != nil { return err }
	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	if err != nil { return err }
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("service control returned HTTP %d", response.StatusCode)
	}
	return nil
}
```

## 7. API 控制端点

在 `internal/api/server.go` 中加入 `context` import，并加入以下接口和字段：

```go
type Controller interface {
	Stop(context.Context) error
	Switch(context.Context, string) error
}
```

在 `Config` 中加入：

```go
Control Controller
```

在 `Server` 中加入：

```go
control Controller
```

在 `New` 返回值中加入：

```go
control: config.Control,
```

在 `Handler` 中加入路由：

```go
group.POST("/service/stop", s.stopService)
group.POST("/service/switch", s.switchService)
```

在 `internal/api/server.go` 末尾加入：

```go
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

聊天处理器必须继续保留这一条约束：

```go
if status.Model != entry.Name {
	c.JSON(http.StatusConflict, gin.H{"error": "requested model is not loaded"})
	return
}
```

聊天处理器不能调用 `Control.Switch`。模型切换只能来自明确的 `switch` 请求。

## 8. 常驻 serve

阶段 3 的 `serve` 需要在模型停止后继续保留 HTTP API，因此不能再使用“等待后端退出就退出整个 serve”的主循环。将 `cmd/serve.go` 替换为下面的版本：

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
	"mini-ollama/internal/control"
	"mini-ollama/internal/history"

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
		Use: "serve [MODEL]",
		Short: "Start the model manager and HTTP API",
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			model := ""
			if len(args) == 1 { model = args[0] }
			return serveModels(cmd, model, options)
		},
	}
	flags := command.Flags()
	flags.IntVarP(&options.contextSize, "context-size", "c", 0, "context size")
	flags.StringVar(&options.gpuLayers, "gpu-layers", "", "GPU layers")
	flags.StringVar(&options.device, "device", "", "one CUDA device")
	flags.StringVar(&options.backendHost, "backend-host", "127.0.0.1", "llama-server host")
	flags.IntVar(&options.backendPort, "backend-port", 0, "llama-server port")
	flags.StringVar(&options.apiHost, "host", options.apiHost, "mini-ollama API host")
	flags.IntVar(&options.apiPort, "port", options.apiPort, "mini-ollama API port")
	flags.StringVar(&options.modelsDir, "models-dir", "", "model directory")
	flags.StringVar(&options.dataDir, "data-dir", "", "data directory")
	return command
}

func serveModels(cmd *cobra.Command, modelArg string, options serveOptions) error {
	var err error
	if options.modelsDir == "" { options.modelsDir, err = defaultModelsDir(); if err != nil { return err } }
	if options.dataDir == "" { options.dataDir, err = defaultDataDir(); if err != nil { return err } }
	if err := os.MkdirAll(options.modelsDir, 0o755); err != nil { return err }
	if options.apiPort < 1 || options.apiPort > 65535 { return fmt.Errorf("API port must be between 1 and 65535") }
	modelCatalog, err := catalog.New(options.modelsDir)
	if err != nil { return err }
	serverPath, err := llamaServerPath()
	if err != nil { return err }
	store, err := history.Open(filepath.Join(options.dataDir, "mini-ollama.db"))
	if err != nil { return err }
	defer store.Close()

	manager := control.New(control.Config{
		Catalog: modelCatalog,
		ServerPath: serverPath,
		Device: options.device,
		ContextSize: options.contextSize,
		GPULayers: options.gpuLayers,
		BackendHost: options.backendHost,
		BackendPort: options.backendPort,
		Stdout: cmd.OutOrStdout(),
		Stderr: cmd.ErrOrStderr(),
	})
	runContext, stopSignals := signal.NotifyContext(cmd.Context(), os.Interrupt, syscall.SIGTERM)
	defer stopSignals()
	if modelArg != "" {
		entry, err := modelCatalog.Resolve(modelArg)
		if err != nil { return err }
		if err := manager.Start(runContext, entry.Name); err != nil { return err }
	}
	apiServer := api.New(api.Config{
		Catalog: modelCatalog,
		History: store,
		BackendURL: manager.BackendURL,
		Status: manager.Status,
		Control: manager,
	})
	httpServer := &http.Server{Addr: net.JoinHostPort(options.apiHost, strconv.Itoa(options.apiPort)), Handler: apiServer.Handler()}
	serverErrors := make(chan error, 1)
	go func() { err := httpServer.ListenAndServe(); if err != nil && err != http.ErrServerClosed { serverErrors <- err } }()
	fmt.Fprintf(cmd.OutOrStdout(), "mini-ollama API is ready: http://%s\n", httpServer.Addr)

	select {
	case <-runContext.Done():
	case err := <-serverErrors:
		return fmt.Errorf("HTTP server: %w", err)
	}

	shutdownContext, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = httpServer.Shutdown(shutdownContext)
	return manager.Stop(shutdownContext)
}
```

## 9. 注册命令

在 `cmd/cmd.go` 中加入：

```go
rootCmd.AddCommand(
	newVersionCommand(),
	newServeCommand(),
	newModelsCommand(),
	newChatCommand(),
	newStopCommand(),
	newSwitchCommand(),
	newRunCommand(),
)
```

## 10. API 模型元数据

`/api/v1/models` 可以返回有限元数据，但不能返回绝对路径。将 `Entry` 的字段改为带 JSON 标签的形式：

```go
type Entry struct {
	Name        string `json:"name"`
	Path        string `json:"-"`
	Description string `json:"description,omitempty"`
	SizeBytes   int64  `json:"size_bytes"`
	SHA256      string `json:"-"`
	Manifest    string `json:"-"`
}
```

在 `internal/api/server.go` 中，将 models handler 改成：

```go
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
```

## 11. manifest 校验

在 `readManifest` 成功解析后调用 `validateManifest`。将下面函数加入 `internal/catalog/catalog.go`：

```go
func validateManifest(manifest Manifest) error {
	if strings.TrimSpace(manifest.Name) != "" {
		if strings.ContainsAny(manifest.Name, `/\\`) || strings.Contains(manifest.Name, "..") {
			return fmt.Errorf("manifest name is invalid: %s", manifest.Name)
		}
	}
	if manifest.SizeBytes < 0 {
		return errors.New("manifest size_bytes must not be negative")
	}
	if manifest.SHA256 != "" {
		value := strings.TrimSpace(manifest.SHA256)
		if len(value) != sha256.Size*2 {
			return errors.New("manifest sha256 must contain 64 hexadecimal characters")
		}
		if _, err := hex.DecodeString(value); err != nil {
			return fmt.Errorf("manifest sha256 is invalid: %w", err)
		}
	}
	return nil
}
```

`readManifest` 中的解析部分应为：

```go
var manifest Manifest
if err := json.Unmarshal(data, &manifest); err != nil {
	return Manifest{}, false, fmt.Errorf("decode manifest %s: %w", path, err)
}
if err := validateManifest(manifest); err != nil {
	return Manifest{}, false, fmt.Errorf("validate manifest %s: %w", path, err)
}
return manifest, true, nil
```

`manifest.file` 必须限制在模型目录内部。现有 `scanModelDirectory` 中处理 `manifest.File` 的代码替换为：

```go
if hasManifest && manifest.File != "" {
	cleanFile := filepath.Clean(manifest.File)
	if filepath.IsAbs(cleanFile) || cleanFile == "." || cleanFile == ".." || strings.HasPrefix(cleanFile, ".."+string(os.PathSeparator)) {
		return Entry{}, false, fmt.Errorf("manifest file escapes model directory: %s", manifest.File)
	}
	candidate := filepath.Join(directory, cleanFile)
	if filepath.Dir(candidate) != filepath.Clean(directory) || !isGGUF(candidate) {
		return Entry{}, false, fmt.Errorf("manifest file must be a GGUF directly inside %s", directory)
	}
	file = candidate
}
```

## 12. 生命周期安全要求

阶段 3 的 API 允许停止和切换，现有 `Service` 必须满足以下状态规则：

```text
stopped  -> starting -> ready
ready    -> stopping -> stopped
starting -> stopping -> stopped
ready    -> error
error    -> stopped（显式 stop）
```

### 先用日常流程理解

把 `Service` 想成“负责一台 llama-server 的管理员”，状态表示管理员当前正在做什么：

| 状态 | 实际含义 | 此时允许的主要操作 |
| --- | --- | --- |
| `stopped` | 没有正在运行的 llama-server | 可以 `Start` |
| `starting` | 进程已经启动，正在等待 `/health` 返回成功 | 等待启动结果；不能再次 `Start` |
| `ready` | 模型已经加载，可以聊天 | 可以聊天或明确 `Stop`/`Switch` |
| `stopping` | 已经收到停止信号，正在等待进程退出 | 等待退出；不能再次操作 |
| `error` | llama-server 异常退出或启动失败 | 记录错误；显式 `Stop` 或 `Switch` 后恢复 |

例如 `switch Qwen3-8B-GGUF` 不是直接把路径换掉，而是以下顺序：

```text
当前模型 ready
      │
      ▼
停止旧 llama-server
      │
      ▼
确认旧进程已经退出
      │
      ▼
加载 Qwen3-8B-GGUF
      │
      ▼
等待新模型 health=200
      │
      ▼
新模型 ready
```

### 这几条修改分别在防什么

1. “在同一把 mutex 内检查并写入 `starting`”防止两个请求同时通过 `stopped` 检查，启动两份 llama-server。
2. `finishOnce` 防止监控 goroutine和启动失败处理同时关闭同一个 `done` channel，避免 `close of closed channel` 崩溃。
3. `Stop` 遇到 `error` 直接清理，是因为异常进程已经退出，没有进程可以再次等待。
4. 清空 `backendURL` 防止模型已经停止后，聊天请求还拿到一个已经失效的后端地址。
5. `Controller` 是模型级管理员，API 只能通过它切换；聊天请求只能读状态，不能偷偷切换模型。

你当前代码中的对应位置是：`Start` 在文件前半部分，`Stop` 在 `func (s *Service) Stop`，进程退出处理在 `func (s *Service) finish`。这部分先理解流程即可；不要把下面五条描述直接当成五个独立的小函数添加，否则很容易出现重复关闭 channel 或重复加锁。

在现有 `Service` 中执行以下完整修改：

1. `Start` 在同一把 mutex 内检查 `LifecycleStopped` 并立即写入 `LifecycleStarting`，再执行外部进程启动。
2. 增加 `finishOnce sync.Once`，监控 goroutine 和启动失败路径只能关闭 `done` 一次。
3. `Stop` 遇到 `LifecycleError` 时直接清理为 `LifecycleStopped`，不能写入 `LifecycleStopping` 后等待一个已经关闭的流程。
4. `finish` 清空 `backendURL`、`process` 和 `cancel`，避免 stop 后返回旧后端地址。
5. `Controller` 的 mutex 串行化 `Start`、`Stop` 和 `Switch`，API handler 不直接操作 `Service`。

`Controller.Switch` 的固定顺序是：

```go
if err := controller.Stop(ctx); err != nil {
	return err
}
return controller.Start(ctx, targetModel)
```

聊天 handler 只能调用 `Status`、`BackendURL` 和模型查询，不能调用 `Switch`。

## 13. 测试和竞态检查

在 `internal/catalog/catalog_test.go` 中增加：

```go
func TestManifestRejectsPathTraversal(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "demo")
	if err := os.Mkdir(directory, 0o755); err != nil { t.Fatal(err) }
	if err := os.WriteFile(filepath.Join(directory, "model.gguf"), []byte("gguf"), 0o644); err != nil { t.Fatal(err) }
	manifest := `{"file":"../outside.gguf"}`
	if err := os.WriteFile(filepath.Join(directory, "manifest.json"), []byte(manifest), 0o644); err != nil { t.Fatal(err) }
	if _, err := New(root); err == nil { t.Fatal("expected manifest traversal error") }
}
```

文件：`internal/control/controller_test.go`

```go
package control

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"mini-ollama/internal/catalog"
	modelservice "mini-ollama/internal/service"
)

func TestStopWithoutLoadedModelIsIdempotent(t *testing.T) {
	root := t.TempDir()
	catalog, err := catalog.New(root)
	if err != nil { t.Fatal(err) }
	manager := New(Config{Catalog: catalog, ServerPath: "/bin/true"})
	if err := manager.Stop(context.Background()); err != nil { t.Fatal(err) }
	if err := manager.Stop(context.Background()); err != nil { t.Fatal(err) }
	if status := manager.Status(); status.Lifecycle != modelservice.LifecycleStopped { t.Fatalf("status = %q", status.Lifecycle) }
}

func TestSwitchUnknownModelDoesNotStopCurrentModel(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "known")
	if err := os.Mkdir(directory, 0o755); err != nil { t.Fatal(err) }
	if err := os.WriteFile(filepath.Join(directory, "model.gguf"), []byte("gguf"), 0o644); err != nil { t.Fatal(err) }
	catalog, err := catalog.New(root)
	if err != nil { t.Fatal(err) }
	manager := New(Config{Catalog: catalog, ServerPath: "/bin/true"})
	if err := manager.Switch(context.Background(), "missing"); err == nil { t.Fatal("expected unknown model error") }
	if status := manager.Status(); status.Lifecycle != modelservice.LifecycleStopped { t.Fatalf("status = %q", status.Lifecycle) }
}
```

在仓库根目录执行：

```bash
gofmt -w \
  cmd/models.go cmd/service_control.go cmd/serve.go cmd/cmd.go \
  internal/catalog/catalog.go internal/catalog/catalog_test.go \
  internal/catalog/resolve.go internal/control/controller.go \
  internal/api/server.go internal/service/service.go

go test ./...
go test -race ./...
go vet ./...
go build -o /tmp/mini-ollama-stage3 .
```

## 14. 手工验收

先列出模型，确认 API 使用模型目录名：

```bash
/tmp/mini-ollama-stage3 models \
  --models-dir /rtai_cephfs/liangjm/models/GGUF
```

启动常驻 API，不指定初始模型：

```bash
CUDA_VISIBLE_DEVICES=0 /tmp/mini-ollama-stage3 serve \
  --models-dir /rtai_cephfs/liangjm/models/GGUF \
  --data-dir /rtai_cephfs/liangjm/models/.mini-ollama-data \
  --device 0 \
  --gpu-layers all
```

显式加载 1B 模型：

```bash
/tmp/mini-ollama-stage3 switch Llama-3.2-1B-Instruct-GGUF
```

停止模型但保持 API 进程：

```bash
/tmp/mini-ollama-stage3 stop
```

显式切换到 8B 模型：

```bash
/tmp/mini-ollama-stage3 switch Qwen3-8B-GGUF
```

检查状态：

```bash
curl --fail http://127.0.0.1:11434/api/v1/status
curl --fail http://127.0.0.1:11434/api/v1/models
```

当聊天请求中的模型名与当前加载模型不一致时，API 必须返回 HTTP 409，不能自动切换。

## 15. 阶段 3 完成标准

- `models` 能列出本地模型，默认按目录名显示；
- manifest 名称、文件、大小和 SHA-256 能被验证；
- API 和 CLI 不暴露绝对 GGUF 路径；
- `serve` 不带模型时 API 仍然启动；
- `stop` 停止后端但不停止 HTTP API；
- `switch MODEL` 先停止旧模型，再加载新模型；
- 聊天请求不能触发隐式模型切换；
- unknown model 返回 404，模型不匹配返回 409；
- `go test ./...`、`go test -race ./...`、`go vet ./...` 和构建全部通过；
- Linux + CUDA 下至少完成一次 1B 到 8B 的 stop/switch 验证。
