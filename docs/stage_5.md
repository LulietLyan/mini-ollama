# 阶段 5：服务稳定化、TUI 与客户端兼容

阶段 5 实现终端 TUI，并引入 Bubble Tea。TUI 只作为 HTTP API 客户端运行，不直接启动或管理 `llama-server`。项目的交互入口为：

```text
CLI / TUI ──────────┐
                    ▼
             mini-ollama serve
                    │
                    ├── Gin 自定义 HTTP API
                    ├── SQLite 会话历史
                    └── 单模型 Controller
                    │
                    ▼
              llama-server
```

本阶段的目标是把已有功能变成可以长期运行和重复验收的服务：修复模型进程生命周期竞态，统一 API 错误和 SSE 行为，保护会话并发写入，允许显式刷新模型目录，实现由 TUI 驱动的模型选择和聊天，并补上不需要加载真实模型的集成测试。OpenAI 兼容 API、多卡调度、远程访问和认证仍然不在本阶段范围内。

本阶段不再只给设计提示。完整文件代码拆分在以下文档中，按文件覆盖输入即可：

- [运行时完整代码](stage_5_runtime_code.md)：`cmd/serve.go`、`cmd/models.go`；
- [API 完整代码](stage_5_api_code.md)：Gin API、HTTP 客户端、SSE 测试；
- [存储和导入完整代码](stage_5_storage_code.md)：SQLite、HF 导入清理及测试；
- [TUI 完整代码](stage_5_tui_code.md)：Bubble Tea TUI、命令注册及测试。

生命周期部分的完整 `service.go` 和 `service_test.go` 直接放在本文第 2 节，避免关键并发代码被拆散。

## 1. 当前基线和必须修复的问题

先在仓库根目录确认基线：

```bash
go test ./...
go test -race ./...
go vet ./...
```

当前测试可以通过，但覆盖不足。进入阶段 5 前要认识以下风险：

1. `internal/service/service.go` 在外部进程启动之后才写入 `starting`，两个并发 `Start` 可能同时通过 `stopped` 检查。
2. `finish` 直接关闭 `done`，启动失败路径和监控 goroutine 可能重复关闭同一个 channel。
3. `Stop` 遇到 `error` 仍然写入 `stopping`，而异常进程已经退出，状态可能卡住。
4. `finish` 没有清空后端地址、进程和取消函数，停止后可能返回旧地址。
5. `serve` 在确认 HTTP 端口绑定成功前就打印 ready；HTTP 绑定失败时，已经启动的 llama-server 可能没有清理。
6. `internal/api/client.go` 没有把 SSE 的 `error` 事件转换成 Go error，也没有要求收到 `done`。
7. 同一个 conversation 的并发聊天会交错追加消息，导致上下文顺序不确定。
8. `Catalog.Refresh` 已存在，但没有通过 CLI 或 API 暴露；阶段 4 发布模型后必须重启 serve 才能看到它。
9. `internal/history/store.go` 的 SQLite 并发参数和 `messages` JSON 字段还没有被专门测试。
10. HF 转换失败后可能留下半成品 `.gguf-stage-*` 目录，需要补清理和取消测试。

## 2. 生命周期安全

文件：`internal/service/service.go`

### 2.1 `Start` 的锁边界

`Start` 必须先在同一把 `mu` 中完成状态预留，再执行外部进程操作。状态预留至少包括：

```text
检查 lifecycle == stopped
创建本次运行的 `serviceRun`，其中包含 done channel、cancel 和 finishOnce
写入 lifecycle = starting
写入 model、startedAt 和空错误
释放 mu
选择端口、创建 context、启动 llama-server
```

这样第二个 `Start` 会立即看到 `starting` 并返回错误。`Stop` 也能在进程等待 health 时拿到本次运行的取消函数。

`done`、`cancel`、`process` 和 `backendURL` 都必须属于一次具体运行。旧运行结束后，下一次成功 `Start` 要重新创建，而不是复用旧 channel。

### 2.2 只允许一次收尾

不要把本次运行的 channel 和 `sync.Once` 直接放在 `Service` 上；它们必须绑定到一次具体运行：

```go
type serviceRun struct {
	done       chan struct{}
	finishOnce sync.Once
	cancel     context.CancelFunc
	process    *runner.Process
	backendURL string
}
```

每次新运行创建新的 `serviceRun`。监控 goroutine、启动失败路径和停止路径都只能调用统一的 `finish(run, err)`，由该运行自己的 `finishOnce.Do` 保护最终清理。这样旧运行的 goroutine 不会关闭新运行的 `done`。

`finish` 的收尾顺序：

```text
加 mu
根据当前状态和 processErr 决定 stopped 或 error
保存错误字符串
取出 done
清空 backendURL、process 和 cancel
释放 mu
只关闭一次 done
```

`finish` 不应该在持有 `mu` 时等待进程，也不应该再次调用 `Stop`。

### 2.3 可直接替换的代码

将 `internal/service/service.go` 整个文件替换为下面代码。它已经包含 import、生命周期实现、参数校验、参数构造、端口选择和 health 就绪检测；不要把下面代码追加到旧文件末尾，否则会出现重复定义：

```go
package service

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"mini-ollama/internal/runner"
)

type Lifecycle string

const (
	LifecycleStopped  Lifecycle = "stopped"
	LifecycleStarting Lifecycle = "starting"
	LifecycleReady    Lifecycle = "ready"
	LifecycleStopping Lifecycle = "stopping"
	LifecycleError    Lifecycle = "error"
)

type Config struct {
	ServerPath  string
	ModelPath   string
	Device      string
	ContextSize int
	GPULayers   string
	Host        string
	ModelName   string
	Port        int
	Stdout      io.Writer
	Stderr      io.Writer
}

type Status struct {
	Lifecycle    Lifecycle
	Model        string
	Error        string
	StartedAt    *time.Time
	RequestCount uint64
}

type serviceRun struct {
	done        chan struct{}
	exited      chan struct{}
	finishOnce  sync.Once
	cancel      context.CancelFunc
	process     *runner.Process
	backendURL  string
	startupErr  error
	terminalErr error
}

// Service manages the single llama-server process owned by serve.
type Service struct {
	mu sync.Mutex

	config Config

	lifecycle Lifecycle
	model     string
	err       string

	startedAt    *time.Time
	requestCount uint64
	run          *serviceRun
}

func New(config Config) *Service {
	return &Service{
		config:    config,
		lifecycle: LifecycleStopped,
	}
}

func (s *Service) Start(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}
	processContext, cancel := context.WithCancel(context.Background())
	run := &serviceRun{
		done:   make(chan struct{}),
		exited: make(chan struct{}),
		cancel: cancel,
	}
	now := time.Now()

	s.mu.Lock()
	if s.lifecycle != LifecycleStopped {
		current := s.lifecycle
		s.mu.Unlock()
		cancel()
		return fmt.Errorf("cannot start service from lifecycle %q", current)
	}
	config := s.config
	model := strings.TrimSpace(config.ModelName)
	if model == "" {
		model = modelName(config.ModelPath)
	}
	s.lifecycle = LifecycleStarting
	s.model = model
	s.err = ""
	s.startedAt = &now
	s.run = run
	s.mu.Unlock()

	if err := validateConfig(config); err != nil {
		cancel()
		s.finish(run, err)
		return err
	}
	if err := ctx.Err(); err != nil {
		cancel()
		s.finish(run, err)
		return err
	}

	host := strings.TrimSpace(config.Host)
	if host == "" {
		host = "127.0.0.1"
	}

	port := config.Port
	if port == 0 {
		var err error
		port, err = selectAvailablePort(host)
		if err != nil {
			cancel()
			s.finish(run, err)
			return err
		}
	}

	backendURL := "http://" + net.JoinHostPort(host, strconv.Itoa(port))
	if err := processContext.Err(); err != nil {
		cancel()
		s.finish(run, err)
		return err
	}

	process, err := runner.Start(processContext, runner.Config{
		Path:   config.ServerPath,
		Args:   buildServerArgs(config.ModelPath, host, port, config),
		Env:    buildProcessEnv(config.Device),
		Stdout: config.Stdout,
		Stderr: config.Stderr,
	})
	if err != nil {
		cancel()
		err = fmt.Errorf("start llama-server: %w", err)
		s.finish(run, err)
		return err
	}

	s.mu.Lock()
	run.process = process
	run.backendURL = backendURL
	current := s.run == run
	s.mu.Unlock()
	go s.monitor(run)
	if !current {
		cancel()
		<-run.done
		return context.Canceled
	}

	readyContext, cancelReady := context.WithCancel(ctx)
	stopReady := context.AfterFunc(processContext, cancelReady)
	err = waitForReady(readyContext, backendURL+"/health", run.exited)
	stopReady()
	cancelReady()
	if err != nil {
		s.mu.Lock()
		if s.run == run {
			run.startupErr = err
		}
		s.mu.Unlock()
		cancel()
		<-run.done
		return err
	}

	s.mu.Lock()
	if s.run != run || s.lifecycle != LifecycleStarting {
		s.mu.Unlock()
		cancel()
		<-run.done
		return context.Canceled
	}
	s.lifecycle = LifecycleReady
	s.mu.Unlock()

	return nil
}

func (s *Service) Stop(ctx context.Context) error {
	s.mu.Lock()

	switch s.lifecycle {
	case LifecycleStopped:
		s.mu.Unlock()
		return nil
	case LifecycleError:
		s.lifecycle = LifecycleStopped
		s.model = ""
		s.err = ""
		s.startedAt = nil
		s.run = nil
		s.mu.Unlock()
		return nil
	}

	run := s.run
	if run == nil {
		s.lifecycle = LifecycleStopped
		s.model = ""
		s.err = ""
		s.startedAt = nil
		s.mu.Unlock()
		return nil
	}

	if s.lifecycle != LifecycleStopping {
		s.lifecycle = LifecycleStopping
	}
	cancel := run.cancel
	done := run.done
	s.mu.Unlock()

	if cancel != nil {
		cancel()
	}

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *Service) Wait(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	run, lifecycle, statusErr := s.run, s.lifecycle, s.err
	s.mu.Unlock()

	if run == nil {
		if lifecycle == LifecycleError && statusErr != "" {
			return errors.New(statusErr)
		}
		return nil
	}

	select {
	case <-run.done:
		s.mu.Lock()
		err := run.terminalErr
		s.mu.Unlock()
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *Service) Status() Status {
	s.mu.Lock()
	defer s.mu.Unlock()

	status := Status{
		Lifecycle:    s.lifecycle,
		Model:        s.model,
		Error:        s.err,
		RequestCount: s.requestCount,
	}

	if s.startedAt != nil {
		startedAt := *s.startedAt
		status.StartedAt = &startedAt
	}

	return status
}

func (s *Service) BackendURL() (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.lifecycle != LifecycleReady || s.run == nil || s.run.backendURL == "" {
		return "", errors.New("backend is not running")
	}

	return s.run.backendURL, nil
}

func (s *Service) monitor(run *serviceRun) {
	err := run.process.Wait()
	close(run.exited)
	if err == nil {
		err = errors.New("llama-server exited unexpectedly")
	} else {
		err = fmt.Errorf("llama-server exited: %w", err)
	}
	s.finish(run, err)
}

func (s *Service) finish(run *serviceRun, processErr error) {
	run.finishOnce.Do(func() {
		s.mu.Lock()
		if s.run == run {
			if run.startupErr != nil {
				processErr = run.startupErr
			}
			stopping := s.lifecycle == LifecycleStopping
			if stopping || processErr == nil {
				s.lifecycle = LifecycleStopped
				s.model = ""
				s.err = ""
				s.startedAt = nil
				run.terminalErr = nil
			} else {
				s.lifecycle = LifecycleError
				s.err = processErr.Error()
				run.terminalErr = processErr
			}

			run.backendURL = ""
			run.process = nil
			run.cancel = nil
			s.run = nil
		} else if processErr != nil {
			run.terminalErr = processErr
		}
		s.mu.Unlock()

		close(run.done)
	})
}

func validateConfig(config Config) error {
	if config.ServerPath == "" {
		return errors.New("server path is empty")
	}

	if config.ModelPath == "" {
		return errors.New("model path is empty")
	}

	if !strings.EqualFold(
		filepath.Ext(config.ModelPath),
		".gguf",
	) {
		return fmt.Errorf(
			"model must be a GGUF file: %s",
			config.ModelPath,
		)
	}

	info, err := os.Stat(config.ModelPath)
	if err != nil {
		return fmt.Errorf("check model: %w", err)
	}

	if !info.Mode().IsRegular() {
		return fmt.Errorf(
			"model path is not a regular file: %s",
			config.ModelPath,
		)
	}

	if config.ContextSize < 0 {
		return errors.New("context size must be zero or greater")
	}

	if config.Port < 0 || config.Port > 65535 {
		return errors.New("port must be between 0 and 65535")
	}

	if strings.Contains(strings.TrimSpace(config.Device), ",") {
		return fmt.Errorf(
			"device must select one CUDA device: %q",
			config.Device,
		)
	}

	return nil
}

func buildServerArgs(
	modelPath string,
	host string,
	port int,
	config Config,
) []string {
	args := []string{
		"-m",
		modelPath,
		"--host",
		host,
		"--port",
		strconv.Itoa(port),
	}

	if config.ContextSize > 0 {
		args = append(
			args,
			"-c",
			strconv.Itoa(config.ContextSize),
		)
	}

	if layers := strings.TrimSpace(config.GPULayers); layers != "" {
		args = append(
			args,
			"--gpu-layers",
			layers,
		)
	}

	return args
}

func buildProcessEnv(device string) []string {
	device = strings.TrimSpace(device)

	if device == "" {
		return nil
	}

	const prefix = "CUDA_VISIBLE_DEVICES="

	env := make([]string, 0, len(os.Environ())+1)

	for _, item := range os.Environ() {
		if strings.HasPrefix(item, prefix) {
			continue
		}

		env = append(env, item)
	}

	return append(env, prefix+device)
}

func selectAvailablePort(host string) (int, error) {
	listener, err := net.Listen(
		"tcp",
		net.JoinHostPort(host, "0"),
	)
	if err != nil {
		return 0, fmt.Errorf(
			"select available backend port: %w",
			err,
		)
	}
	defer listener.Close()

	address, ok := listener.Addr().(*net.TCPAddr)
	if !ok {
		return 0, fmt.Errorf(
			"unexpected listener address: %T",
			listener.Addr(),
		)
	}

	return address.Port, nil
}

func waitForReady(
	ctx context.Context,
	healthURL string,
	processExited <-chan struct{},
) error {
	waitContext, cancel := context.WithTimeout(
		ctx,
		2*time.Minute,
	)
	defer cancel()

	client := &http.Client{
		Timeout: 1 * time.Second,
	}

	var lastErr error

	for {
		select {
		case <-processExited:
			return errors.New(
				"llama-server exited before becoming ready",
			)
		default:
		}

		request, err := http.NewRequestWithContext(
			waitContext,
			http.MethodGet,
			healthURL,
			nil,
		)
		if err != nil {
			return fmt.Errorf(
				"create readiness request: %w",
				err,
			)
		}

		response, err := client.Do(request)
		if err == nil {
			_, _ = io.Copy(io.Discard, response.Body)
			_ = response.Body.Close()

			if response.StatusCode >= 200 &&
				response.StatusCode < 300 {
				return nil
			}

			lastErr = fmt.Errorf(
				"readiness endpoint returned HTTP %d",
				response.StatusCode,
			)
		} else {
			lastErr = err
		}

		timer := time.NewTimer(200 * time.Millisecond)

		select {
		case <-processExited:
			if lastErr != nil {
				return fmt.Errorf(
					"llama-server exited before readiness: %w",
					lastErr,
				)
			}

			return errors.New(
				"llama-server exited before becoming ready",
			)

		case <-waitContext.Done():
			if lastErr != nil {
				return fmt.Errorf(
					"wait for readiness: %w",
					lastErr,
				)
			}

			return waitContext.Err()

		case <-timer.C:
		}
	}
}

func modelName(modelPath string) string {
	filename := filepath.Base(modelPath)
	return strings.TrimSuffix(
		filename,
		filepath.Ext(filename),
	)
}
```

### 2.4 `Stop` 的状态规则

实现下列规则：

```text
stopped  -> 直接成功
starting -> 写 stopping，调用 cancel，等待 done
ready    -> 写 stopping，调用 cancel，等待 done
error    -> 直接清理为 stopped，不等待已经退出的进程
stopping -> 等待同一个 done
```

`Stop` 的等待必须支持调用方 context 取消。停止超时只返回 context error；下一次 `Status` 仍然要能反映真实生命周期，不能伪造为 ready。

### 2.5 Controller 的操作顺序

文件：`internal/control/controller.go`

`Controller` 继续使用一把操作锁串行化模型操作。`Switch` 的语义必须保持：

```text
验证目标模型存在
停止当前服务并确认完成
清空旧服务引用
创建目标模型的 Service
启动目标模型并等待 ready
```

不要在聊天 handler 中调用 `Switch`。聊天只读取 `Status`、当前后端地址和模型目录。由于 `Switch` 已经持有 Controller 的操作锁，内部可以调用 `stopLocked` 和 `startLocked`；不要在持有操作锁时再次调用会获取同一把锁的公开 `Stop`、`Start`，否则会死锁。

阶段 5 的默认语义是：已经开始的 `Start` 会占用 Controller 操作锁直到启动成功或失败；并发 `Stop` 会排队等待。`Service` 自身仍然支持启动阶段取消，下一步如果需要让 API 立即取消排队操作，再单独设计 Controller 的操作句柄。

因此你实际要做的输入顺序只有三步：

1. 用上面的完整代码覆盖 `internal/service/service.go`；
2. 用测试章节中的完整代码覆盖 `internal/service/service_test.go`；
3. `internal/control/controller.go` 当前已经有 `operationMu`，并且 `Switch` 使用 `stopLocked` 后再 `startLocked`，这一节不需要重复改动它。

输入完成后先只验证 service：

```bash
gofmt -w internal/service/service.go internal/service/service_test.go
go test ./internal/service
go test -race ./internal/service
```

这三个命令通过后，再继续阶段 5 后面的 API、SQLite、目录刷新和 TUI 部分。

## 3. serve 的 HTTP 生命周期

文件：`cmd/serve.go`

完整可输入版本见 [stage_5_runtime_code.md](stage_5_runtime_code.md) 的 `cmd/serve.go`，不要只根据下面的片段修改。

不要直接把 `ListenAndServe` 放进 goroutine 后立即打印 ready。改为：

```go
listener, err := net.Listen("tcp", httpServer.Addr)
if err != nil {
	_ = manager.Stop(shutdownContext)
	return fmt.Errorf("bind HTTP server: %w", err)
}

go func() {
	if err := httpServer.Serve(listener); err != nil && err != http.ErrServerClosed {
		serverErrors <- err
	}
}()
```

`http.Server` 至少配置：

```go
ReadHeaderTimeout: 10 * time.Second,
IdleTimeout:       60 * time.Second,
WriteTimeout:      0,
```

流式聊天不能设置很短的 `WriteTimeout`，否则长回复会被服务器主动截断。退出路径使用一个统一的 `defer`：先关闭 HTTP server，再用有限 context 调用 `manager.Stop`。无论 HTTP 绑定失败、serve 收到 SIGINT，还是后端异常，都不能遗留 llama-server 子进程。

## 4. API 契约

阶段 5 保留自定义 API。TUI 使用这些已有路由，不新增独立的后端协议，也不新增 OpenAI 路由：

Gin 服务端、HTTP 客户端和 SSE 测试的完整文件见 [stage_5_api_code.md](stage_5_api_code.md)。

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/v1/health` | mini-ollama HTTP 进程可响应 |
| GET | `/api/v1/models` | 返回当前模型目录快照 |
| POST | `/api/v1/models/refresh` | 显式重新扫描模型目录 |
| GET | `/api/v1/status` | 返回 lifecycle、model、error、started_at、request_count |
| POST | `/api/v1/chat` | 使用 `conversation_id` 聊天，支持 SSE |
| POST | `/api/v1/conversations` | 创建会话 |
| GET | `/api/v1/conversations` | 列出会话 |
| GET | `/api/v1/conversations/:id` | 读取会话和消息 |
| DELETE | `/api/v1/conversations/:id` | 删除会话 |
| POST | `/api/v1/service/stop` | 显式停止模型 |
| POST | `/api/v1/service/switch` | 显式切换模型 |

保留现有 JSON 错误形式，至少保证每个错误响应有一个字符串 `error` 字段；不要把绝对模型路径、后端端口、进程 PID 或 token 放入响应。

### 4.1 聊天输入限制

文件：`internal/api/server.go`

在绑定 JSON 前限制请求体，例如：

```go
c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 1<<20)
```

然后限制模型名、conversation ID 和单条消息长度。长度限制应在常量中集中定义并写测试，不要在多个 handler 中复制数字。

### 4.2 SSE 事件规则

成功的流式响应必须以 `done` 结束：

```text
event: token
data: {"conversation_id":"...","delta":"..."}

event: done
data: {"conversation_id":"..."}

```

上游错误使用：

```text
event: error
data: {"error":"..."}

```

HTTP 状态已经写出后不能再改成 502，所以流中发生错误时必须发送 `error` 事件，并且不能发送 `done`。客户端没有收到 `done` 时必须返回错误，即使 TCP 连接是正常 EOF。

文件：`internal/api/client.go` 和 `internal/llama/client.go`

客户端解析器需要：

1. 处理 `token`、`error`、`done` 三种事件；
2. `error` 事件转为 Go error；
3. `done` 之前 EOF 视为截断错误；
4. `onDelta == nil` 时使用空回调；
5. 回调返回 error 时立即取消读取并把错误返回给调用方。

## 5. TUI 实现

文件：`internal/tui/tui.go`、`internal/tui/tui_test.go`、`cmd/tui.go`、`cmd/cmd.go`

完整代码见 [stage_5_tui_code.md](stage_5_tui_code.md)。本节解释交互数据流，代码不要自行简写。

TUI 采用 Bubble Tea 的单模型循环。状态全部保存在 Bubble Tea model 中，后台 HTTP 请求完成后通过 `tea.Program.Send` 转换为消息，再由 `Update` 修改状态；后台 goroutine 不直接写终端。

当前交互：

```text
↑/↓ 或 j/k       选择模型
Enter             通过 /api/v1/service/switch 加载或切换模型
输入文字 + Enter   创建 conversation 并发送流式聊天请求
Ctrl-S            调用 /api/v1/service/stop
Ctrl-R            重新读取模型列表
Ctrl-C            取消当前请求并退出 TUI
```

模型切换必须调用服务端 `switch`，不能只修改 TUI 内部的模型名称。聊天请求使用已有 `conversation_id` 和 SSE；`token` 事件追加到当前 assistant 消息，`done` 事件结束本轮，`error` 事件显示错误并恢复输入状态。`context.WithCancel` 连接到当前聊天请求，因此退出 TUI 或取消 context 时，上游 HTTP 请求也会停止。

`ollama/` 中的 TUI 只用于理解 Bubble Tea 的状态、事件和渲染组织方式；本项目的实现是独立代码，没有复制或修改官方源码。

## 6. 会话并发和取消

文件：`internal/api/server.go`

会话锁、引用计数和请求体限制已经包含在 [stage_5_api_code.md](stage_5_api_code.md) 的完整 `server.go` 中。

同一个 conversation 同时只能有一个聊天请求。可以在 `Server` 中维护按 conversation ID 分组的互斥锁：

```text
请求进入
取得 conversation lock
读取会话并追加 user 消息
调用 llama-server
追加 assistant 消息
释放 conversation lock
```

锁必须通过 `defer` 释放。不同 conversation 之间不应互相阻塞。

客户端断开时，`c.Request.Context()` 会取消；这个 context 必须原样传递给 llama HTTP 请求，使上游连接也尽快关闭。上游失败时需要明确记录本轮 user 消息的语义。阶段 5 采用“保留 user 消息、没有 assistant 消息”的简单规则，客户端重试时应由客户端决定是否新建消息，不要在服务端自动重复追加。

## 7. 模型目录刷新

文件：`internal/api/server.go`、`cmd/models.go`、`internal/catalog/catalog.go`

API 刷新和 CLI 刷新的完整实现见 [stage_5_api_code.md](stage_5_api_code.md) 与 [stage_5_runtime_code.md](stage_5_runtime_code.md)。

`Catalog.Refresh` 已经存在。阶段 5 增加：

```text
POST /api/v1/models/refresh
```

返回刷新后的模型列表。正在运行的模型不因刷新自动切换或停止；刷新只改变后续模型解析使用的目录快照。

CLI 增加：

```bash
mini-ollama models --refresh --models-dir /path/to/models
```

目录扫描仍然必须拒绝 manifest 路径穿越、外部符号链接、重复模型名和多个 GGUF 无法确定主文件的情况。补充这些情况的单元测试。

## 8. SQLite 可靠性

文件：`internal/history/store.go`

完整 SQLite 文件和测试见 [stage_5_storage_code.md](stage_5_storage_code.md)。

完成以下小修改：

1. `Conversation.Messages` 的 JSON 标签改为 `messages`；
2. 打开数据库后设置单写连接策略和合理的 busy timeout，避免多个 handler 同时写入时立即返回 locked；
3. 所有写操作继续使用事务；
4. `GetConversation`、`DeleteConversation` 和消息追加的 not found 错误分别测试；
5. 数据库迁移保持幂等，不能删除已有会话。

不要把 SQLite 文件放在仓库中。模型、下载缓存和 GGUF 仍然使用 `/rtai_cephfs/liangjm/models/`；SQLite 数据库是小文件，在本机 NFS/CephFS 锁协议不稳定时应放在本地 ext4 路径，例如 `/tmp/mini-ollama-data-liangym`。不要把数据库放到当前已经满的 `/home` 挂载点。

本机 CephFS 是 NFS4.2 且 `local_lock=none`。因此 `internal/history/store.go` 不能强制使用 WAL。你需要手动把 `Open` 和 `migrate` 中的初始化改成下面的原则：

```go
db, err := sql.Open("sqlite", path)
if err != nil {
	return nil, fmt.Errorf("open sqlite database: %w", err)
}
db.SetMaxOpenConns(1)
db.SetMaxIdleConns(1)
```

迁移 SQL 中删除：

```sql
PRAGMA journal_mode = WAL;
```

改为单独执行：

```sql
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;
PRAGMA journal_mode = DELETE;
```

然后再执行 `CREATE TABLE` 和 `CREATE INDEX`。`DELETE` journal 适合这个单进程 SQLite 数据库；即使这样，CephFS 服务端仍可能因配额或 NFS `fsync` 返回 `ENOSPC`，所以数据库继续放本地小文件系统更可靠。现有 CephFS 数据库先保留，不要直接删除；它目前没有会话数据，可以在确认备份后重新指定本地 `--data-dir`。

## 9. Hugging Face 导入清理

文件：`internal/hfimport/import.go`

完整导入器和测试见 [stage_5_storage_code.md](stage_5_storage_code.md)。

阶段 4 的大文件操作继续全部使用 `/rtai_cephfs/liangjm/models/`。阶段 5 只修复失败路径，不改变模型来源和量化参数：

1. 转换失败或 context 取消时，清理当前 `.gguf-stage-*` 目录；
2. 已经完成且有正确 SHA-256 marker 的 artifact 才允许复用；
3. marker 先写临时文件，再使用同目录原子 rename；
4. 发布过程不覆盖已有模型；
5. 清理失败时保留原错误，并在日志中只打印路径，不打印 token。

测试使用小型临时文件模拟转换器失败和取消，不加载真实模型。真实 1B、8B 转换只在 CephFS 工作目录进行。

## 10. 必须新增的测试

新增或补充以下测试；这些测试不需要 CUDA 和真实 GGUF：

```text
internal/service/service_test.go
  - 并发 Start 只有一个成功
  - error 状态 Stop 后回到 stopped
  - finish 不会重复关闭 done
  - 停止后 BackendURL 不返回旧地址

internal/control/controller_test.go
  - Switch 按 stop -> start 顺序执行
  - 未知模型不会停止当前模型
  - 并发操作不会启动两个模型

internal/api/server_test.go
  - health/models/status
  - conversation CRUD
  - chat 非流式成功
  - chat token/done SSE
  - chat error SSE
  - 模型未 ready、模型不匹配和会话不存在
  - 同一 conversation 的并发请求

internal/api/client_test.go
  - token 事件
  - error 事件
  - 缺少 done 的截断流
  - callback 取消

internal/llama/client_test.go
  - 上游 error JSON
  - 缺少 [DONE]
  - 多行 data

internal/catalog/catalog_test.go
  - Refresh
  - 重复模型名
  - manifest 路径穿越
  - 外部符号链接

internal/history/store_test.go
  - messages JSON 字段
  - CRUD not found
  - 并发追加消息

internal/hfimport/import_test.go
  - 转换失败清理 stage 目录
  - context 取消
  - marker 原子写入
```

API 测试使用 `httptest.NewServer` 或 Gin handler，不启动真实 HTTP 端口。llama 测试使用假的 HTTP handler 返回固定 JSON/SSE。进程生命周期测试使用假的可执行程序或小型脚本，不启动真实 llama-server。

下面是 `internal/service/service_test.go` 的完整文件。它包含原有参数测试，以及并发启动、error 停止、finish 幂等和旧后端地址清理测试：

```go
package service

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestNewServiceStatus(t *testing.T) {
	modelService := New(Config{})

	status := modelService.Status()

	if status.Lifecycle != LifecycleStopped {
		t.Fatalf(
			"lifecycle = %q, want %q",
			status.Lifecycle,
			LifecycleStopped,
		)
	}

	if status.Model != "" {
		t.Fatalf("model = %q, want empty", status.Model)
	}

	if status.RequestCount != 0 {
		t.Fatalf(
			"request count = %d, want zero",
			status.RequestCount,
		)
	}
}

func TestModelName(t *testing.T) {
	got := modelName(
		"/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
	)

	want := "Llama-3.2-1B-Instruct-Q4_K_M"

	if got != want {
		t.Fatalf("model name = %q, want %q", got, want)
	}
}

func TestBuildServerArgs(t *testing.T) {
	got := buildServerArgs(
		"/models/model.gguf",
		"127.0.0.1",
		39001,
		Config{
			ContextSize: 4096,
			GPULayers:   "all",
		},
	)

	want := []string{
		"-m",
		"/models/model.gguf",
		"--host",
		"127.0.0.1",
		"--port",
		"39001",
		"-c",
		"4096",
		"--gpu-layers",
		"all",
	}

	if len(got) != len(want) {
		t.Fatalf("args = %#v, want %#v", got, want)
	}

	for index := range want {
		if got[index] != want[index] {
			t.Fatalf(
				"args[%d] = %q, want %q",
				index,
				got[index],
				want[index],
			)
		}
	}
}

func TestValidateConfigRejectsMultipleDevices(t *testing.T) {
	err := validateConfig(Config{
		ServerPath: "/bin/true",
		ModelPath:  "/models/model.gguf",
		Device:     "0,1",
	})

	if err == nil {
		t.Fatal("expected multiple-device validation error")
	}
}

func TestConcurrentStartOnlyOneReservesService(t *testing.T) {
	root := t.TempDir()
	model := filepath.Join(root, "model.gguf")
	if err := os.WriteFile(model, []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	script := filepath.Join(root, "backend.sh")
	if err := os.WriteFile(script, []byte("#!/bin/sh\nsleep 30\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	service := New(Config{ServerPath: script, ModelPath: model, Host: "127.0.0.1"})
	firstDone := make(chan error, 1)
	go func() { firstDone <- service.Start(context.Background()) }()

	deadline := time.Now().Add(2 * time.Second)
	for service.Status().Lifecycle != LifecycleStarting {
		if time.Now().After(deadline) {
			t.Fatal("service did not enter starting")
		}
		time.Sleep(5 * time.Millisecond)
	}

	secondErr := service.Start(context.Background())
	if secondErr == nil || !strings.Contains(secondErr.Error(), "starting") {
		t.Fatalf("second start error = %v", secondErr)
	}
	if err := service.Stop(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := <-firstDone; err == nil {
		t.Fatal("first start unexpectedly succeeded")
	}
	if got := service.Status().Lifecycle; got != LifecycleStopped {
		t.Fatalf("lifecycle = %q, want stopped", got)
	}
}

func TestErrorStopResetsService(t *testing.T) {
	service := New(Config{})
	service.lifecycle = LifecycleError
	service.model = "broken"
	service.err = "backend failed"
	if err := service.Stop(context.Background()); err != nil {
		t.Fatal(err)
	}
	status := service.Status()
	if status.Lifecycle != LifecycleStopped || status.Model != "" || status.Error != "" {
		t.Fatalf("status = %+v", status)
	}
}

func TestFinishIsIdempotentAndClearsBackend(t *testing.T) {
	service := New(Config{})
	run := &serviceRun{done: make(chan struct{}), backendURL: "http://127.0.0.1:1234"}
	service.lifecycle = LifecycleReady
	service.run = run
	service.finish(run, errors.New("backend failed"))
	service.finish(run, errors.New("second failure"))
	select {
	case <-run.done:
	case <-time.After(time.Second):
		t.Fatal("done was not closed")
	}
	if _, err := service.BackendURL(); err == nil {
		t.Fatal("stale backend URL was returned")
	}
	if got := service.Status().Lifecycle; got != LifecycleError {
		t.Fatalf("lifecycle = %q, want error", got)
	}
}
```

## 11. 验证命令

全部手动输入后，在仓库根目录执行：

```bash
gofmt -w \
  cmd/*.go \
  internal/api/*.go \
  internal/catalog/*.go \
  internal/control/*.go \
  internal/hfimport/*.go \
  internal/history/*.go \
  internal/llama/*.go \
  internal/service/*.go

go test ./...
go test -race ./...
go vet ./...

go build -o /tmp/mini-ollama-stage5 .
```

不要把模型或 SQLite 数据写入当前仓库。阶段 5 的模型和转换文件使用 `/rtai_cephfs/liangjm/models/`；SQLite 数据库使用本地 ext4 路径 `/tmp/mini-ollama-data-liangym`，阶段 5 可执行文件也使用 `/tmp` 构建以避免占用 CephFS 配额。

## 12. Linux + CUDA 手工验收

开始推理前先明确使用的模型和路径。最低验收使用一个 1B GGUF 和一个 8B GGUF，例如：

```text
模型目录：/rtai_cephfs/liangjm/models/GGUF
数据目录：/rtai_cephfs/liangjm/models/.mini-ollama-data
服务地址：http://127.0.0.1:11434
```

启动常驻 API，不自动加载模型：

```bash
CUDA_VISIBLE_DEVICES=0 \
/tmp/mini-ollama-stage5 serve \
  --models-dir /rtai_cephfs/liangjm/models/GGUF \
  --data-dir /tmp/mini-ollama-data-liangym \
  --device 0 \
  --gpu-layers all
```

另一个终端按固定顺序验收：

```bash
curl --fail http://127.0.0.1:11434/api/v1/health
curl --fail http://127.0.0.1:11434/api/v1/models
curl --fail -X POST http://127.0.0.1:11434/api/v1/models/refresh
curl --fail http://127.0.0.1:11434/api/v1/status

# 使用实际模型名创建会话并执行一轮 chat
curl --fail -X POST http://127.0.0.1:11434/api/v1/conversations \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODEL_NAME"}'

# 明确停止、检查 stopped，再切换到另一个模型
/tmp/mini-ollama-stage5 stop
/tmp/mini-ollama-stage5 switch OTHER_MODEL_NAME
/tmp/mini-ollama-stage5 models \
  --models-dir /rtai_cephfs/liangjm/models/GGUF

# 在另一个终端启动 TUI；它只连接上面的 API
/tmp/mini-ollama-stage5 tui --server http://127.0.0.1:11434
```

验收记录至少包含：health、models、refresh、status、一次非流式 chat、一次 SSE chat、stop、从 1B 切换到 8B、再次查询 status。模型路径必须是 CephFS 路径；SQLite 数据库使用本地 ext4 路径，不能使用仓库目录中的模型副本。

## 13. 阶段 5 完成标准

- TUI 可以读取模型列表、显式切换模型、停止模型并完成 SSE 聊天；
- 并发 `Start` 不会启动两个 llama-server；
- `Stop`、异常退出和 `Switch` 的生命周期状态可收敛；
- 停止后不会返回旧后端地址；
- HTTP 绑定失败不会遗留 llama-server；
- SSE 错误、截断和客户端取消都能被调用方识别；
- 同一会话的消息不会因并发请求乱序；
- 模型目录可以显式刷新，刷新不会隐式切换当前模型；
- SQLite 会话和消息测试覆盖正常、not found 和并发场景；
- HF 转换失败不会把不可验证的半成品当成可复用 artifact；
- `go test ./...`、`go test -race ./...`、`go vet ./...` 和构建通过；
- Linux + CUDA 下完成 1B、8B、chat、stop 和 switch 手工验收。
