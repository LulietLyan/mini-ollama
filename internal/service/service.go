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

	"mini-ollama/internal/gpu"
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
	ServerPath        string
	ModelPath         string
	Device            string
	ContextSize       int
	GPULayers         string
	SplitMode         string
	TensorSplit       string
	AutoGPU           bool
	GPUMemoryFraction float64
	GPUProbe          gpu.Probe
	MemoryReserveMiB  int64
	RequiredMemoryMiB int64
	Host              string
	ModelName         string
	Port              int
	Stdout            io.Writer
	Stderr            io.Writer
}

type Status struct {
	Lifecycle         Lifecycle  `json:"lifecycle"`
	Model             string     `json:"model"`
	Error             string     `json:"error"`
	StartedAt         *time.Time `json:"started_at,omitempty"`
	ReadyAt           *time.Time `json:"ready_at,omitempty"`
	LoadDurationMs    int64      `json:"load_duration_ms"`
	RequestCount      uint64     `json:"request_count"`
	Device            string     `json:"device"`
	TensorSplit       string     `json:"tensor_split"`
	RequiredMemoryMiB int64      `json:"required_memory_mib"`
	AutoGPU           bool       `json:"auto_gpu"`
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

// 管理服务对应的单个 llama-server 进程
type Service struct {
	mu sync.Mutex

	config Config

	lifecycle Lifecycle
	model     string
	err       string

	startedAt      *time.Time
	readyAt        *time.Time
	loadDurationMs int64
	requestCount   uint64
	run            *serviceRun
}

func New(config Config) *Service {
	return &Service{
		config:    config,
		lifecycle: LifecycleStopped,
	}
}

// 启动、动态端口和健康检查
// 检查 lifecycle == stopped
// 创建本次运行的 `serviceRun`，其中包含 done channel、cancel 和 finishOnce
// 写入 lifecycle = starting
// 写入 model、startedAt 和空错误
// 释放 mu
// 选择端口、创建 context、启动 llama-server
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
	config, err := resolveRuntimeConfig(ctx, config)
	if err != nil {
		cancel()
		s.finish(run, err)
		return err
	}
	s.mu.Lock()
	if s.run == run {
		s.config = config
	}
	s.mu.Unlock()
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
	readyAt := time.Now()
	s.readyAt = &readyAt
	s.loadDurationMs = readyAt.Sub(now).Milliseconds()
	s.lifecycle = LifecycleReady
	s.mu.Unlock()

	return nil
}

// 取消进程，等待退出
func (s *Service) Stop(ctx context.Context) error {
	s.mu.Lock()

	// stopped  -> 直接成功
	// error    -> 直接清理为 stopped，不等待已经退出的进程
	switch s.lifecycle {
	case LifecycleStopped:
		s.mu.Unlock()
		return nil
	case LifecycleError:
		s.lifecycle = LifecycleStopped
		s.model = ""
		s.err = ""
		s.startedAt = nil
		s.readyAt = nil
		s.loadDurationMs = 0
		s.run = nil
		s.mu.Unlock()
		return nil
	}

	// starting -> 写 stopping，调用 cancel，等待 done
	// ready    -> 写 stopping，调用 cancel，等待 done
	run := s.run
	if run == nil {
		s.lifecycle = LifecycleStopped
		s.model = ""
		s.err = ""
		s.startedAt = nil
		s.readyAt = nil
		s.loadDurationMs = 0
		s.mu.Unlock()
		return nil
	}

	// stopping -> 等待同一个 done
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

// 让 serve 命令在模型异常退出时结束
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
		Lifecycle:         s.lifecycle,
		Model:             s.model,
		Error:             s.err,
		LoadDurationMs:    s.loadDurationMs,
		RequestCount:      s.requestCount,
		Device:            s.config.Device,
		TensorSplit:       s.config.TensorSplit,
		AutoGPU:           s.config.AutoGPU,
		RequiredMemoryMiB: s.config.RequiredMemoryMiB,
	}

	if s.startedAt != nil {
		startedAt := *s.startedAt
		status.StartedAt = &startedAt
	}
	if s.readyAt != nil {
		readyAt := *s.readyAt
		status.ReadyAt = &readyAt
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
				s.readyAt = nil
				s.loadDurationMs = 0
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

	if strings.EqualFold(strings.TrimSpace(config.Device), "auto") && !config.AutoGPU {
		return errors.New("device=auto requires auto GPU allocation")
	}
	if !strings.EqualFold(strings.TrimSpace(config.Device), "auto") {
		if err := validateDeviceList(config.Device); err != nil {
			return err
		}
	}
	if err := validateSplitMode(config.SplitMode); err != nil {
		return err
	}
	if err := validateTensorSplit(config.TensorSplit); err != nil {
		return err
	}

	return nil
}

func validateDeviceList(value string) error {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil
	}
	seen := make(map[int]struct{})
	for _, item := range strings.Split(value, ",") {
		item = strings.TrimSpace(item)
		device, err := strconv.Atoi(item)
		if err != nil || device < 0 {
			return fmt.Errorf("device must be a comma-separated list of non-negative CUDA indexes: %q", value)
		}
		if _, exists := seen[device]; exists {
			return fmt.Errorf("device contains duplicate CUDA index: %d", device)
		}
		seen[device] = struct{}{}
	}
	return nil
}

func validateSplitMode(value string) error {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" {
		return nil
	}
	switch value {
	case "none", "layer", "row", "tensor":
		return nil
	default:
		return fmt.Errorf("split mode must be none, layer, row, or tensor: %q", value)
	}
}

func validateTensorSplit(value string) error {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil
	}
	positive := false
	for _, item := range strings.Split(value, ",") {
		ratio, err := strconv.ParseFloat(strings.TrimSpace(item), 64)
		if err != nil || ratio < 0 {
			return fmt.Errorf("tensor split must be comma-separated non-negative numbers: %q", value)
		}
		if ratio > 0 {
			positive = true
		}
	}
	if !positive {
		return fmt.Errorf("tensor split must contain at least one positive value")
	}
	return nil
}

func resolveRuntimeConfig(ctx context.Context, config Config) (Config, error) {
	if !config.AutoGPU {
		return config, nil
	}
	probe := config.GPUProbe
	if probe == nil {
		probe = gpu.NVIDIAProbe{}
	}
	devices, err := probe.List(ctx)
	if err != nil {
		return config, err
	}
	modelInfo, err := os.Stat(config.ModelPath)
	if err != nil {
		return config, fmt.Errorf("stat model for GPU allocation: %w", err)
	}
	requiredMiB, err := gpu.RequiredMemoryMiB(modelInfo.Size(), config.ContextSize, config.MemoryReserveMiB, config.GPUMemoryFraction)
	if err != nil {
		return config, err
	}

	requested := make([]int, 0)
	deviceValue := strings.TrimSpace(config.Device)
	if deviceValue != "" && !strings.EqualFold(deviceValue, "auto") {
		for _, value := range strings.Split(deviceValue, ",") {
			index, parseErr := strconv.Atoi(strings.TrimSpace(value))
			if parseErr != nil {
				return config, fmt.Errorf("parse requested GPU: %w", parseErr)
			}
			requested = append(requested, index)
		}
	}
	allocation, err := gpu.Allocate(devices, requested, requiredMiB)
	if err != nil {
		return config, err
	}
	config.Device = allocation.DeviceList
	config.RequiredMemoryMiB = allocation.RequiredMiB
	if strings.TrimSpace(config.GPULayers) == "" {
		config.GPULayers = "all"
	}
	if len(allocation.Devices) > 1 {
		if strings.TrimSpace(config.SplitMode) == "" {
			config.SplitMode = "layer"
		}
		if strings.TrimSpace(config.TensorSplit) == "" {
			config.TensorSplit = allocation.TensorSplit
		}
	}
	return config, nil
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

	if splitMode := strings.TrimSpace(config.SplitMode); splitMode != "" {
		args = append(args, "--split-mode", splitMode)
	}
	if tensorSplit := strings.TrimSpace(config.TensorSplit); tensorSplit != "" {
		args = append(args, "--tensor-split", tensorSplit)
	}

	// llama-server exposes Prometheus-compatible timings at /metrics.
	args = append(args, "--metrics")

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
