package cmd

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"mini-ollama/internal/runner"

	"github.com/spf13/cobra"
)

type runOptions struct {
	contextSize int
	gpuLayers   string
	device      string
	host        string
	port        int
}

func newRunCommand() *cobra.Command {
	options := runOptions{}

	runCmd := &cobra.Command{
		Use:   "run MODEL",
		Short: "Run a local GGUF model",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runModel(cmd, args[0], options)
		},
	}

	flags := runCmd.Flags()

	flags.IntVarP(
		&options.contextSize,
		"context-size",
		"c",
		0,
		"context size passed to llama-server",
	)

	flags.StringVar(
		&options.gpuLayers,
		"gpu-layers",
		"",
		"number of model layers to offload to GPU",
	)

	flags.StringVar(
		&options.device,
		"device",
		"",
		"CUDA device visible to llama-server, for example 0",
	)

	flags.StringVar(
		&options.host,
		"host",
		"",
		"server host address",
	)

	flags.IntVar(
		&options.port,
		"port",
		0,
		"server port",
	)

	return runCmd
}

func runModel(
	cmd *cobra.Command,
	modelArg string,
	options runOptions,
) error {
	modelPath, err := resolveGGUFPath(modelArg)
	if err != nil {
		return err
	}

	if err := validateRunOptions(options); err != nil {
		return err
	}

	if cmd.Name() == "serve" && options.port == 0 {
		options.port, err = selectAvailablePort(options.host)
		if err != nil {
			return err
		}
	}

	serverPath, err := llamaServerPath()
	if err != nil {
		return err
	}

	healthURL, err := buildHealthURL(options)
	if err != nil {
		return err
	}

	serverArgs := buildServerArgs(modelPath, options)

	runContext, stopSignals := signal.NotifyContext(
		cmd.Context(),
		os.Interrupt,
		syscall.SIGTERM,
	)
	defer stopSignals()

	processContext, stopProcess := context.WithCancel(runContext)
	defer stopProcess()

	// 启动一个 CMD 进程，开启 llama.cpp 服务
	process, err := runner.Start(processContext, runner.Config{
		Path:   serverPath,
		Args:   serverArgs,
		Env:    buildProcessEnv(options.device),
		Stdin:  cmd.InOrStdin(),
		Stdout: cmd.OutOrStdout(),
		Stderr: cmd.ErrOrStderr(),
	})
	if err != nil {
		return fmt.Errorf("start llama-server: %w", err)
	}

	// 监听本地 8080 端口
	err = runner.WaitForHTTPReady(
		processContext,
		runner.ReadinessConfig{
			URL:      healthURL,
			Timeout:  2 * time.Minute,
			Interval: 200 * time.Millisecond,
		},
	)
	if err != nil {
		stopProcess()
		_ = process.Wait()

		if runContext.Err() != nil {
			return nil
		}

		return err
	}

	fmt.Fprintln(
		cmd.OutOrStdout(),
		"llama-server is ready:",
		healthURL,
	)

	err = process.Wait()

	if runContext.Err() != nil {
		return nil
	}

	if err != nil {
		return fmt.Errorf("llama-server exited: %w", err)
	}

	return nil
}

func resolveGGUFPath(modelArg string) (string, error) {
	modelPath, err := filepath.Abs(modelArg)
	if err != nil {
		return "", fmt.Errorf("resolve model path: %w", err)
	}

	if !strings.EqualFold(filepath.Ext(modelPath), ".gguf") {
		return "", fmt.Errorf("model must be a GGUF file: %s", modelPath)
	}

	info, err := os.Stat(modelPath)
	if err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("model not found: %s", modelPath)
		}

		return "", fmt.Errorf("check model: %w", err)
	}

	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("model path is not a regular file: %s", modelPath)
	}

	return modelPath, nil
}

func validateRunOptions(options runOptions) error {
	if options.contextSize < 0 {
		return fmt.Errorf("context size must be zero or greater")
	}

	if options.port < 0 || options.port > 65535 {
		return fmt.Errorf("port must be between 0 and 65535")
	}

	if err := validateGPULayers(options.gpuLayers); err != nil {
		return err
	}

	device := strings.TrimSpace(options.device)
	if strings.Contains(device, ",") {
		return fmt.Errorf(
			"device must select one CUDA device: %q",
			device,
		)
	}

	return nil
}

func validateGPULayers(value string) error {
	value = strings.TrimSpace(value)

	if value == "" {
		return nil
	}

	switch strings.ToLower(value) {
	case "auto", "all":
		return nil
	}

	layers, err := strconv.Atoi(value)
	if err != nil || layers < 0 {
		return fmt.Errorf(
			"gpu layers must be a non-negative integer, auto, or all: %q",
			value,
		)
	}

	return nil
}

func buildServerArgs(
	modelPath string,
	options runOptions,
) []string {
	args := []string{
		"-m",
		modelPath,
	}

	if options.contextSize > 0 {
		args = append(
			args,
			"-c",
			strconv.Itoa(options.contextSize),
		)
	}

	if gpuLayers := strings.TrimSpace(options.gpuLayers); gpuLayers != "" {
		args = append(
			args,
			"--gpu-layers",
			gpuLayers,
		)
	}

	if options.host != "" {
		args = append(
			args,
			"--host",
			options.host,
		)
	}

	if options.port != 0 {
		args = append(
			args,
			"--port",
			strconv.Itoa(options.port),
		)
	}

	return args
}

func buildHealthURL(options runOptions) (string, error) {
	host := strings.TrimSpace(options.host)
	if host == "" {
		host = "127.0.0.1"
	}

	if strings.HasSuffix(host, ".sock") {
		return "", fmt.Errorf(
			"HTTP readiness check does not support UNIX socket host: %s",
			host,
		)
	}

	port := options.port
	if port == 0 {
		port = 8080
	}

	return "http://" +
		net.JoinHostPort(host, strconv.Itoa(port)) +
		"/health", nil
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
	host = strings.TrimSpace(host)

	if host == "" {
		host = "127.0.0.1"
	}

	if strings.HasSuffix(host, ".sock") {
		return 0, fmt.Errorf(
			"dynamic TCP port does not support UNIX socket host: %s",
			host,
		)
	}

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
