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
	contextSize       int
	gpuLayers         string
	device            string
	splitMode         string
	tensorSplit       string
	autoGPU           bool
	gpuMemoryFraction float64
	memoryReserveMiB  int64
	maxLoadedModels   int
	backendHost       string
	backendPort       int
	apiHost           string
	apiPort           int
	modelsDir         string
	dataDir           string
}

func newServeCommand() *cobra.Command {
	options := serveOptions{apiHost: "127.0.0.1", apiPort: 11434}
	command := &cobra.Command{
		Use:   "serve [MODEL]",
		Short: "Start the model manager and HTTP API",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			model := ""
			if len(args) == 1 {
				model = args[0]
			}
			return serveModels(cmd, model, options)
		},
	}

	flags := command.Flags()

	flags.IntVarP(
		&options.contextSize,
		"context-size",
		"c",
		0,
		"context size",
	)

	flags.StringVar(
		&options.gpuLayers,
		"gpu-layers",
		"",
		"GPU layers",
	)
	flags.StringVar(
		&options.device,
		"device",
		"",
		"CUDA device indexes, for example 0 or 0,1",
	)

	flags.StringVar(
		&options.splitMode,
		"split-mode",
		"",
		"multi-GPU split mode: none, layer, row, or tensor",
	)

	flags.StringVar(&options.tensorSplit,
		"tensor-split",
		"",
		"multi-GPU tensor proportions, for example 3,1",
	)

	flags.BoolVar(
		&options.autoGPU,
		"auto-gpu",
		false,
		"allocate GPUs from free NVIDIA memory",
	)

	flags.Float64Var(
		&options.gpuMemoryFraction,
		"gpu-memory-fraction",
		0.90,
		"maximum fraction of each GPU memory used by auto allocation",
	)

	flags.Int64Var(
		&options.memoryReserveMiB,
		"memory-reserve-mib",
		512,
		"reserved GPU memory for runtime overhead",
	)

	flags.IntVar(
		&options.maxLoadedModels,
		"max-loaded-models",
		1,
		"maximum llama-server processes kept loaded for scheduling",
	)

	flags.StringVar(
		&options.backendHost,
		"backend-host",
		"127.0.0.1",
		"llama-server host",
	)

	flags.IntVar(
		&options.backendPort,
		"backend-port",
		0,
		"llama-server port",
	)

	flags.StringVar(
		&options.apiHost,
		"host",
		options.apiHost,
		"mini-ollama API host",
	)

	flags.IntVar(
		&options.apiPort,
		"port",
		options.apiPort,
		"mini-ollama API port",
	)

	flags.StringVar(
		&options.modelsDir,
		"models-dir",
		"",
		"model directory",
	)

	flags.StringVar(
		&options.dataDir,
		"data-dir",
		"",
		"data directory",
	)

	return command
}

func serveModels(cmd *cobra.Command, modelArg string, options serveOptions) error {
	var err error

	if options.modelsDir == "" {
		options.modelsDir, err = defaultModelsDir()
		if err != nil {
			return err
		}
	}
	if options.dataDir == "" {
		options.dataDir, err = defaultDataDir()
		if err != nil {
			return err
		}
	}
	if err := os.MkdirAll(options.modelsDir, 0o755); err != nil {
		return err
	}
	if options.apiPort < 1 || options.apiPort > 65535 {
		return fmt.Errorf("API port must be between 1 and 65535")
	}

	modelCatalog, err := catalog.New(options.modelsDir)
	if err != nil {
		return err
	}

	serverPath, err := llamaServerPath()
	if err != nil {
		return err
	}

	store, err := history.Open(filepath.Join(options.dataDir, "mini-ollama.db"))
	if err != nil {
		return err
	}
	defer store.Close()

	manager := control.New(control.Config{
		Catalog:           modelCatalog,
		ServerPath:        serverPath,
		Device:            options.device,
		ContextSize:       options.contextSize,
		GPULayers:         options.gpuLayers,
		SplitMode:         options.splitMode,
		TensorSplit:       options.tensorSplit,
		AutoGPU:           options.autoGPU,
		GPUMemoryFraction: options.gpuMemoryFraction,
		MemoryReserveMiB:  options.memoryReserveMiB,
		MaxLoadedModels:   options.maxLoadedModels,
		BackendHost:       options.backendHost,
		BackendPort:       options.backendPort,
		Stdout:            cmd.OutOrStdout(),
		Stderr:            cmd.ErrOrStderr(),
	})

	runContext, stopSignals := signal.NotifyContext(cmd.Context(), os.Interrupt, syscall.SIGTERM)
	defer stopSignals()
	if modelArg != "" {
		entry, err := modelCatalog.Resolve(modelArg)
		if err != nil {
			return err
		}
		if err := manager.Start(runContext, entry.Name); err != nil {
			return err
		}
	}

	apiServer := api.New(api.Config{
		Catalog:        modelCatalog,
		History:        store,
		BackendURL:     manager.BackendURL,
		AcquireBackend: manager.Acquire,
		Status:         manager.Status,
		Statuses:       manager.Statuses,
		Control:        manager,
	})
	httpServer := &http.Server{
		Addr:              net.JoinHostPort(options.apiHost, strconv.Itoa(options.apiPort)),
		Handler:           apiServer.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       60 * time.Second,
		WriteTimeout:      0,
	}
	serverErrors := make(chan error, 1)
	listener, err := net.Listen("tcp", httpServer.Addr)
	if err != nil {
		shutdownContext, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = manager.Stop(shutdownContext)
		return fmt.Errorf("bind HTTP server: %w", err)
	}

	shutdown := func() {
		shutdownContext, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(shutdownContext)
		_ = manager.Stop(shutdownContext)
	}
	defer shutdown()

	go func() {
		err := httpServer.Serve(listener)
		if err != nil && err != http.ErrServerClosed {
			serverErrors <- err
		}
	}()

	fmt.Fprintf(cmd.OutOrStdout(), "mini-ollama API is ready: http://%s\n", httpServer.Addr)

	select {
	case <-runContext.Done():
	case err := <-serverErrors:
		return fmt.Errorf("HTTP server: %w", err)
	}

	return nil
}
