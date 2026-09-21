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
	mu          sync.Mutex
	operationMu sync.Mutex
	config      Config
	service     *modelservice.Service
	entry       catalog.Entry
}

func New(config Config) *Controller {
	return &Controller{config: config}
}

func (c *Controller) startLocked(ctx context.Context, modelName string) error {
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
		ServerPath:  c.config.ServerPath,
		ModelPath:   entry.Path,
		ModelName:   entry.Name,
		Device:      c.config.Device,
		ContextSize: c.config.ContextSize,
		GPULayers:   c.config.GPULayers,
		Host:        c.config.BackendHost,
		Port:        c.config.BackendPort,
		Stdout:      c.config.Stdout,
		Stderr:      c.config.Stderr,
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

func (c *Controller) stopLocked(ctx context.Context) error {
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
