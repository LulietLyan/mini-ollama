package control

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
	"time"

	"mini-ollama/internal/catalog"
	"mini-ollama/internal/gpu"
	modelservice "mini-ollama/internal/service"
)

type Config struct {
	Catalog           *catalog.Catalog
	ServerPath        string
	Device            string
	ContextSize       int
	GPULayers         string
	SplitMode         string
	TensorSplit       string
	AutoGPU           bool
	GPUMemoryFraction float64
	MemoryReserveMiB  int64
	GPUProbe          gpu.Probe
	MaxLoadedModels   int
	BackendHost       string
	BackendPort       int
	Stdout            io.Writer
	Stderr            io.Writer
}

type loadedModel struct {
	entry    catalog.Entry
	service  *modelservice.Service
	lastUsed time.Time
	refs     int
}

type Controller struct {
	mu          sync.Mutex
	operationMu sync.Mutex
	config      Config
	services    map[string]*loadedModel
	active      string
}

func New(config Config) *Controller {
	if config.MaxLoadedModels <= 0 {
		config.MaxLoadedModels = 1
	}
	return &Controller{config: config, services: make(map[string]*loadedModel)}
}

func (c *Controller) serviceConfig(entry catalog.Entry) modelservice.Config {
	return modelservice.Config{
		ServerPath:        c.config.ServerPath,
		ModelPath:         entry.Path,
		ModelName:         entry.Name,
		Device:            c.config.Device,
		ContextSize:       c.config.ContextSize,
		GPULayers:         c.config.GPULayers,
		SplitMode:         c.config.SplitMode,
		TensorSplit:       c.config.TensorSplit,
		AutoGPU:           c.config.AutoGPU,
		GPUMemoryFraction: c.config.GPUMemoryFraction,
		MemoryReserveMiB:  c.config.MemoryReserveMiB,
		GPUProbe:          c.config.GPUProbe,
		Host:              c.config.BackendHost,
		Port:              c.config.BackendPort,
		Stdout:            c.config.Stdout,
		Stderr:            c.config.Stderr,
	}
}

func (c *Controller) startModelLocked(ctx context.Context, modelName string) error {
	entry, ok := c.config.Catalog.Find(modelName)
	if !ok {
		return fmt.Errorf("model not found: %s", modelName)
	}
	c.mu.Lock()
	if _, exists := c.services[entry.Name]; exists {
		c.mu.Unlock()
		return fmt.Errorf("model is already loaded: %s", entry.Name)
	}
	c.mu.Unlock()

	service := modelservice.New(c.serviceConfig(entry))
	if err := service.Start(ctx); err != nil {
		return err
	}
	c.mu.Lock()
	c.services[entry.Name] = &loadedModel{entry: entry, service: service, lastUsed: time.Now()}
	c.active = entry.Name
	c.mu.Unlock()
	return nil
}

func (c *Controller) evictOneLocked(ctx context.Context) error {
	c.mu.Lock()
	var candidate *loadedModel
	for _, item := range c.services {
		if item.refs != 0 {
			continue
		}
		if candidate == nil || item.lastUsed.Before(candidate.lastUsed) {
			candidate = item
		}
	}
	if candidate == nil {
		c.mu.Unlock()
		return errors.New("all loaded models are busy")
	}
	name := candidate.entry.Name
	service := candidate.service
	c.mu.Unlock()
	if err := service.Stop(ctx); err != nil {
		return fmt.Errorf("evict model %s: %w", name, err)
	}
	c.mu.Lock()
	delete(c.services, name)
	if c.active == name {
		c.active = ""
	}
	c.mu.Unlock()
	return nil
}

// Acquire ensures that a model is loaded, possibly evicting the least recently
// used idle model. The returned release function must be called after inference.
func (c *Controller) Acquire(ctx context.Context, modelName string) (string, func(), error) {
	modelName = strings.TrimSpace(modelName)
	if modelName == "" {
		return "", nil, errors.New("model name is empty")
	}
	c.operationMu.Lock()
	defer c.operationMu.Unlock()
	entry, ok := c.config.Catalog.Find(modelName)
	if !ok {
		return "", nil, fmt.Errorf("model not found: %s", modelName)
	}
	c.mu.Lock()
	item := c.services[entry.Name]
	if item != nil {
		item.refs++
		item.lastUsed = time.Now()
		c.active = entry.Name
		service := item.service
		c.mu.Unlock()
		url, err := service.BackendURL()
		if err != nil {
			c.release(entry.Name)
			return "", nil, err
		}
		return url, c.releaseFunc(entry.Name), nil
	}
	loadedCount := len(c.services)
	c.mu.Unlock()
	if loadedCount >= c.config.MaxLoadedModels {
		if err := c.evictOneLocked(ctx); err != nil {
			return "", nil, err
		}
	}
	if err := c.startModelLocked(ctx, entry.Name); err != nil {
		return "", nil, err
	}
	c.mu.Lock()
	item = c.services[entry.Name]
	item.refs++
	item.lastUsed = time.Now()
	service := item.service
	c.mu.Unlock()
	url, err := service.BackendURL()
	if err != nil {
		c.release(entry.Name)
		return "", nil, err
	}
	return url, c.releaseFunc(entry.Name), nil
}

func (c *Controller) releaseFunc(modelName string) func() {
	var once sync.Once
	return func() { once.Do(func() { c.release(modelName) }) }
}

func (c *Controller) release(modelName string) {
	c.mu.Lock()
	if item := c.services[modelName]; item != nil && item.refs > 0 {
		item.refs--
		item.lastUsed = time.Now()
	}
	c.mu.Unlock()
}

func (c *Controller) Start(ctx context.Context, modelName string) error {
	c.operationMu.Lock()
	defer c.operationMu.Unlock()
	c.mu.Lock()
	if len(c.services) >= c.config.MaxLoadedModels {
		c.mu.Unlock()
		return fmt.Errorf("maximum loaded model count reached: %d", c.config.MaxLoadedModels)
	}
	c.mu.Unlock()
	return c.startModelLocked(ctx, modelName)
}

func (c *Controller) Stop(ctx context.Context) error {
	c.operationMu.Lock()
	defer c.operationMu.Unlock()
	c.mu.Lock()
	items := make([]*loadedModel, 0, len(c.services))
	for _, item := range c.services {
		items = append(items, item)
	}
	c.mu.Unlock()
	for _, item := range items {
		if err := item.service.Stop(ctx); err != nil {
			return err
		}
	}
	c.mu.Lock()
	c.services = make(map[string]*loadedModel)
	c.active = ""
	c.mu.Unlock()
	return nil
}

func (c *Controller) Switch(ctx context.Context, modelName string) error {
	modelName = strings.TrimSpace(modelName)
	if _, ok := c.config.Catalog.Find(modelName); !ok {
		return fmt.Errorf("model not found: %s", modelName)
	}
	c.operationMu.Lock()
	defer c.operationMu.Unlock()
	if err := c.stopAllLocked(ctx); err != nil {
		return err
	}
	return c.startModelLocked(ctx, modelName)
}

func (c *Controller) stopAllLocked(ctx context.Context) error {
	c.mu.Lock()
	items := make([]*loadedModel, 0, len(c.services))
	for _, item := range c.services {
		items = append(items, item)
	}
	c.mu.Unlock()
	for _, item := range items {
		if err := item.service.Stop(ctx); err != nil {
			return err
		}
	}
	c.mu.Lock()
	c.services = make(map[string]*loadedModel)
	c.active = ""
	c.mu.Unlock()
	return nil
}

func (c *Controller) Status() modelservice.Status {
	c.mu.Lock()
	name := c.active
	item := c.services[name]
	if item == nil {
		for _, candidate := range c.services {
			item = candidate
			break
		}
	}
	c.mu.Unlock()
	if item == nil {
		return modelservice.Status{Lifecycle: modelservice.LifecycleStopped}
	}
	return item.service.Status()
}

func (c *Controller) Statuses() map[string]modelservice.Status {
	c.mu.Lock()
	items := make(map[string]*loadedModel, len(c.services))
	for name, item := range c.services {
		items[name] = item
	}
	c.mu.Unlock()
	result := make(map[string]modelservice.Status, len(items))
	for name, item := range items {
		result[name] = item.service.Status()
	}
	return result
}

func (c *Controller) BackendURL() (string, error) {
	status := c.Status()
	if status.Model == "" {
		return "", errors.New("model service is stopped")
	}
	c.mu.Lock()
	item := c.services[status.Model]
	c.mu.Unlock()
	if item == nil {
		return "", errors.New("model service is stopped")
	}
	return item.service.BackendURL()
}

func (c *Controller) CurrentModel() string { return c.Status().Model }

func (c *Controller) LoadedModels() []string {
	c.mu.Lock()
	result := make([]string, 0, len(c.services))
	for name := range c.services {
		result = append(result, name)
	}
	c.mu.Unlock()
	sort.Strings(result)
	return result
}
