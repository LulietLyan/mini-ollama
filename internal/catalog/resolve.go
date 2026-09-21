package catalog

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// 无论给定 value 是模型名抑或是路径，都尽量将最终的模型 Entry 实体解析出来
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
