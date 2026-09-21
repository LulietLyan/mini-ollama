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
	Name        string `json:"name"`
	Path        string `json:"-"`
	Description string `json:"description,omitempty"`
	SizeBytes   int64  `json:"size_bytes"`
	SHA256      string `json:"-"`
	Manifest    string `json:"-"`
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
			if err != nil {
				return nil, err
			}
			if !ok {
				continue
			}
			if _, exists := entries[entry.Name]; exists {
				return nil, fmt.Errorf("duplicate model name: %s", entry.Name)
			}
			entries[entry.Name] = entry
			continue
		}
		if !item.Type().IsRegular() || !isGGUF(path) {
			continue
		}
		info, err := item.Info()
		if err != nil {
			return nil, err
		}
		name := strings.TrimSuffix(item.Name(), filepath.Ext(item.Name()))
		entries[name] = Entry{Name: name, Path: path, SizeBytes: info.Size()}
	}
	return entries, nil
}

// 扫描一个模型目录，找到其中的 GGUF 模型文件
// 并结合 manifest.json 生成一个 Entry
func scanModelDirectory(directory string) (Entry, bool, error) {
	manifestPath := filepath.Join(directory, "manifest.json")
	manifest, hasManifest, err := readManifest(manifestPath)
	if err != nil {
		return Entry{}, false, err
	}

	files, err := os.ReadDir(directory)
	if err != nil {
		return Entry{}, false, fmt.Errorf("read model directory: %w", err)
	}
	gguf := make([]string, 0, 1)
	for _, item := range files {
		if !item.IsDir() && item.Type().IsRegular() && isGGUF(item.Name()) {
			gguf = append(gguf, filepath.Join(directory, item.Name()))
		}
	}
	if len(gguf) == 0 {
		return Entry{}, false, nil
	}
	if len(gguf) > 1 && (!hasManifest || manifest.File == "") {
		return Entry{}, false, fmt.Errorf("multiple GGUF files without manifest file field: %s", directory)
	}

	file := gguf[0]
	if hasManifest && manifest.File != "" {
		cleanFile := filepath.Clean(manifest.File)
		if filepath.IsAbs(cleanFile) ||
			cleanFile == "." ||
			cleanFile == ".." ||
			strings.HasPrefix(cleanFile, ".."+string(os.PathSeparator)) {
			return Entry{}, false, fmt.Errorf("manifest file escapes model directory: %s", manifest.File)
		}

		candidate := filepath.Join(directory, cleanFile)
		if filepath.Dir(candidate) != filepath.Clean(directory) || !isGGUF(candidate) {
			return Entry{}, false, fmt.Errorf("manifest file must be a GGUF directly inside %s", directory)
		}
		file = candidate
	}

	info, err := os.Stat(file)
	if err != nil {
		return Entry{}, false, fmt.Errorf("check model file: %w", err)
	}

	name := filepath.Base(directory)
	if hasManifest && strings.TrimSpace(manifest.Name) != "" {
		name = strings.TrimSpace(manifest.Name)
	}
	entry := Entry{Name: name, Path: file, SizeBytes: info.Size()}
	if hasManifest {
		entry.Description = manifest.Description
		entry.SHA256 = strings.ToLower(strings.TrimSpace(manifest.SHA256))
		entry.Manifest = manifestPath
		if manifest.SizeBytes > 0 {
			entry.SizeBytes = manifest.SizeBytes
		}
	}
	return entry, true, nil
}

// 读取一个 JSON 格式的 manifest 文件
// 并返回文件是否存在、读取/解析是否出错，以及解析出来的 Manifest
func readManifest(path string) (Manifest, bool, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return Manifest{}, false, nil
	}
	if err != nil {
		return Manifest{}, false, fmt.Errorf("read manifest: %w", err)
	}

	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return Manifest{}, false, fmt.Errorf("decode manifest %s: %w", path, err)
	}
	if err := validateManifest(manifest); err != nil {
		return Manifest{}, false, fmt.Errorf("validate manifest %s: %w", path, err)
	}

	return manifest, true, nil
}

// 扩展名是否为 gguf
func isGGUF(path string) bool { return strings.EqualFold(filepath.Ext(path), ".gguf") }

// 根据 path 与 root 的相对位置关系决定返回父目录名还是文件名（不包含扩展名）
func nameForPath(root, path string) string {
	parent := filepath.Dir(path)
	if filepath.Clean(parent) != filepath.Clean(root) {
		return filepath.Base(parent)
	}
	return strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
}

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
