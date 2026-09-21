package catalog

import (
	"os"
	"path/filepath"
	"testing"
)

func TestManifestOverridesModelName(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "directory-name")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "model.gguf"), []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	manifest := `{"name":"custom:name","file":"model.gguf","size_bytes":4}`
	if err := os.WriteFile(filepath.Join(directory, "manifest.json"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}

	catalog, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := catalog.Find("custom:name"); !ok {
		t.Fatal("manifest name was not used")
	}
}

func TestVerifySize(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "model.gguf")
	if err := os.WriteFile(path, []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	entry, ok := catalog.Find("model")
	if !ok {
		t.Fatal("model was not found")
	}
	entry.SizeBytes++
	if err := catalog.Verify(entry); err == nil {
		t.Fatal("expected size mismatch")
	}
}

func TestManifestRejectsPathTraversal(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "demo")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "model.gguf"), []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	manifest := `{"file":"../outside.gguf"}`
	if err := os.WriteFile(filepath.Join(directory, "manifest.json"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := New(root); err == nil {
		t.Fatal("expected manifest traversal error")
	}
}
