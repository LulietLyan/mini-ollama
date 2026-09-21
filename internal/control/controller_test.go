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
	if err != nil {
		t.Fatal(err)
	}
	manager := New(Config{Catalog: catalog, ServerPath: "/bin/true"})
	if err := manager.Stop(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := manager.Stop(context.Background()); err != nil {
		t.Fatal(err)
	}
	if status := manager.Status(); status.Lifecycle != modelservice.LifecycleStopped {
		t.Fatalf("status = %q", status.Lifecycle)
	}
}

func TestSwitchUnknownModelDoesNotStopCurrentModel(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "known")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "model.gguf"), []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := catalog.New(root)
	if err != nil {
		t.Fatal(err)
	}
	manager := New(Config{Catalog: catalog, ServerPath: "/bin/true"})
	if err := manager.Switch(context.Background(), "missing"); err == nil {
		t.Fatal("expected unknown model error")
	}
	if status := manager.Status(); status.Lifecycle != modelservice.LifecycleStopped {
		t.Fatalf("status = %q", status.Lifecycle)
	}
}
