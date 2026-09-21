package hfimport

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestValidateRemoteIdentity(t *testing.T) {
	options := Options{
		RepoID:    "example/model",
		Revision:  strings.Repeat("a", 40),
		Name:      "example:1b",
		ModelsDir: "/outside/models",
		WorkDir:   "/outside/work",
		Python:    "python",
		Quantizer: "llama-quantize",
	}
	if err := validate(options); err != nil {
		t.Fatal(err)
	}
	options.Name = "../escape"
	if err := validate(options); err == nil {
		t.Fatal("accepted unsafe model name")
	}
	options.Name = "example:1b"
	options.Revision = "main"
	if err := validate(options); err == nil {
		t.Fatal("accepted movable revision")
	}
}

func TestEnsureArtifactReusesVerifiedFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "converted.gguf")
	called := 0
	generate := func(output string) error {
		called++
		return os.WriteFile(output, []byte("GGUFpayload"), 0o600)
	}
	if err := ensureArtifact(context.Background(), path, generate); err != nil {
		t.Fatal(err)
	}
	if err := ensureArtifact(context.Background(), path, generate); err != nil {
		t.Fatal(err)
	}
	if called != 1 {
		t.Fatalf("generate called %d times; want 1", called)
	}
	if err := os.WriteFile(path, []byte("GGUFchanged"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := ensureArtifact(context.Background(), path, generate); err == nil {
		t.Fatal("accepted modified cached GGUF")
	}
}

func TestPublishNeverOverwritesModel(t *testing.T) {
	root := t.TempDir()
	models := filepath.Join(root, "models")
	if err := os.Mkdir(models, 0o755); err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(root, "source.gguf")
	if err := os.WriteFile(source, []byte("GGUFpayload"), 0o600); err != nil {
		t.Fatal(err)
	}
	result, err := publish(models, "demo:1b", source)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(models, "demo:1b", "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var metadata manifest
	if err := json.Unmarshal(data, &metadata); err != nil {
		t.Fatal(err)
	}
	if metadata.Name != "demo:1b" || metadata.SizeBytes != result.SizeBytes || metadata.SHA256 != result.SHA256 {
		t.Fatalf("invalid manifest: %+v", metadata)
	}
	if _, err := publish(models, "demo:1b", source); err == nil {
		t.Fatal("existing model was overwritten")
	}
	unchanged, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	if string(unchanged) != "GGUFpayload" {
		t.Fatal("original weights were changed")
	}
}

func TestWithoutHFToken(t *testing.T) {
	result := withoutHFToken([]string{"PATH=/bin", "HF_TOKEN=secret"})
	if len(result) != 1 || result[0] != "PATH=/bin" {
		t.Fatalf("environment = %v", result)
	}
}
