package cmd

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestResolveGGUFPath(t *testing.T) {
	tempDir := t.TempDir()

	modelPath := filepath.Join(tempDir, "model.gguf")

	if err := os.WriteFile(modelPath, []byte("test"), 0o644); err != nil {
		t.Fatalf("write model file: %v", err)
	}

	got, err := resolveGGUFPath(modelPath)
	if err != nil {
		t.Fatalf("resolveGGUFPath returned error: %v", err)
	}

	if got != modelPath {
		t.Fatalf("resolved path = %q, want %q", got, modelPath)
	}
}

func TestResolveGGUFPathErrors(t *testing.T) {
	tempDir := t.TempDir()

	tests := []struct {
		name string
		path string
		want string
	}{
		{
			name: "wrong extension",
			path: filepath.Join(tempDir, "model.bin"),
			want: "model must be a GGUF file",
		},
		{
			name: "missing file",
			path: filepath.Join(tempDir, "missing.gguf"),
			want: "model not found",
		},
		{
			name: "directory",
			path: filepath.Join(tempDir, "directory.gguf"),
			want: "not a regular file",
		},
	}

	if err := os.Mkdir(filepath.Join(tempDir, "directory.gguf"), 0o755); err != nil {
		t.Fatalf("create directory: %v", err)
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := resolveGGUFPath(test.path)
			if err == nil {
				t.Fatal("expected error, got nil")
			}

			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf(
					"error = %q, want substring %q",
					err,
					test.want,
				)
			}
		})
	}
}

func TestValidateRunOptions(t *testing.T) {
	tests := []struct {
		name    string
		options runOptions
		wantErr string
	}{
		{
			name: "defaults are valid",
		},
		{
			name: "positive context size",
			options: runOptions{
				contextSize: 4096,
			},
		},
		{
			name: "valid port",
			options: runOptions{
				port: 8080,
			},
		},
		{
			name: "valid gpu layers",
			options: runOptions{
				gpuLayers: "all",
			},
		},
		{
			name: "valid auto gpu layers",
			options: runOptions{
				gpuLayers: "auto",
			},
		},
		{
			name: "negative context size",
			options: runOptions{
				contextSize: -1,
			},
			wantErr: "context size",
		},
		{
			name: "negative port",
			options: runOptions{
				port: -1,
			},
			wantErr: "port",
		},
		{
			name: "port too large",
			options: runOptions{
				port: 65536,
			},
			wantErr: "port",
		},
		{
			name: "invalid gpu layers",
			options: runOptions{
				gpuLayers: "invalid",
			},
			wantErr: "gpu layers",
		},
		{
			name: "negative gpu layers",
			options: runOptions{
				gpuLayers: "-1",
			},
			wantErr: "gpu layers",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateRunOptions(test.options)

			if test.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}

				return
			}

			if err == nil {
				t.Fatalf("expected error containing %q", test.wantErr)
			}

			if !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf(
					"error = %q, want substring %q",
					err,
					test.wantErr,
				)
			}
		})
	}
}

func TestBuildServerArgs(t *testing.T) {
	options := runOptions{
		contextSize: 4096,
		gpuLayers:   "  all ",
		host:        "127.0.0.1",
		port:        8080,
	}

	got := buildServerArgs("/models/test.gguf", options)

	want := []string{
		"-m",
		"/models/test.gguf",
		"-c",
		"4096",
		"--gpu-layers",
		"all",
		"--host",
		"127.0.0.1",
		"--port",
		"8080",
	}

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("args = %#v, want %#v", got, want)
	}
}

func TestBuildServerArgsWithDefaults(t *testing.T) {
	got := buildServerArgs(
		"/models/test.gguf",
		runOptions{},
	)

	want := []string{
		"-m",
		"/models/test.gguf",
	}

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("args = %#v, want %#v", got, want)
	}
}

func TestBuildHealthURL(t *testing.T) {
	tests := []struct {
		name    string
		options runOptions
		want    string
	}{
		{
			name: "defaults",
			want: "http://127.0.0.1:8080/health",
		},
		{
			name: "custom host and port",
			options: runOptions{
				host: "localhost",
				port: 9090,
			},
			want: "http://localhost:9090/health",
		},
		{
			name: "ipv6 host",
			options: runOptions{
				host: "::1",
				port: 9090,
			},
			want: "http://[::1]:9090/health",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := buildHealthURL(test.options)
			if err != nil {
				t.Fatalf("buildHealthURL returned error: %v", err)
			}

			if got != test.want {
				t.Fatalf("URL = %q, want %q", got, test.want)
			}
		})
	}
}

func TestBuildHealthURLRejectsUnixSocket(t *testing.T) {
	_, err := buildHealthURL(runOptions{
		host: "/tmp/llama-server.sock",
	})
	if err == nil {
		t.Fatal("expected UNIX socket error, got nil")
	}
}

func TestValidateRunOptionsRejectsMultipleDevices(t *testing.T) {
	err := validateRunOptions(runOptions{
		device: "0,1",
	})
	if err == nil {
		t.Fatal("expected multiple-device validation error")
	}
}

func TestBuildProcessEnv(t *testing.T) {
	env := buildProcessEnv("3")

	found := false

	for _, item := range env {
		if item == "CUDA_VISIBLE_DEVICES=3" {
			found = true
			break
		}
	}

	if !found {
		t.Fatal("CUDA_VISIBLE_DEVICES=3 was not found")
	}
}

func TestSelectAvailablePort(t *testing.T) {
	port, err := selectAvailablePort("127.0.0.1")
	if err != nil {
		t.Fatalf("selectAvailablePort returned error: %v", err)
	}

	if port <= 0 || port > 65535 {
		t.Fatalf("invalid port: %d", port)
	}
}
