package service

import (
	"testing"
)

func TestNewServiceStatus(t *testing.T) {
	modelService := New(Config{})

	status := modelService.Status()

	if status.Lifecycle != LifecycleStopped {
		t.Fatalf(
			"lifecycle = %q, want %q",
			status.Lifecycle,
			LifecycleStopped,
		)
	}

	if status.Model != "" {
		t.Fatalf("model = %q, want empty", status.Model)
	}

	if status.RequestCount != 0 {
		t.Fatalf(
			"request count = %d, want zero",
			status.RequestCount,
		)
	}
}

func TestModelName(t *testing.T) {
	got := modelName(
		"/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
	)

	want := "Llama-3.2-1B-Instruct-Q4_K_M"

	if got != want {
		t.Fatalf("model name = %q, want %q", got, want)
	}
}

func TestBuildServerArgs(t *testing.T) {
	got := buildServerArgs(
		"/models/model.gguf",
		"127.0.0.1",
		39001,
		Config{
			ContextSize: 4096,
			GPULayers:   "all",
			SplitMode:   "layer",
			TensorSplit: "3,1",
		},
	)

	want := []string{
		"-m",
		"/models/model.gguf",
		"--host",
		"127.0.0.1",
		"--port",
		"39001",
		"-c",
		"4096",
		"--gpu-layers",
		"all",
		"--split-mode",
		"layer",
		"--tensor-split",
		"3,1",
		"--metrics",
	}

	if len(got) != len(want) {
		t.Fatalf("args = %#v, want %#v", got, want)
	}

	for index := range want {
		if got[index] != want[index] {
			t.Fatalf(
				"args[%d] = %q, want %q",
				index,
				got[index],
				want[index],
			)
		}
	}
}

func TestValidateDeviceListAcceptsMultipleDevices(t *testing.T) {
	if err := validateDeviceList("0,1"); err != nil {
		t.Fatalf("unexpected multi-device validation error: %v", err)
	}
}

func TestValidateDeviceListRejectsDuplicateDevices(t *testing.T) {
	if err := validateDeviceList("0,0"); err == nil {
		t.Fatal("expected duplicate-device validation error")
	}
}
