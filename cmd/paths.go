package cmd

import (
	"fmt"
	"os"
	"path/filepath"
)

func defaultModelsDir() (string, error) {
	if value := os.Getenv("MINI_OLLAMA_MODELS_DIR"); value != "" {
		return value, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("find user home directory: %w", err)
	}
	return filepath.Join(home, ".mini-ollama", "models"), nil
}

func defaultDataDir() (string, error) {
	if value := os.Getenv("MINI_OLLAMA_DATA_DIR"); value != "" {
		return value, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("find user home directory: %w", err)
	}
	return filepath.Join(home, ".mini-ollama"), nil
}
