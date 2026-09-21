package cmd

import (
	"fmt"
	"os"
	"path/filepath"
)

// 从候选集合中找到并返回一个有效的 llamaServer 路径
func llamaServerPath() (string, error) {
	candidates := make([]string, 0, 4)

	if configuredPath := os.Getenv("MINI_OLLAMA_SERVER"); configuredPath != "" {
		candidates = append(candidates, configuredPath)
	}

	if workDir, err := os.Getwd(); err == nil {
		candidates = append(candidates, filepath.Join(
			workDir,
			"build",
			"llama-server",
			"bin",
			"llama-server",
		))
	}

	if executablePath, err := os.Executable(); err == nil {
		executableDir := filepath.Dir(executablePath)

		candidates = append(candidates, filepath.Join(
			executableDir,
			"build",
			"llama-server",
			"bin",
			"llama-server",
		))

		candidates = append(candidates, filepath.Join(
			filepath.Dir(executableDir),
			"build",
			"llama-server",
			"bin",
			"llama-server",
		))
	}

	checked := make(map[string]bool)

	for _, candidate := range candidates {
		candidate = filepath.Clean(candidate)

		if checked[candidate] {
			continue
		}
		checked[candidate] = true

		info, err := os.Stat(candidate)
		if err != nil {
			continue
		}

		if info.IsDir() {
			return "", fmt.Errorf("llama-server path is a directory: %s", candidate)
		}

		// 检查权限，Unix 文件权限里的执行权限位
		if info.Mode()&0111 == 0 {
			return "", fmt.Errorf("llama-server is not executable: %s", candidate)
		}

		return candidate, nil
	}

	return "", fmt.Errorf(
		"llama-server not found; expected build/llama-server/bin/llama-server",
	)
}
