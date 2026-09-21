package hfimport

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"
)

type Options struct {
	RepoID    string
	Revision  string
	SourceDir string
	Name      string
	ModelsDir string
	WorkDir   string
	Python    string
	Quantizer string
	TokenFile string
	Stdout    io.Writer
	Stderr    io.Writer
}

type Result struct {
	Name      string
	Path      string
	SizeBytes int64
	SHA256    string
}

type manifest struct {
	Name      string `json:"name"`
	File      string `json:"file"`
	SizeBytes int64  `json:"size_bytes"`
	SHA256    string `json:"sha256"`
}

// 限制模型名称、仓库名称等
var safeName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]*$`)
var repoID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$`)
var revisionID = regexp.MustCompile(`^[0-9a-fA-F]{40}$`)

// Import()
//
//	│
//	├── ① validate(options)
//	│
//	├── ② 确定 modelsDir / workDir
//	│
//	├── ③ 检查目录安全性
//	│
//	├── ④ 创建 job + 加锁
//	│
//	├── ⑤ 准备 HF_TOKEN
//	│
//	├── ⑥ 如果是远程模型
//	│      └── 下载 Hugging Face snapshot
//	│
//	├── ⑦ 验证 HF 模型目录
//	│
//	├── ⑧ HF → F16 GGUF
//	│
//	├── ⑨ F16 GGUF → Q4_K_M GGUF
//	│
//	└── ⑩ publish()
//	        │
//	        ├── 复制到 modelsDir
//	        ├── 校验 SHA256
//	        ├── 写 manifest.json
//	        └── 原子 rename
func Import(ctx context.Context, options Options) (Result, error) {
	if err := validate(options); err != nil {
		return Result{}, err
	}

	root, err := os.Getwd()
	if err != nil {
		return Result{}, err
	}

	modelsDir, err := filepath.Abs(options.ModelsDir)
	if err != nil {
		return Result{}, err
	}

	workDir, err := filepath.Abs(options.WorkDir)
	if err != nil {
		return Result{}, err
	}

	if within(root, modelsDir) || within(root, workDir) {
		return Result{}, errors.New("models and work directories must be outside the repository")
	}
	if within(workDir, modelsDir) || within(modelsDir, workDir) {
		return Result{}, errors.New("models and work directories must not overlap")
	}

	if err := os.MkdirAll(modelsDir, 0o755); err != nil {
		return Result{}, err
	}
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		return Result{}, err
	}

	root, err = filepath.EvalSymlinks(root)
	if err != nil {
		return Result{}, err
	}

	modelsDir, err = filepath.EvalSymlinks(modelsDir)
	if err != nil {
		return Result{}, err
	}

	workDir, err = filepath.EvalSymlinks(workDir)
	if err != nil {
		return Result{}, err
	}

	if within(root, modelsDir) || within(root, workDir) ||
		within(workDir, modelsDir) || within(modelsDir, workDir) {
		return Result{}, errors.New("models and work directories must be outside the repository and not overlap")
	}

	if _, err := os.Lstat(filepath.Join(modelsDir, options.Name)); err == nil {
		return Result{}, fmt.Errorf("model already exists: %s", options.Name)
	} else if !errors.Is(err, os.ErrNotExist) {
		return Result{}, err
	}

	identity := options.RepoID + "@" + options.Revision
	sourceDir := options.SourceDir
	if sourceDir != "" {
		sourceDir, err = filepath.Abs(sourceDir)
		if err != nil {
			return Result{}, err
		}
		if err := validateSource(sourceDir); err != nil {
			return Result{}, err
		}
		fingerprint, err := sourceFingerprint(sourceDir)
		if err != nil {
			return Result{}, err
		}
		identity = "local:" + sourceDir + ":" + fingerprint
	}

	// 以上的过程可以简要描述为：
	// 每个模型文件的内容 hash 拼接到一起
	// 模型文件目录的文件名 hash 拼接到一起
	// 上述二者拼接成 identity 并 sha256 后得到 key
	// 截取 key 前 8 字节作为 job 目录名
	key := sha256.Sum256([]byte(identity))
	job := filepath.Join(workDir, options.Name+"-"+hex.EncodeToString(key[:8]))
	if err := os.MkdirAll(job, 0o700); err != nil {
		return Result{}, err
	}

	// 给当前 job 加互斥锁（lock），防止多个进程/线程同时处理同一个 job
	unlock, err := lock(filepath.Join(job, ".lock"))
	if err != nil {
		return Result{}, err
	}
	defer unlock()

	pythonEnv := withoutHFToken(os.Environ())
	if options.TokenFile != "" {
		secret, err := os.ReadFile(options.TokenFile)
		if err != nil {
			return Result{}, fmt.Errorf("read token file: %w", err)
		}
		pythonEnv = append(pythonEnv, "HF_TOKEN="+strings.TrimSpace(string(secret)))
	} else if token := os.Getenv("HF_TOKEN"); token != "" {
		pythonEnv = append(pythonEnv, "HF_TOKEN="+token)
	}

	if sourceDir == "" {
		if !hasProxy() {
			return Result{}, errors.New("source ~/.bashrc and set proxy variables before downloading")
		}
		sourceDir = filepath.Join(job, "snapshot")
		if err := os.MkdirAll(sourceDir, 0o700); err != nil {
			return Result{}, err
		}
		arguments := []string{
			"scripts/hf_snapshot.py",
			"--repo", options.RepoID,
			"--revision", options.Revision,
			"--directory", sourceDir,
			"--report", filepath.Join(job, "download-verified.json"),
		}
		if err := run(ctx, pythonEnv, options.Stdout, options.Stderr, options.Python, arguments...); err != nil {
			return Result{}, fmt.Errorf("download or verify HF snapshot: %w", err)
		}
	}
	if err := validateSource(sourceDir); err != nil {
		return Result{}, err
	}

	// 先把 Hugging Face 模型转换成 F16 GGUF，再把 F16 GGUF 量化成 Q4_K_M GGUF
	converted := filepath.Join(job, "converted.f16.gguf")
	if err := ensureArtifact(ctx, converted, func(output string) error {
		return run(ctx, withoutHFToken(os.Environ()), options.Stdout, options.Stderr,
			options.Python, "third_party/llama.cpp/convert_hf_to_gguf.py",
			"--outfile", output, "--outtype", "f16", sourceDir)
	}); err != nil {
		return Result{}, fmt.Errorf("convert HF weights: %w", err)
	}

	quantized := filepath.Join(job, "quantized.Q4_K_M.gguf")
	if err := ensureArtifact(ctx, quantized, func(output string) error {
		return run(ctx, withoutHFToken(os.Environ()), options.Stdout, options.Stderr,
			options.Quantizer, converted, output, "Q4_K_M")
	}); err != nil {
		return Result{}, fmt.Errorf("quantize GGUF: %w", err)
	}

	// 把已经生成好的 Q4_K_M GGUF 正式发布到模型目录
	return publish(modelsDir, options.Name, quantized)
}

func validate(options Options) error {
	if !safeName.MatchString(options.Name) ||
		strings.Contains(options.Name, "..") {
		return errors.New("model name must contain only letters, digits, '.', '_', ':' or '-'")
	}

	if options.ModelsDir == "" ||
		options.WorkDir == "" ||
		options.Python == "" ||
		options.Quantizer == "" {
		return errors.New("models-dir, work-dir, python and quantizer are required")
	}

	if options.SourceDir != "" {
		if options.RepoID != "" ||
			options.Revision != "" ||
			options.TokenFile != "" {
			return errors.New("source-dir cannot be combined with repo, revision or token-file")
		}
	} else if !repoID.MatchString(options.RepoID) ||
		!revisionID.MatchString(options.Revision) {
		return errors.New("remote import requires OWNER/REPO and a 40-character commit revision")
	}

	return nil
}

func hasProxy() bool {
	for _, name := range []string{"HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"} {
		if os.Getenv(name) != "" {
			return true
		}
	}

	return false
}

func withoutHFToken(environment []string) []string {
	result := make([]string, 0, len(environment))
	for _, item := range environment {
		if !strings.HasPrefix(item, "HF_TOKEN=") {
			result = append(result, item)
		}
	}

	return result
}

func within(parent, child string) bool {
	relative, err := filepath.Rel(filepath.Clean(parent), filepath.Clean(child))

	return err == nil &&
		relative != ".." &&
		!strings.HasPrefix(relative, ".."+string(os.PathSeparator))
}

func lock(path string) (func(), error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open job lock: %w", err)
	}
	if err := syscall.Flock(int(file.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		_ = file.Close()
		return nil, errors.New("another import is using this job or model directory")
	}

	return func() {
		_ = syscall.Flock(int(file.Fd()), syscall.LOCK_UN)
		_ = file.Close()
	}, nil
}

func run(ctx context.Context, environment []string, stdout, stderr io.Writer, program string, args ...string) error {
	command := exec.CommandContext(ctx, program, args...)
	command.Env = environment
	command.Stdout = stdout
	command.Stderr = stderr
	if err := command.Run(); err != nil {
		return fmt.Errorf("%s failed: %w", filepath.Base(program), err)
	}

	return nil
}

func validateSource(directory string) error {
	info, err := os.Stat(filepath.Join(directory, "config.json"))
	if err != nil || !info.Mode().IsRegular() {
		return errors.New("HF source must contain config.json")
	}
	weights, err := filepath.Glob(filepath.Join(directory, "*.safetensors"))
	if err != nil {
		return err
	}
	if len(weights) == 0 {
		return errors.New("HF source must contain safetensors weights")
	}

	return nil
}

func sourceFingerprint(directory string) (string, error) {
	items, err := os.ReadDir(directory)
	if err != nil {
		return "", err
	}

	combined := sha256.New()
	for _, item := range items {
		if item.IsDir() {
			continue
		}
		if !item.Type().IsRegular() {
			return "", fmt.Errorf("source contains non-regular file: %s", item.Name())
		}
		checksum, size, err := digest(filepath.Join(directory, item.Name()))
		if err != nil {
			return "", err
		}
		fmt.Fprintf(combined, "%s:%d:%s\n", item.Name(), size, checksum)
	}

	return hex.EncodeToString(combined.Sum(nil)), nil
}

func digest(path string) (string, int64, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return "", 0, err
	}
	if !info.Mode().IsRegular() {
		return "", 0, errors.New("artifact must be a regular file")
	}
	checksum := sha256.New()
	count, err := io.Copy(checksum, file)
	if err != nil {
		return "", 0, err
	}

	return hex.EncodeToString(checksum.Sum(nil)), count, nil
}

func validGGUF(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	var signature [4]byte
	if _, err := io.ReadFull(file, signature[:]); err != nil {
		return err
	}
	if string(signature[:]) != "GGUF" {
		return errors.New("artifact has no GGUF header")
	}

	return nil
}

func ensureArtifact(ctx context.Context, path string, generate func(string) error) error {
	marker := path + ".sha256"
	if _, err := os.Stat(path); err == nil {
		if err := validGGUF(path); err != nil {
			return err
		}
		want, err := os.ReadFile(marker)
		if err != nil {
			return fmt.Errorf("existing artifact has no checksum marker: %w", err)
		}
		got, _, err := digest(path)
		if err != nil {
			return err
		}
		if got != strings.TrimSpace(string(want)) {
			return errors.New("cached GGUF checksum mismatch")
		}
		return nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}

	temporaryDir, err := os.MkdirTemp(filepath.Dir(path), ".gguf-stage-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(temporaryDir)
	output := filepath.Join(temporaryDir, "model.gguf")
	if err := generate(output); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validGGUF(output); err != nil {
		return err
	}
	hash, size, err := digest(output)
	if err != nil {
		return err
	}
	if size <= 4 {
		return errors.New("GGUF artifact is empty")
	}
	markerTemp := marker + fmt.Sprintf(".tmp-%d", os.Getpid())
	if err := os.WriteFile(markerTemp, []byte(hash+"\n"), 0o600); err != nil {
		return err
	}
	if err := os.Rename(output, path); err != nil {
		_ = os.Remove(markerTemp)
		return err
	}
	if err := os.Rename(markerTemp, marker); err != nil {
		_ = os.Remove(path)
		_ = os.Remove(markerTemp)
		return err
	}
	return nil
}

func publish(modelsDir, name, source string) (Result, error) {
	unlock, err := lock(filepath.Join(modelsDir, ".publish.lock"))
	if err != nil {
		return Result{}, err
	}
	defer unlock()
	target := filepath.Join(modelsDir, name)
	if _, err := os.Lstat(target); err == nil {
		return Result{}, fmt.Errorf("model already exists: %s", name)
	}
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return Result{}, err
	}

	// Stage in the parent of modelsDir: rename then stays on the same filesystem,
	// while an unfinished directory is never visible to the model scanner.
	stage, err := os.MkdirTemp(filepath.Dir(modelsDir), ".mini-ollama-publish-")
	if err != nil {
		return Result{}, err
	}
	defer os.RemoveAll(stage)
	fileName := name + ".Q4_K_M.gguf"
	output := filepath.Join(stage, fileName)
	input, err := os.Open(source)
	if err != nil {
		return Result{}, err
	}
	defer input.Close()
	destination, err := os.OpenFile(output, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return Result{}, err
	}

	_, copyErr := io.Copy(destination, input)
	closeErr := destination.Close()
	if copyErr != nil {
		return Result{}, copyErr
	}
	if closeErr != nil {
		return Result{}, closeErr
	}
	if err := validGGUF(output); err != nil {
		return Result{}, err
	}
	originalHash, originalSize, err := digest(source)
	if err != nil {
		return Result{}, err
	}
	outputHash, outputSize, err := digest(output)
	if err != nil {
		return Result{}, err
	}
	if originalHash != outputHash || originalSize != outputSize {
		return Result{}, errors.New("published copy differs from quantized GGUF")
	}

	metadata := manifest{Name: name, File: fileName, SizeBytes: outputSize, SHA256: outputHash}
	data, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return Result{}, err
	}
	if err := os.WriteFile(filepath.Join(stage, "manifest.json"), append(data, '\n'), 0o644); err != nil {
		return Result{}, err
	}
	if err := os.Rename(stage, target); err != nil {
		return Result{}, fmt.Errorf("publish model from %s: %w", stage, err)
	}

	return Result{
		Name:      name,
		Path:      filepath.Join(target, fileName),
		SizeBytes: outputSize,
		SHA256:    outputHash,
	}, nil
}
