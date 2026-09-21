# 阶段 4：Hugging Face 下载、GGUF 转换和量化

本阶段增加独立的 `mini-ollama import-hf` 命令。它从固定的 Hugging Face commit 下载 safetensors 权重，校验远端 LFS 文件的大小和 SHA-256，在缓存目录生成 F16 GGUF 和 Q4_K_M GGUF，最后发布到阶段 3 的模型目录。也支持 `--source-dir` 导入已经存在的本地 HF 权重，此模式不访问网络。

本文档给出需要手动输入的**完整文件**。仅编辑本阶段列出的文件；不修改 `ollama/`，不复制它，也不覆盖原始权重。尚未输入的阶段 2、3 代码先按各自文档完成；本阶段仅添加 CLI 工作流，不新增下载 HTTP 接口。`serve` 正在运行时其目录快照不会自动刷新，发布成功后重新启动 `serve` 才能在 `/api/v1/models` 中看到新模型。

## 使用边界和准备

源代码目录内不得放模型、下载缓存或中间 GGUF。当前机器的所有大文件读写都放在 CephFS：工作目录使用 `/rtai_cephfs/liangjm/models/.cache/mini-ollama-import`，已发布的模型放 `/rtai_cephfs/liangjm/models/GGUF`。不要把 `--work-dir` 改到仓库目录、家目录下的临时目录或本机 `/tmp`；本机空间只用于源码、日志和小型构建产物。其他 Linux/macOS 机器可用 `--work-dir`、`--models-dir` 指定外部目录。下载前先检查目标 repo、commit、缓存路径和发布路径；命令不会自动下载模型，也不会自行运行推理。

项目已有的 `hfd.sh` 在某些分支会打印包含 token 的下载命令，阶段 4 不调用它。Go 程序可从 `--token-file` 读取 token，仅作为子进程环境变量 `HF_TOKEN` 传给 Python；不会将其放到参数、URL、manifest 或程序日志中。不要使用 `set -x` 执行涉及凭证的命令。

本机联网先在当前 shell 加载 `~/.bashrc` 的代理并检查相关环境变量。Go 命令会要求代理变量存在，但不会代替你配置代理：

```bash
source ~/.bashrc
env | rg -i '^(https?_proxy|all_proxy|HF_ENDPOINT)=' | cut -d= -f1
```

下载脚本默认使用 `https://hf-mirror.com`，也可以用 `HF_ENDPOINT` 覆盖。代理仍然由 `~/.bashrc` 提供；镜像地址不是代理地址，两者都需要按本机网络情况配置。无需把 token 写入 `~/.bashrc`，需要认证时只传 token 文件路径，例如 `--token-file "$PWD/huggingface_token.txt"`，程序不会打印或复制文件内容。

后续 Go 命令中的 `--repo` 是 Hugging Face `OWNER/REPO`，`--revision` 必须是 40 位完整 commit SHA。使用固定 commit 避免 `main` 移动时把不同版本混到同一缓存。下载器会检查仓库元数据给出的文件大小，并核对每个 `.safetensors` 的远端 LFS SHA-256；配置和 tokenizer 文件也计算本地 SHA-256 以便追踪，但 Git blob ID 不是文件的 SHA-256，不能当作远端 SHA-256 使用。

转换器来自仓库当前检出的 `third_party/llama.cpp/convert_hf_to_gguf.py`；量化工具来自同一构建树。现在 `llama-quantize` 尚未构建，先执行：

```bash
cmake --build build/llama-server --target llama-quantize --parallel 4
build/llama-server/bin/llama-quantize --help
```

Python 包只通过 conda 安装；不运行 `pip install`。在 conda 环境中安装转换器需要的包（仓库脚本加载 torch，本机基础 Python 尚无 torch）：

```bash
conda create -n mini-ollama-hf -c conda-forge python=3.11 pytorch transformers=4.57.6 huggingface_hub safetensors sentencepiece protobuf numpy tokenizers -y
conda activate mini-ollama-hf
python -c 'import torch, transformers, huggingface_hub, safetensors, sentencepiece, google.protobuf, numpy'
```

某些新模型的转换器还需要专用 Python 依赖；遇到明确的缺包错误时只用 conda 补齐。阶段 4 暂不支持模型的图像/音频投影文件、多文件 GGUF 输出、重度重新量化，导入前确保有足够空间保留原权重、F16 中间文件和量化输出。

## 文件 1：`scripts/hf_snapshot.py`

这个脚本通过 `huggingface_hub` 固定 commit 下载并校验 safetensors。首版只处理仓库根目录中的 safetensors 和常见配置、tokenizer 文件；嵌套权重或仅有 PyTorch `.bin` 的仓库会明确失败。`snapshot_download` 在 `local_dir/.cache/huggingface` 记录断点信息；失败后重试会复用已有缓存，但每次都重新校验文件大小和 LFS SHA-256。所有路径均位于由 Go 指定的外部工作目录。

```python
#!/usr/bin/env python3
"""Download one immutable HF snapshot and verify its model weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys


ALLOWED_SUFFIXES = (
    ".safetensors", ".json", ".model", ".tiktoken", ".vocab", ".txt",
)
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def endpoint():
    return (os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")


def wanted(path):
    parts = PurePosixPath(path).parts
    return len(parts) == 1 and not parts[0].startswith(".") and parts[0].endswith(ALLOWED_SUFFIXES)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_tree(directory, entries):
    """Verify expected size for every file and HF LFS digest for weights."""
    record = []
    weights = 0
    for entry in sorted(entries, key=lambda item: item.path):
        relative = PurePosixPath(entry.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid repository file path")
        path = directory.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError("downloaded file missing or symbolic link")
        size = path.stat().st_size
        if entry.size is None or size != entry.size or size <= 0:
            raise ValueError("downloaded file size mismatch")
        checksum = file_digest(path)
        if entry.path.endswith(".safetensors"):
            weights += 1
            remote_sha = (entry.lfs or {}).get("sha256")
            if not remote_sha or checksum.lower() != remote_sha.lower():
                raise ValueError("safetensors LFS SHA-256 mismatch or unavailable")
        record.append({"path": entry.path, "size_bytes": size, "sha256": checksum})
    if weights == 0:
        raise ValueError("repository has no safetensors weights")
    return record


def run(repo_id, revision, directory, report_path):
    # Delay importing Hub until the conda environment is activated.
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.hf_api import RepoFile

    token = os.environ.get("HF_TOKEN") or None
    hub_endpoint = endpoint()
    api = HfApi(endpoint=hub_endpoint, token=token)
    info = api.model_info(repo_id=repo_id, revision=revision)
    if info.sha.lower() != revision.lower():
        raise ValueError("repository commit differs from requested revision")

    entries = [
        item for item in api.list_repo_tree(
            repo_id=repo_id, repo_type="model", revision=revision,
            recursive=True, expand=True, token=token,
        ) if isinstance(item, RepoFile) and wanted(item.path)
    ]
    if not entries:
        raise ValueError("repository has no supported conversion files")
    paths = [entry.path for entry in entries]
    snapshot_download(
        repo_id=repo_id, repo_type="model", revision=revision,
        local_dir=str(directory), allow_patterns=paths, token=token,
        max_workers=4, endpoint=hub_endpoint,
    )
    record = verify_tree(directory, entries)
    report = {"repository": repo_id, "revision": revision, "files": record}
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print("Verified", len(record), "downloaded files at", directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(args.repo, args.revision, args.directory, args.report)
    except ValueError as exc:
        print("Hugging Face verification failed:", str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        # Never print request headers, environment variables or token values.
        print("Hugging Face download or verification failed:", type(exc).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## 文件 2：`internal/hfimport/import.go`

Go 代码管理下载、转换、量化和发布。一次导入的工作目录有进程锁；失败时原始权重和中间产物保留，重新运行时会校验已完成 GGUF 的 SHA-256，再决定是否复用。最终发布目录已经存在时返回错误，绝不覆盖。

```go
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
	RepoID     string
	Revision   string
	SourceDir  string
	Name       string
	ModelsDir  string
	WorkDir    string
	Python     string
	Quantizer  string
	TokenFile  string
	Stdout     io.Writer
	Stderr     io.Writer
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

var safeName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]*$`)
var repoID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$`)
var revisionID = regexp.MustCompile(`^[0-9a-fA-F]{40}$`)

func Import(ctx context.Context, options Options) (Result, error) {
	if err := validate(options); err != nil {
		return Result{}, err
	}
	root, err := os.Getwd()
	if err != nil { return Result{}, err }
	modelsDir, err := filepath.Abs(options.ModelsDir)
	if err != nil { return Result{}, err }
	workDir, err := filepath.Abs(options.WorkDir)
	if err != nil { return Result{}, err }
	if within(root, modelsDir) || within(root, workDir) {
		return Result{}, errors.New("models and work directories must be outside the repository")
	}
	if within(workDir, modelsDir) || within(modelsDir, workDir) {
		return Result{}, errors.New("models and work directories must not overlap")
	}
	if err := os.MkdirAll(modelsDir, 0o755); err != nil { return Result{}, err }
	if err := os.MkdirAll(workDir, 0o755); err != nil { return Result{}, err }
	root, err = filepath.EvalSymlinks(root)
	if err != nil { return Result{}, err }
	modelsDir, err = filepath.EvalSymlinks(modelsDir)
	if err != nil { return Result{}, err }
	workDir, err = filepath.EvalSymlinks(workDir)
	if err != nil { return Result{}, err }
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
		if err != nil { return Result{}, err }
		if err := validateSource(sourceDir); err != nil { return Result{}, err }
		fingerprint, err := sourceFingerprint(sourceDir)
		if err != nil { return Result{}, err }
		identity = "local:" + sourceDir + ":" + fingerprint
	}
	key := sha256.Sum256([]byte(identity))
	job := filepath.Join(workDir, options.Name+"-"+hex.EncodeToString(key[:8]))
	if err := os.MkdirAll(job, 0o700); err != nil { return Result{}, err }
	unlock, err := lock(filepath.Join(job, ".lock"))
	if err != nil { return Result{}, err }
	defer unlock()

	pythonEnv := withoutHFToken(os.Environ())
	if options.TokenFile != "" {
		secret, err := os.ReadFile(options.TokenFile)
		if err != nil { return Result{}, fmt.Errorf("read token file: %w", err) }
		pythonEnv = append(pythonEnv, "HF_TOKEN="+strings.TrimSpace(string(secret)))
	} else if token := os.Getenv("HF_TOKEN"); token != "" {
		pythonEnv = append(pythonEnv, "HF_TOKEN="+token)
	}

	if sourceDir == "" {
		if !hasProxy() {
			return Result{}, errors.New("source ~/.bashrc and set proxy variables before downloading")
		}
		sourceDir = filepath.Join(job, "snapshot")
		if err := os.MkdirAll(sourceDir, 0o700); err != nil { return Result{}, err }
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
	if err := validateSource(sourceDir); err != nil { return Result{}, err }

	converted := filepath.Join(job, "converted.f16.gguf")
	if err := ensureArtifact(ctx, converted, func(output string) error {
		return run(ctx, withoutHFToken(os.Environ()), options.Stdout, options.Stderr,
			options.Python, "third_party/llama.cpp/convert_hf_to_gguf.py",
			"--outfile", output, "--outtype", "f16", sourceDir)
	}); err != nil { return Result{}, fmt.Errorf("convert HF weights: %w", err) }

	quantized := filepath.Join(job, "quantized.Q4_K_M.gguf")
	if err := ensureArtifact(ctx, quantized, func(output string) error {
		return run(ctx, withoutHFToken(os.Environ()), options.Stdout, options.Stderr,
			options.Quantizer, converted, output, "Q4_K_M")
	}); err != nil { return Result{}, fmt.Errorf("quantize GGUF: %w", err) }

	return publish(modelsDir, options.Name, quantized)
}

func validate(options Options) error {
	if !safeName.MatchString(options.Name) || strings.Contains(options.Name, "..") {
		return errors.New("model name must contain only letters, digits, '.', '_', ':' or '-'")
	}
	if options.ModelsDir == "" || options.WorkDir == "" || options.Python == "" || options.Quantizer == "" {
		return errors.New("models-dir, work-dir, python and quantizer are required")
	}
	if options.SourceDir != "" {
		if options.RepoID != "" || options.Revision != "" || options.TokenFile != "" {
			return errors.New("source-dir cannot be combined with repo, revision or token-file")
		}
	} else if !repoID.MatchString(options.RepoID) || !revisionID.MatchString(options.Revision) {
		return errors.New("remote import requires OWNER/REPO and a 40-character commit revision")
	}
	return nil
}

func hasProxy() bool {
	for _, name := range []string{"HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"} {
		if os.Getenv(name) != "" { return true }
	}
	return false
}

func withoutHFToken(environment []string) []string {
	result := make([]string, 0, len(environment))
	for _, item := range environment {
		if !strings.HasPrefix(item, "HF_TOKEN=") { result = append(result, item) }
	}
	return result
}

func within(parent, child string) bool {
	relative, err := filepath.Rel(filepath.Clean(parent), filepath.Clean(child))
	return err == nil && relative != ".." &&
		!strings.HasPrefix(relative, ".."+string(os.PathSeparator))
}

func lock(path string) (func(), error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil { return nil, fmt.Errorf("open job lock: %w", err) }
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
	if err := command.Run(); err != nil { return fmt.Errorf("%s failed: %w", filepath.Base(program), err) }
	return nil
}

func validateSource(directory string) error {
	info, err := os.Stat(filepath.Join(directory, "config.json"))
	if err != nil || !info.Mode().IsRegular() { return errors.New("HF source must contain config.json") }
	weights, err := filepath.Glob(filepath.Join(directory, "*.safetensors"))
	if err != nil { return err }
	if len(weights) == 0 { return errors.New("HF source must contain safetensors weights") }
	return nil
}

func sourceFingerprint(directory string) (string, error) {
	items, err := os.ReadDir(directory)
	if err != nil { return "", err }
	combined := sha256.New()
	for _, item := range items {
		if item.IsDir() { continue }
		if !item.Type().IsRegular() { return "", fmt.Errorf("source contains non-regular file: %s", item.Name()) }
		checksum, size, err := digest(filepath.Join(directory, item.Name()))
		if err != nil { return "", err }
		fmt.Fprintf(combined, "%s:%d:%s\n", item.Name(), size, checksum)
	}
	return hex.EncodeToString(combined.Sum(nil)), nil
}

func digest(path string) (string, int64, error) {
	file, err := os.Open(path)
	if err != nil { return "", 0, err }
	defer file.Close()
	info, err := file.Stat()
	if err != nil { return "", 0, err }
	if !info.Mode().IsRegular() { return "", 0, errors.New("artifact must be a regular file") }
	checksum := sha256.New()
	count, err := io.Copy(checksum, file)
	if err != nil { return "", 0, err }
	return hex.EncodeToString(checksum.Sum(nil)), count, nil
}

func validGGUF(path string) error {
	file, err := os.Open(path)
	if err != nil { return err }
	defer file.Close()
	var signature [4]byte
	if _, err := io.ReadFull(file, signature[:]); err != nil { return err }
	if string(signature[:]) != "GGUF" { return errors.New("artifact has no GGUF header") }
	return nil
}

func ensureArtifact(ctx context.Context, path string, generate func(string) error) error {
	marker := path + ".sha256"
	if _, err := os.Stat(path); err == nil {
		if err := validGGUF(path); err != nil { return err }
		want, err := os.ReadFile(marker)
		if err != nil { return fmt.Errorf("existing artifact has no checksum marker: %w", err) }
		got, _, err := digest(path)
		if err != nil { return err }
		if got != strings.TrimSpace(string(want)) { return errors.New("cached GGUF checksum mismatch") }
		return nil
	} else if !errors.Is(err, os.ErrNotExist) { return err }

	temporaryDir, err := os.MkdirTemp(filepath.Dir(path), ".gguf-stage-")
	if err != nil { return err }
	output := filepath.Join(temporaryDir, "model.gguf")
	if err := generate(output); err != nil { return err }
	if err := ctx.Err(); err != nil { return err }
	if err := validGGUF(output); err != nil { return err }
	hash, size, err := digest(output)
	if err != nil { return err }
	if size <= 4 { return errors.New("GGUF artifact is empty") }
	if err := os.Rename(output, path); err != nil { return err }
	if err := os.WriteFile(marker, []byte(hash+"\n"), 0o600); err != nil { return err }
	return os.Remove(temporaryDir)
}

func publish(modelsDir, name, source string) (Result, error) {
	unlock, err := lock(filepath.Join(modelsDir, ".publish.lock"))
	if err != nil { return Result{}, err }
	defer unlock()
	target := filepath.Join(modelsDir, name)
	if _, err := os.Lstat(target); err == nil { return Result{}, fmt.Errorf("model already exists: %s", name) }
	if err != nil && !errors.Is(err, os.ErrNotExist) { return Result{}, err }

	// Stage in the parent of modelsDir: rename then stays on the same filesystem,
	// while an unfinished directory is never visible to the model scanner.
	stage, err := os.MkdirTemp(filepath.Dir(modelsDir), ".mini-ollama-publish-")
	if err != nil { return Result{}, err }
	fileName := name + ".Q4_K_M.gguf"
	output := filepath.Join(stage, fileName)
	input, err := os.Open(source)
	if err != nil { return Result{}, err }
	defer input.Close()
	destination, err := os.OpenFile(output, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil { return Result{}, err }
	_, copyErr := io.Copy(destination, input)
	closeErr := destination.Close()
	if copyErr != nil { return Result{}, copyErr }
	if closeErr != nil { return Result{}, closeErr }
	if err := validGGUF(output); err != nil { return Result{}, err }
	originalHash, originalSize, err := digest(source)
	if err != nil { return Result{}, err }
	outputHash, outputSize, err := digest(output)
	if err != nil { return Result{}, err }
	if originalHash != outputHash || originalSize != outputSize {
		return Result{}, errors.New("published copy differs from quantized GGUF")
	}
	metadata := manifest{Name: name, File: fileName, SizeBytes: outputSize, SHA256: outputHash}
	data, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil { return Result{}, err }
	if err := os.WriteFile(filepath.Join(stage, "manifest.json"), append(data, '\n'), 0o644); err != nil {
		return Result{}, err
	}
	if err := os.Rename(stage, target); err != nil { return Result{}, fmt.Errorf("publish model from %s: %w", stage, err) }
	return Result{Name: name, Path: filepath.Join(target, fileName), SizeBytes: outputSize, SHA256: outputHash}, nil
}
```

## 文件 3：`internal/hfimport/import_test.go`

这些测试只写 `t.TempDir()` 中的少量字节，不会访问 Hugging Face、加载模型或调用 GPU。

```go
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
		RepoID: "example/model",
		Revision: strings.Repeat("a", 40),
		Name: "example:1b",
		ModelsDir: "/outside/models",
		WorkDir: "/outside/work",
		Python: "python",
		Quantizer: "llama-quantize",
	}
	if err := validate(options); err != nil { t.Fatal(err) }
	options.Name = "../escape"
	if err := validate(options); err == nil { t.Fatal("accepted unsafe model name") }
	options.Name = "example:1b"
	options.Revision = "main"
	if err := validate(options); err == nil { t.Fatal("accepted movable revision") }
}

func TestEnsureArtifactReusesVerifiedFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "converted.gguf")
	called := 0
	generate := func(output string) error {
		called++
		return os.WriteFile(output, []byte("GGUFpayload"), 0o600)
	}
	if err := ensureArtifact(context.Background(), path, generate); err != nil { t.Fatal(err) }
	if err := ensureArtifact(context.Background(), path, generate); err != nil { t.Fatal(err) }
	if called != 1 { t.Fatalf("generate called %d times; want 1", called) }
	if err := os.WriteFile(path, []byte("GGUFchanged"), 0o600); err != nil { t.Fatal(err) }
	if err := ensureArtifact(context.Background(), path, generate); err == nil {
		t.Fatal("accepted modified cached GGUF")
	}
}

func TestPublishNeverOverwritesModel(t *testing.T) {
	root := t.TempDir()
	models := filepath.Join(root, "models")
	if err := os.Mkdir(models, 0o755); err != nil { t.Fatal(err) }
	source := filepath.Join(root, "source.gguf")
	if err := os.WriteFile(source, []byte("GGUFpayload"), 0o600); err != nil { t.Fatal(err) }
	result, err := publish(models, "demo:1b", source)
	if err != nil { t.Fatal(err) }
	data, err := os.ReadFile(filepath.Join(models, "demo:1b", "manifest.json"))
	if err != nil { t.Fatal(err) }
	var metadata manifest
	if err := json.Unmarshal(data, &metadata); err != nil { t.Fatal(err) }
	if metadata.Name != "demo:1b" || metadata.SizeBytes != result.SizeBytes || metadata.SHA256 != result.SHA256 {
		t.Fatalf("invalid manifest: %+v", metadata)
	}
	if _, err := publish(models, "demo:1b", source); err == nil {
		t.Fatal("existing model was overwritten")
	}
	unchanged, err := os.ReadFile(source)
	if err != nil { t.Fatal(err) }
	if string(unchanged) != "GGUFpayload" { t.Fatal("original weights were changed") }
}

func TestWithoutHFToken(t *testing.T) {
	result := withoutHFToken([]string{"PATH=/bin", "HF_TOKEN=secret"})
	if len(result) != 1 || result[0] != "PATH=/bin" { t.Fatalf("environment = %v", result) }
}
```

## 文件 4：`scripts/test_hf_snapshot.py`

这个无网络测试构造一个假的 Hugging Face 元数据对象，验证远端 SHA 不匹配时会被拒绝。只测本地校验逻辑，不调用 Hub。

```python
import hashlib
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from hf_snapshot import verify_tree


class VerifyTreeTest(unittest.TestCase):
    def test_default_endpoint_is_mirror(self):
        from hf_snapshot import DEFAULT_ENDPOINT
        self.assertEqual(DEFAULT_ENDPOINT, "https://hf-mirror.com")

    def test_endpoint_environment_override(self):
        import os
        from hf_snapshot import endpoint
        old_value = os.environ.get("HF_ENDPOINT")
        try:
            os.environ["HF_ENDPOINT"] = "https://example.invalid/"
            self.assertEqual(endpoint(), "https://example.invalid")
        finally:
            if old_value is None:
                os.environ.pop("HF_ENDPOINT", None)
            else:
                os.environ["HF_ENDPOINT"] = old_value

    def test_lfs_sha256(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "model.safetensors"
            path.write_bytes(b"sample weights")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            entry = SimpleNamespace(
                path=path.name,
                size=path.stat().st_size,
                lfs={"sha256": sha},
            )
            report = verify_tree(directory, [entry])
            self.assertEqual(report[0]["sha256"], sha)
            entry.lfs = {"sha256": "0" * 64}
            with self.assertRaises(ValueError):
                verify_tree(directory, [entry])


if __name__ == "__main__":
    unittest.main()
```

## 文件 5：`cmd/import_hf.go`

命令从当前仓库根目录运行，因而 Python 下载脚本与 llama.cpp 转换脚本可用相对路径定位。`--source-dir` 处理已有的本地 HF safetensors；远端模式则必须传 `--repo` 和完整 `--revision`。`--models-dir` 与前面的阶段保持一致。

```go
package cmd

import (
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"mini-ollama/internal/hfimport"

	"github.com/spf13/cobra"
)

func newImportHFCommand() *cobra.Command {
	options := hfimport.Options{
		Python: "python",
		Quantizer: filepath.Join("build", "llama-server", "bin", "llama-quantize"),
	}
	command := &cobra.Command{
		Use: "import-hf MODEL_NAME",
		Short: "Download, convert and quantize a Hugging Face model",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			options.Name = args[0]
			if options.ModelsDir == "" {
				var err error
				options.ModelsDir, err = defaultModelsDir()
				if err != nil { return err }
			}
			if options.WorkDir == "" { options.WorkDir = os.Getenv("MINI_OLLAMA_WORK_DIR") }
			if options.WorkDir == "" { return fmt.Errorf("--work-dir or MINI_OLLAMA_WORK_DIR is required") }
			options.Stdout = cmd.OutOrStdout()
			options.Stderr = cmd.ErrOrStderr()

			if _, err := os.Stat(options.Quantizer); err != nil {
				return fmt.Errorf("llama-quantize is unavailable: %w", err)
			}
			fmt.Fprintf(cmd.OutOrStdout(),
				"Source: repo=%s revision=%s local=%s\nCache: %s\nDestination: %s\n",
				options.RepoID, options.Revision, options.SourceDir,
				options.WorkDir, filepath.Join(options.ModelsDir, options.Name))

			ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt, syscall.SIGTERM)
			defer stop()
			result, err := hfimport.Import(ctx, options)
			if err != nil { return err }
			fmt.Fprintf(cmd.OutOrStdout(), "Published %s (%d bytes, SHA-256 %s)\n", result.Path, result.SizeBytes, result.SHA256)
			return nil
		},
	}
	flags := command.Flags()
	flags.StringVar(&options.RepoID, "repo", "", "Hugging Face OWNER/REPO")
	flags.StringVar(&options.Revision, "revision", "", "40-character Hugging Face commit SHA")
	flags.StringVar(&options.SourceDir, "source-dir", "", "existing local HF weights; disables download")
	flags.StringVar(&options.ModelsDir, "models-dir", "", "published GGUF directory")
	flags.StringVar(&options.WorkDir, "work-dir", "", "external download and conversion directory")
	flags.StringVar(&options.Python, "python", options.Python, "conda Python executable")
	flags.StringVar(&options.Quantizer, "quantizer", options.Quantizer, "llama-quantize executable")
	flags.StringVar(&options.TokenFile, "token-file", "", "path to HF token file; contents never printed")
	return command
}
```

## 文件 6：`cmd/cmd.go`

完成阶段 3 后，根命令完整内容如下。新增 `newImportHFCommand()`，其余命令沿用已有实现：

```go
package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

func NewCLI() *cobra.Command {
	rootCmd := &cobra.Command{
		Use: "mini-ollama",
		Short: "A command-line local model runner",
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) != 0 { return fmt.Errorf("unexpected arguments: %v", args) }
			return cmd.Help()
		},
	}
	rootCmd.AddCommand(
		newVersionCommand(),
		newServeCommand(),
		newRunCommand(),
		newModelsCommand(),
		newChatCommand(),
		newStopCommand(),
		newSwitchCommand(),
		newImportHFCommand(),
	)
	return rootCmd
}
```

## 格式化和不加载模型的测试

全部源文件输入后在仓库根目录运行：

```bash
gofmt -w cmd/import_hf.go cmd/cmd.go internal/hfimport/import.go internal/hfimport/import_test.go
python -m unittest discover -s scripts -p 'test_hf_snapshot.py'
go test ./...
go test -race ./internal/hfimport
go vet ./...
mkdir -p /rtai_cephfs/liangjm/models/.cache/mini-ollama-bin
go build -o /rtai_cephfs/liangjm/models/.cache/mini-ollama-bin/mini-ollama-stage4 .
```

`go test ./...` 依赖你已经完成阶段 2、3 的其他代码；本阶段的无网络 Go 测试可单独运行：

```bash
go test ./internal/hfimport
```

文档写入时，Go 文件还未输入，且你正在输入前一阶段，所以上面是**输入完成后的验证命令**，不是已经通过的测试结果。

## 运行前的明确模型和路径

最小下载与转换验收使用公开模型 `HuggingFaceTB/SmolLM2-135M-Instruct`，固定 commit `12fd25f77366fa6b3b4b768ec3050bf629380bac`。在本机运行后：

- 原始 HF 权重：`/rtai_cephfs/liangjm/models/.cache/mini-ollama-import/<job>/snapshot/`；
- 中间 F16 与 Q4_K_M GGUF：同一 `<job>/` 工作目录；
- 最终文件：`/rtai_cephfs/liangjm/models/GGUF/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-GGUF.Q4_K_M.gguf`；
- 最终校验：同目录 `manifest.json` 的 `size_bytes` 和 `sha256`。

`<job>` 是模型名和来源的短哈希构成的稳定目录名；正常的网络中断可以用相同工作目录重试，以复用已经校验且未变化的下载文件。若日志出现 `OSError: ... requested and 0 written`、`short write` 或转换进程在写 GGUF 时退出，先把这次任务视为中间文件不完整，改用新的 CephFS 工作目录重试，不要继续使用该目录中的半成品。发布成功后再运行会因目标已存在而直接报错，不再下载。第一次执行前确认模型仓库与本机目录；这一小模型只用于功能验收，不替代项目要求的 1B 与 8B 推理验收。

例如本机一次写入失败后，可创建新的 CephFS 工作目录再试。下面的目录名只是示例，日期或序号可以按实际重试次数修改；两个命令都不会把大文件写入仓库或本机 `/tmp`：

```bash
retry_work=/rtai_cephfs/liangjm/models/.cache/mini-ollama-import-retry-20260921
mkdir -p "$retry_work"
df -h /rtai_cephfs/liangjm/models
```

旧目录中的失败任务目录先保留用于排查。确认不再需要后，只删除明确的 `.gguf-stage-*` 子目录，不要删除整个模型缓存根目录或其他模型的目录。

以下命令是手工执行的下载、转换和量化入口；**本文档创建过程不会运行它**：

```bash
source ~/.bashrc
conda activate mini-ollama-hf
/rtai_cephfs/liangjm/models/.cache/mini-ollama-bin/mini-ollama-stage4 import-hf SmolLM2-135M-Instruct-GGUF \
  --repo HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --models-dir /rtai_cephfs/liangjm/models/GGUF \
  --work-dir /rtai_cephfs/liangjm/models/.cache/mini-ollama-import-retry-20260921 \
  --python "$(command -v python)" \
  --quantizer build/llama-server/bin/llama-quantize
```

需要认证的 repo 可在上述命令尾部加 `--token-file /path/to/private/token-file`。此参数仅提供文件路径；程序不会输出其内容。已经存在的原始 safetensors 则可完全离线转换，本机示例的来源是 `/rtai_cephfs/liangjm/models/Qwen2.5-7B-Instruct/`，最终发布为新名称且不会覆盖已有 GGUF：

```bash
conda activate mini-ollama-hf
/rtai_cephfs/liangjm/models/.cache/mini-ollama-bin/mini-ollama-stage4 import-hf Qwen2.5-7B-Instruct-local-GGUF \
  --source-dir /rtai_cephfs/liangjm/models/Qwen2.5-7B-Instruct \
  --models-dir /rtai_cephfs/liangjm/models/GGUF \
  --work-dir /rtai_cephfs/liangjm/models/.cache/mini-ollama-import-retry-20260921 \
  --python "$(command -v python)" \
  --quantizer build/llama-server/bin/llama-quantize
```

本地权重模式只证明输入可转换，并记录生成的 GGUF 校验值；它不声称本地源文件已对照 Hugging Face 的远端 LFS SHA-256 验证。若使用本地源文件且需要来源级校验，应先取得该版本可靠的远端校验信息。

导入完成后手工验证输出：

```bash
/rtai_cephfs/liangjm/models/.cache/mini-ollama-bin/mini-ollama-stage4 models \
  --models-dir /rtai_cephfs/liangjm/models/GGUF --verify
```

再次强调：导入不会自动启动 llama-server 或加载 GPU 模型。要手工做推理验收，先告知所用模型与最终 GGUF 路径，再使用 `serve` 或 `switch`；当前运行中的 `serve` 需重启才能重新扫描到刚发布的模型。

## 完成标准

- 下载固定 commit，原始权重在仓库外，重复执行可复用下载缓存；
- 下载的每个 safetensors 权重都有远端 LFS 大小与 SHA-256 校验；
- 转换器和量化工具均来自当前 llama.cpp 检出与构建；
- HF 原始权重、F16 中间产物和已有 GGUF 不被覆盖；
- 最终 GGUF 的大小与 SHA-256 记录在 manifest 中，并能由阶段 3 的 `models --verify` 通过；
- Ctrl-C 中断后保留缓存，下一次导入能安全重试；
- Python 单测、Go 单测、race 检查、`go vet` 与构建通过；
- 手工选择模型及路径后完成一次公开小模型从下载到发布的端到端验收。

下载与远端校验采用 Hugging Face 的 [snapshot_download 文档](https://huggingface.co/docs/huggingface_hub/guides/download) 和 [RepoFile 元数据文档](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)；SmolLM2 的示例 commit 来自[该模型的官方仓库元数据](https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M-Instruct)。
