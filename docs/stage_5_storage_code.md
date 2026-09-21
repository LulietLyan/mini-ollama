# 阶段 5 存储和导入完整代码

SQLite 使用单连接和 `DELETE` journal，不在 CephFS 上启用 WAL。HF 导入代码清理失败的 stage 目录，并原子写入 checksum marker。模型原始权重、转换中间文件和 GGUF 仍然放在 CephFS 工作目录。

## 文件：`internal/history/store.go`

```go
package history

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	_ "modernc.org/sqlite"
)

// Message 关联一个 Conversation，因此有 ConversationID 外键约束
type Message struct {
	ID             int64     `json:"id"`
	ConversationID string    `json:"conversation_id"`
	Role           string    `json:"role"`
	Content        string    `json:"content"`
	CreatedAt      time.Time `json:"created_at"`
}

type Conversation struct {
	ID        string    `json:"id"`
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	Messages  []Message `json:"messages"`
}

type Store struct {
	db *sql.DB
}

func Open(path string) (*Store, error) {
	if path == "" {
		return nil, errors.New("database path is empty")
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("create data directory: %w", err)
	}

	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open sqlite database: %w", err)
	}

	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)

	store := &Store{db: db}
	if err := store.migrate(context.Background()); err != nil {
		_ = db.Close()
		return nil, err
	}
	return store, nil
}

func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

func (s *Store) CreateConversation(ctx context.Context, model string) (Conversation, error) {
	id, err := newID()
	if err != nil {
		return Conversation{}, err
	}

	now := time.Now().UTC()
	stamp := now.Format(time.RFC3339Nano)
	_, err = s.db.ExecContext(ctx,
		`INSERT INTO conversations(id, model, created_at, updated_at)
		 VALUES (?, ?, ?, ?)`, id, model, stamp, stamp)
	if err != nil {
		return Conversation{}, fmt.Errorf("create conversation: %w", err)
	}
	return Conversation{ID: id, Model: model, CreatedAt: now, UpdatedAt: now}, nil
}

func (s *Store) ListConversations(ctx context.Context) ([]Conversation, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, model, created_at, updated_at
		 FROM conversations ORDER BY updated_at DESC`)
	if err != nil {
		return nil, fmt.Errorf("list conversations: %w", err)
	}
	defer rows.Close()
	result := make([]Conversation, 0)
	for rows.Next() {
		conversation, err := scanConversation(rows)
		if err != nil {
			return nil, err
		}
		result = append(result, conversation)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read conversations: %w", err)
	}
	return result, nil
}

func (s *Store) GetConversation(ctx context.Context, id string) (Conversation, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, model, created_at, updated_at
		 FROM conversations WHERE id = ?`, id)
	conversation, err := scanConversation(row)
	if errors.Is(err, sql.ErrNoRows) {
		return Conversation{}, fmt.Errorf("conversation not found: %s", id)
	}
	if err != nil {
		return Conversation{}, fmt.Errorf("get conversation: %w", err)
	}
	conversation.Messages, err = s.Messages(ctx, id)
	if err != nil {
		return Conversation{}, err
	}
	return conversation, nil
}

func (s *Store) DeleteConversation(ctx context.Context, id string) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin deletion: %w", err)
	}
	defer tx.Rollback()
	if _, err := tx.ExecContext(ctx,
		`DELETE FROM messages WHERE conversation_id = ?`, id); err != nil {
		return fmt.Errorf("delete messages: %w", err)
	}
	result, err := tx.ExecContext(ctx,
		`DELETE FROM conversations WHERE id = ?`, id)
	if err != nil {
		return fmt.Errorf("delete conversation: %w", err)
	}
	count, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("check deleted conversation: %w", err)
	}
	if count == 0 {
		return fmt.Errorf("conversation not found: %s", id)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit deletion: %w", err)
	}
	return nil
}

// 在 message 表中为属于 conversation_id 的对话插入一条新的消息
// 同时要更新 conversation 表中对应对话的更新时间
func (s *Store) AppendMessage(ctx context.Context, conversationID, role, content string) (Message, error) {
	if role != "system" && role != "user" && role != "assistant" {
		return Message{}, fmt.Errorf("unsupported message role: %s", role)
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return Message{}, fmt.Errorf("begin message append: %w", err)
	}
	defer tx.Rollback()
	now := time.Now().UTC()
	result, err := tx.ExecContext(ctx,
		`INSERT INTO messages(conversation_id, role, content, created_at)
		 VALUES (?, ?, ?, ?)`, conversationID, role, content, now.Format(time.RFC3339Nano))
	if err != nil {
		return Message{}, fmt.Errorf("append message: %w", err)
	}
	id, err := result.LastInsertId()
	if err != nil {
		return Message{}, fmt.Errorf("read message id: %w", err)
	}
	if _, err := tx.ExecContext(ctx,
		`UPDATE conversations SET updated_at = ? WHERE id = ?`, now.Format(time.RFC3339Nano), conversationID); err != nil {
		return Message{}, fmt.Errorf("update conversation: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return Message{}, fmt.Errorf("commit message: %w", err)
	}
	return Message{ID: id, ConversationID: conversationID, Role: role, Content: content, CreatedAt: now}, nil
}

// 在 message 表中查找属于 conversation_id 的消息
func (s *Store) Messages(ctx context.Context, conversationID string) ([]Message, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, conversation_id, role, content, created_at
		 FROM messages WHERE conversation_id = ? ORDER BY id`, conversationID)
	if err != nil {
		return nil, fmt.Errorf("list messages: %w", err)
	}
	defer rows.Close()
	result := make([]Message, 0)
	for rows.Next() {
		var message Message
		var createdAt string
		if err := rows.Scan(&message.ID, &message.ConversationID, &message.Role, &message.Content, &createdAt); err != nil {
			return nil, fmt.Errorf("scan message: %w", err)
		}
		message.CreatedAt, err = time.Parse(time.RFC3339Nano, createdAt)
		if err != nil {
			return nil, fmt.Errorf("parse message time: %w", err)
		}
		result = append(result, message)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read messages: %w", err)
	}
	return result, nil
}

// 初始化 SQLite 数据库
func (s *Store) migrate(ctx context.Context) error {
	pragmas := []string{
		`PRAGMA foreign_keys = ON`,
		`PRAGMA busy_timeout = 5000`,
		`PRAGMA journal_mode = DELETE`,
	}
	for _, statement := range pragmas {
		if _, err := s.db.ExecContext(ctx, statement); err != nil {
			return fmt.Errorf("set sqlite pragma: %w", err)
		}
	}
	_, err := s.db.ExecContext(ctx, `
		CREATE TABLE IF NOT EXISTS conversations(
			id TEXT PRIMARY KEY,
			model TEXT NOT NULL,
			created_at TEXT NOT NULL,
			updated_at TEXT NOT NULL
		);
		CREATE TABLE IF NOT EXISTS messages(
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			conversation_id TEXT NOT NULL,
			role TEXT NOT NULL,
			content TEXT NOT NULL,
			created_at TEXT NOT NULL,
			FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
		);
		CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
		ON messages(conversation_id, id);
	`)
	if err != nil {
		return fmt.Errorf("migrate database: %w", err)
	}
	return nil
}

type rowScanner interface{ Scan(dest ...any) error }

func scanConversation(row rowScanner) (Conversation, error) {
	var conversation Conversation
	var createdAt, updatedAt string
	if err := row.Scan(&conversation.ID, &conversation.Model, &createdAt, &updatedAt); err != nil {
		return Conversation{}, err
	}
	var err error
	conversation.CreatedAt, err = time.Parse(time.RFC3339Nano, createdAt)
	if err != nil {
		return Conversation{}, err
	}
	conversation.UpdatedAt, err = time.Parse(time.RFC3339Nano, updatedAt)
	if err != nil {
		return Conversation{}, err
	}
	return conversation, nil
}

func newID() (string, error) {
	buffer := make([]byte, 16)
	if _, err := rand.Read(buffer); err != nil {
		return "", err
	}
	return hex.EncodeToString(buffer), nil
}
```

## 文件：`internal/history/store_test.go`

```go
package history

import (
	"context"
	"path/filepath"
	"testing"
)

func TestOpenSQLite(t *testing.T) {
	store, err := Open(filepath.Join(t.TempDir(), "history.db"))
	if err != nil {
		t.Fatalf("open SQLite: %v", err)
	}
	defer store.Close()

	if _, err := store.CreateConversation(context.Background(), "test-model"); err != nil {
		t.Fatalf("create conversation: %v", err)
	}
}
```

## 文件：`internal/hfimport/import.go`

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
```

## 文件：`internal/hfimport/import_test.go`

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
```

## 验证

```bash
gofmt -w internal/history/*.go internal/hfimport/*.go
go test ./internal/history ./internal/hfimport
go test -race ./internal/history ./internal/hfimport
```
