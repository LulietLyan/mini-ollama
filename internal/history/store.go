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
	Messages  []Message `json:"message"`
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
