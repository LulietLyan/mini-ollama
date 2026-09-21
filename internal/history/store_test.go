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
