package tui

import (
	"context"
	"strings"
	"testing"

	"mini-ollama/internal/api"

	tea "github.com/charmbracelet/bubbletea"
)

func TestModelSelectionAndView(t *testing.T) {
	m := newModel(context.Background(), &api.Client{BaseURL: "http://127.0.0.1:11434"})
	m.width, m.height = 100, 30
	m.models = []api.Model{{Name: "alpha"}, {Name: "beta"}}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyDown})
	view := updated.(*model)
	if view.selected != 1 {
		t.Fatalf("selected = %d, want 1", view.selected)
	}
	if !strings.Contains(view.View(), "beta") {
		t.Fatalf("view does not contain selected model: %s", view.View())
	}
}

func TestSubmitMessageRequiresReadyModel(t *testing.T) {
	m := newModel(context.Background(), &api.Client{BaseURL: "http://127.0.0.1:11434"})
	m.input.SetValue("hello")
	if command := m.submitMessage(); command != nil {
		t.Fatal("submitMessage returned a command while no model was ready")
	}
}
