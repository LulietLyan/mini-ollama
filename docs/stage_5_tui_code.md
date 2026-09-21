# 阶段 5 TUI 完整代码

TUI 使用 Bubble Tea，但只调用 mini-ollama HTTP API。它不启动 `llama-server`，模型切换必须经过 `/api/v1/service/switch`。

## 文件：`cmd/tui.go`

```go
package cmd

import (
	"mini-ollama/internal/api"
	"mini-ollama/internal/tui"

	"github.com/spf13/cobra"
)

func newTUICommand() *cobra.Command {
	var serverURL string
	command := &cobra.Command{
		Use:   "tui",
		Short: "Open the terminal UI for a running mini-ollama server",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			client := &api.Client{BaseURL: serverURL}
			return tui.Run(cmd.Context(), client)
		},
	}
	command.Flags().StringVar(&serverURL, "server", "http://127.0.0.1:11434", "mini-ollama HTTP server URL")
	return command
}
```

## 文件：`internal/tui/tui.go`

```go
package tui

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"mini-ollama/internal/api"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// Run starts the terminal UI against an already running mini-ollama API.
// The API remains the source of truth for model lifecycle and chat history.
func Run(ctx context.Context, client *api.Client) error {
	if client == nil {
		return fmt.Errorf("tui API client is nil")
	}
	model := newModel(ctx, client)
	program := tea.NewProgram(model, tea.WithAltScreen())
	model.program = program

	go func() {
		<-ctx.Done()
		program.Quit()
	}()

	_, err := program.Run()
	return err
}

type mode uint8

const (
	modeSelect mode = iota
	modeChat
)

type model struct {
	ctx     context.Context
	client  *api.Client
	program *tea.Program

	mode           mode
	models         []api.Model
	selected       int
	status         api.Status
	conversationID string
	pendingMessage string
	messages       []message
	input          textinput.Model
	width          int
	height         int
	busy           bool
	refreshing     bool
	err            error
	chatCancel     context.CancelFunc

	// program.Send is safe from the SSE worker goroutine. The mutex only
	// protects replacement of the cancel function during a new request.
	mu sync.Mutex
}

type message struct {
	role string
	text string
}

type modelsMsg struct {
	models []api.Model
	status api.Status
	err    error
}

type statusMsg struct {
	status api.Status
	err    error
}

type serviceMsg struct {
	status api.Status
	err    error
}

type conversationMsg struct {
	id  string
	err error
}

type chatDeltaMsg string

type chatDoneMsg struct {
	err error
}

type refreshTickMsg struct{}

func newModel(ctx context.Context, client *api.Client) *model {
	input := textinput.New()
	input.Prompt = "> "
	input.Placeholder = "Type a message and press Enter"
	input.CharLimit = 16 * 1024
	input.Width = 60

	return &model{
		ctx:    ctx,
		client: client,
		mode:   modeSelect,
		input:  input,
	}
}

func (m *model) Init() tea.Cmd {
	return tea.Batch(m.loadModels(), refreshEverySecond())
}

func (m *model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch value := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = value.Width, value.Height
		m.input.Width = max(20, value.Width/2-8)
		return m, nil
	case tea.KeyMsg:
		return m.updateKey(value)
	case modelsMsg:
		m.refreshing = false
		m.err = value.err
		if value.err == nil {
			m.models = value.models
			m.keepSelection()
			m.status = value.status
			if m.status.Model != "" {
				m.mode = modeChat
				m.input.Focus()
			}
		}
		return m, nil
	case statusMsg:
		m.status = value.status
		if value.err != nil {
			m.err = value.err
		}
		if m.status.Lifecycle == "stopped" && !m.busy {
			m.conversationID = ""
			m.mode = modeSelect
			m.input.Blur()
		}
		return m, nil
	case serviceMsg:
		m.busy = false
		m.err = value.err
		if value.err == nil {
			m.status = value.status
			if m.status.Model != "" && m.status.Lifecycle == "ready" {
				m.mode = modeChat
				m.input.Focus()
				m.conversationID = ""
				m.messages = nil
			}
		}
		return m, nil
	case conversationMsg:
		if value.err != nil {
			m.busy = false
			m.err = value.err
			return m, nil
		}
		m.conversationID = value.id
		text := m.pendingMessage
		m.pendingMessage = ""
		return m, m.startChat(value.id, text)
	case chatDeltaMsg:
		m.ensureAssistantMessage()
		m.messages[len(m.messages)-1].text += string(value)
		return m, nil
	case chatDoneMsg:
		m.mu.Lock()
		m.chatCancel = nil
		m.mu.Unlock()
		m.busy = false
		m.err = value.err
		m.input.Focus()
		return m, nil
	case refreshTickMsg:
		if m.ctx.Err() != nil {
			return m, tea.Quit
		}
		return m, tea.Batch(m.loadStatus(), refreshEverySecond())
	}

	if m.mode == modeChat && !m.busy {
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		return m, cmd
	}
	return m, nil
}

func (m *model) updateKey(key tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch key.String() {
	case "ctrl+c":
		m.cancelChatRequest()
		return m, tea.Quit
	case "ctrl+r":
		if !m.refreshing && !m.busy {
			m.refreshing = true
			return m, m.loadModels()
		}
		return m, nil
	case "ctrl+s":
		if !m.busy && m.status.Lifecycle != "stopped" {
			m.busy = true
			return m, m.stopService()
		}
		return m, nil
	case "esc":
		if m.mode == modeChat && !m.busy {
			m.mode = modeSelect
			m.input.Blur()
		}
		return m, nil
	case "up", "k":
		if m.mode == modeSelect && !m.busy && len(m.models) > 0 {
			m.selected = (m.selected + len(m.models) - 1) % len(m.models)
		}
		return m, nil
	case "down", "j":
		if m.mode == modeSelect && !m.busy && len(m.models) > 0 {
			m.selected = (m.selected + 1) % len(m.models)
		}
		return m, nil
	case "enter":
		if m.mode == modeSelect {
			return m, m.selectModel()
		}
	}

	if m.mode == modeChat && !m.busy {
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(key)
		if key.String() == "enter" {
			return m, m.submitMessage()
		}
		return m, cmd
	}
	return m, nil
}

func (m *model) loadModels() tea.Cmd {
	return func() tea.Msg {
		models, err := m.client.RefreshModels(m.ctx)
		if err != nil {
			return modelsMsg{err: err}
		}
		status, err := m.client.Status(m.ctx)
		return modelsMsg{models: models, status: status, err: err}
	}
}

func (m *model) loadStatus() tea.Cmd {
	return func() tea.Msg {
		status, err := m.client.Status(m.ctx)
		return statusMsg{status: status, err: err}
	}
}

func (m *model) selectModel() tea.Cmd {
	if len(m.models) == 0 || m.busy {
		return nil
	}
	selected := m.models[m.selected].Name
	if m.status.Lifecycle == "ready" && m.status.Model == selected {
		m.mode = modeChat
		m.input.Focus()
		return nil
	}
	m.busy = true
	return func() tea.Msg {
		err := m.client.Switch(m.ctx, selected)
		status, statusErr := m.client.Status(m.ctx)
		if err == nil {
			err = statusErr
		}
		return serviceMsg{status: status, err: err}
	}
}

func (m *model) stopService() tea.Cmd {
	return func() tea.Msg {
		err := m.client.Stop(m.ctx)
		status, statusErr := m.client.Status(m.ctx)
		if err == nil {
			err = statusErr
		}
		return serviceMsg{status: status, err: err}
	}
}

func (m *model) submitMessage() tea.Cmd {
	text := strings.TrimSpace(m.input.Value())
	if text == "" || m.busy || m.status.Lifecycle != "ready" || m.status.Model == "" {
		return nil
	}
	m.input.Reset()
	m.pendingMessage = text
	m.messages = append(m.messages, message{role: "you", text: text})
	m.busy = true
	if m.conversationID == "" {
		return m.createConversation(text)
	}
	m.pendingMessage = ""
	return m.startChat(m.conversationID, text)
}

func (m *model) createConversation(text string) tea.Cmd {
	modelName := m.status.Model
	return func() tea.Msg {
		conversation, err := m.client.CreateConversation(m.ctx, modelName)
		if err != nil {
			return conversationMsg{err: err}
		}
		return conversationMsg{id: conversation.ID}
	}
}

func (m *model) startChat(conversationID, text string) tea.Cmd {
	if strings.TrimSpace(text) == "" {
		return nil
	}
	ctx, cancel := context.WithCancel(m.ctx)
	m.mu.Lock()
	m.chatCancel = cancel
	m.mu.Unlock()
	modelName := m.status.Model
	if m.program == nil {
		cancel()
		return func() tea.Msg { return chatDoneMsg{err: fmt.Errorf("tui program is not initialized")} }
	}

	return func() tea.Msg {
		err := m.client.ChatStream(ctx, modelName, conversationID, text, func(delta string) error {
			m.program.Send(chatDeltaMsg(delta))
			return nil
		})
		cancel()
		return chatDoneMsg{err: err}
	}
}

func (m *model) cancelChatRequest() {
	m.mu.Lock()
	cancel := m.chatCancel
	m.chatCancel = nil
	m.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (m *model) keepSelection() {
	if len(m.models) == 0 {
		m.selected = 0
		return
	}
	if m.selected >= len(m.models) {
		m.selected = len(m.models) - 1
	}
}

func (m *model) ensureAssistantMessage() {
	if len(m.messages) == 0 || m.messages[len(m.messages)-1].role != "assistant" {
		m.messages = append(m.messages, message{role: "assistant"})
	}
}

func (m *model) View() string {
	if m.width == 0 {
		return "Loading mini-ollama..."
	}

	title := titleStyle.Render("mini-ollama")
	status := m.status.Lifecycle
	if status == "" {
		status = "connecting"
	}
	statusLine := fmt.Sprintf("status: %s", status)
	if m.status.Model != "" {
		statusLine += "  model: " + m.status.Model
	}

	left := m.renderModels()
	right := m.renderChat()
	columns := lipgloss.JoinHorizontal(lipgloss.Top, left, right)

	help := "↑/↓ select  enter load/send  esc models  ctrl+s stop  ctrl+r refresh  ctrl+c quit"
	if m.err != nil {
		help = errorStyle.Render(m.err.Error()) + "  " + help
	}
	return title + "  " + statusLine + "\n\n" + columns + "\n\n" + help
}

func (m *model) renderModels() string {
	width := max(26, m.width/3)
	lines := []string{sectionStyle.Render("Models")}
	if len(m.models) == 0 {
		lines = append(lines, mutedStyle.Render("No GGUF models found"))
	} else {
		for index, entry := range m.models {
			prefix := "  "
			if index == m.selected && m.mode == modeSelect {
				prefix = "> "
			}
			name := entry.Name
			if index == m.selected && m.mode == modeSelect {
				name = selectedStyle.Render(name)
			}
			lines = append(lines, prefix+name)
			if entry.SizeBytes > 0 {
				lines = append(lines, mutedStyle.Render("    "+formatBytes(entry.SizeBytes)))
			}
		}
	}
	return panelStyle.Width(width).Render(strings.Join(lines, "\n"))
}

func (m *model) renderChat() string {
	width := max(42, m.width-m.width/3-4)
	lines := []string{sectionStyle.Render("Chat")}
	if m.status.Model == "" {
		lines = append(lines, mutedStyle.Render("Select a model and press Enter to load it."))
	} else if len(m.messages) == 0 {
		lines = append(lines, mutedStyle.Render("Model ready. Type a message below."))
	} else {
		for _, item := range m.messages {
			label := "You"
			if item.role == "assistant" {
				label = "Assistant"
			}
			lines = append(lines, sectionStyle.Render(label)+"\n"+item.text)
		}
	}
	if m.mode == modeChat {
		lines = append(lines, "", m.input.View())
	}
	return panelStyle.Width(width).Render(strings.Join(lines, "\n"))
}

func refreshEverySecond() tea.Cmd {
	return tea.Tick(time.Second, func(time.Time) tea.Msg { return refreshTickMsg{} })
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func formatBytes(value int64) string {
	if value < 1024*1024*1024 {
		return strconv.FormatInt(value/(1024*1024), 10) + " MiB"
	}
	return strconv.FormatInt(value/(1024*1024*1024), 10) + " GiB"
}

var (
	titleStyle    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("205"))
	sectionStyle  = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("69"))
	selectedStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("212"))
	mutedStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("241"))
	errorStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("196"))
	panelStyle    = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).Padding(1, 2).Align(lipgloss.Left)
)
```

## 文件：`internal/tui/tui_test.go`

```go
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
```

## 文件：`cmd/cmd.go`

```go
package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

func NewCLI() *cobra.Command {
	// 默认无参数状态下显示帮助信息
	rootCmd := &cobra.Command{
		Use:   "mini-ollama",
		Short: "A command-line local model runner",
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) != 0 {
				return fmt.Errorf("unexpected arguments: %v", args)
			}
			return cmd.Help()
		},
	}

	rootCmd.AddCommand(
		// 显示 llama server 版本
		newVersionCommand(),
		// 运行指定的模型并开启服务
		newServeCommand(),
		// 当前提供的模型
		newModelsCommand(),
		// 通过 mini-ollama 服务进行对话
		newChatCommand(),
		// 通过 mini-ollama HTTP API 使用终端 UI
		newTUICommand(),
		// 停止当前服务
		newStopCommand(),
		// 切换提供服务的模型
		newSwitchCommand(),
		// 运行指定的模型
		newRunCommand(),
		// 拉取 HuggingFace 模型
		newImportHFCommand(),
	)

	return rootCmd
}
```

## 验证

```bash
gofmt -w cmd/tui.go cmd/cmd.go internal/tui/*.go
go test ./internal/tui ./cmd
go test -race ./internal/tui ./cmd
```

## 启动

先启动 `serve`，再在另一个终端运行：

```bash
/tmp/mini-ollama-stage5 tui --server http://127.0.0.1:11434
```
