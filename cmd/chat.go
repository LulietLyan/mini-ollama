package cmd

import (
	"bufio"
	"fmt"
	"mini-ollama/internal/api"
	"os"
	"os/signal"
	"strings"

	"github.com/spf13/cobra"
)

func newChatCommand() *cobra.Command {
	var serverURL string
	command := &cobra.Command{
		Use:   "chat MODEL",
		Short: "Chat with a model managed by mini-ollama serve",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runChat(cmd, args[0], serverURL)
		},
	}

	command.Flags().StringVar(&serverURL, "server", "http://127.0.0.1:11434", "mini-ollama HTTP server URL")

	return command
}

func runChat(cmd *cobra.Command, model, serverURL string) error {
	ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt)
	defer stop()
	client := &api.Client{BaseURL: serverURL}
	conversation, err := client.CreateConversation(ctx, model)
	if err != nil {
		return fmt.Errorf("create conversation: %w", err)
	}
	fmt.Fprintf(cmd.OutOrStdout(), "conversation %s\n", conversation.ID)

	scanner := bufio.NewScanner(cmd.InOrStdin())
	for {
		if _, err := fmt.Fprint(cmd.OutOrStdout(), "> "); err != nil {
			return err
		}
		if !scanner.Scan() {
			break
		}
		message := strings.TrimSpace(scanner.Text())
		if message == "" {
			continue
		}
		if message == "/exit" || message == "/quit" {
			break
		}
		if err := client.ChatStream(ctx, model, conversation.ID, message, func(delta string) error {
			_, err := fmt.Fprint(cmd.OutOrStdout(), delta)
			return err
		}); err != nil {
			return fmt.Errorf("chat request: %w", err)
		}
		fmt.Fprintln(cmd.OutOrStdout())
	}

	return scanner.Err()
}
