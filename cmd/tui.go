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
