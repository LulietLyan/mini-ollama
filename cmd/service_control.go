package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/spf13/cobra"
)

func newStopCommand() *cobra.Command {
	var serverURL string
	command := &cobra.Command{
		Use:   "stop",
		Short: "Stop the loaded model",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return postServiceControl(cmd.Context(), serverURL, "/api/v1/service/stop", nil)
		},
	}

	command.Flags().StringVar(&serverURL, "server", "http://127.0.0.1:11434", "mini-ollama HTTP server URL")
	return command
}

func newSwitchCommand() *cobra.Command {
	var serverURL string
	command := &cobra.Command{
		Use:   "switch MODEL",
		Short: "Explicitly switch the loaded model",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			payload, err := json.Marshal(map[string]string{"model": args[0]})
			if err != nil {
				return err
			}
			return postServiceControl(cmd.Context(), serverURL, "/api/v1/service/switch", payload)
		},
	}

	command.Flags().StringVar(&serverURL, "server", "http://127.0.0.1:11434", "mini-ollama HTTP server URL")
	return command
}

func postServiceControl(ctx context.Context, baseURL, path string, payload []byte) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(baseURL, "/")+path, bytes.NewReader(payload))
	if err != nil {
		return err
	}

	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("service control returned HTTP %d", response.StatusCode)
	}
	return nil
}
