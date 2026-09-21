package cmd

import (
	"encoding/json"
	"fmt"
	"mini-ollama/internal/catalog"
	"text/tabwriter"

	"github.com/spf13/cobra"
)

func newModelsCommand() *cobra.Command {
	var modelsDir string
	var asJSON bool
	var verify bool
	var refresh bool

	command := &cobra.Command{
		Use:   "models",
		Short: "List local models",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			if modelsDir == "" {
				var err error
				modelsDir, err = defaultModelsDir()
				if err != nil {
					return err
				}
			}
			modelCatalog, err := catalog.New(modelsDir)
			if err != nil {
				return err
			}
			if refresh {
				if err := modelCatalog.Refresh(); err != nil {
					return err
				}
			}
			entries := modelCatalog.List()
			if verify {
				for _, entry := range entries {
					if err := modelCatalog.Verify(entry); err != nil {
						return err
					}
				}
			}
			if asJSON {
				return json.NewEncoder(cmd.OutOrStdout()).Encode(entries)
			}
			writer := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 4, 2, ' ', 0)
			fmt.Fprintln(writer, "NAME\tSIZE_BYTES\tDESCRIPTION")
			for _, entry := range entries {
				fmt.Fprintf(writer, "%s\t%d\t%s\n", entry.Name, entry.SizeBytes, entry.Description)
			}
			return writer.Flush()
		},
	}
	command.Flags().StringVar(&modelsDir, "models-dir", "", "model directory")
	command.Flags().BoolVar(&asJSON, "json", false, "print JSON")
	command.Flags().BoolVar(&verify, "verify", false, "verify manifest size and SHA-256")
	command.Flags().BoolVar(&refresh, "refresh", false, "rescan the model directory before listing")

	return command
}
