package cmd

import (
	"os/exec"

	"github.com/spf13/cobra"
)

func newVersionCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Show llama-server version",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			serverPath, err := llamaServerPath()
			if err != nil {
				return err
			}

			process := exec.CommandContext(
				cmd.Context(),
				serverPath,
				"--version",
			)

			process.Stdin = cmd.InOrStdin()
			process.Stdout = cmd.OutOrStdout()
			process.Stderr = cmd.ErrOrStderr()

			return process.Run()
		},
	}
}
