package main

import (
	"context"
	"mini-ollama/cmd"

	"github.com/spf13/cobra"
)

func main() {
	cobra.CheckErr(cmd.NewCLI().ExecuteContext(context.Background()))
}
