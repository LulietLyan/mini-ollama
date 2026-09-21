package cmd

import (
	"fmt"
	"mini-ollama/internal/hfimport"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/spf13/cobra"
)

func newImportHFCommand() *cobra.Command {
	options := hfimport.Options{
		Python:    "python",
		Quantizer: filepath.Join("build", "llama-server", "bin", "llama-quantize"),
	}

	command := &cobra.Command{
		Use:   "import-hf MODEL_NAME",
		Short: "Download, convert and quantize a Hugging Face model",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			options.Name = args[0]
			if options.ModelsDir == "" {
				var err error
				options.ModelsDir, err = defaultModelsDir()
				if err != nil {
					return err
				}
			}
			if options.WorkDir == "" {
				options.WorkDir = os.Getenv("MINI_OLLAMA_WORK_DIR")
			}
			if options.WorkDir == "" {
				return fmt.Errorf("--work-dir or MINI_OLLAMA_WORK_DIR is required")
			}
			options.Stdout = cmd.OutOrStdout()
			options.Stderr = cmd.ErrOrStderr()

			if _, err := os.Stat(options.Quantizer); err != nil {
				return fmt.Errorf("llama-quantize is unavailable: %w", err)
			}

			fmt.Fprintf(cmd.OutOrStdout(),
				"Source: repo=%s revision=%s local=%s\nCache: %s\nDestination: %s\n",
				options.RepoID, options.Revision, options.SourceDir,
				options.WorkDir, filepath.Join(options.ModelsDir, options.Name))

			ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt, syscall.SIGTERM)
			defer stop()

			result, err := hfimport.Import(ctx, options)
			if err != nil {
				return err
			}

			fmt.Fprintf(cmd.OutOrStdout(), "Published %s (%d bytes, SHA-256 %s)\n", result.Path, result.SizeBytes, result.SHA256)
			return nil
		},
	}

	flags := command.Flags()
	flags.StringVar(&options.RepoID, "repo", "", "Hugging Face OWNER/REPO")
	flags.StringVar(&options.Revision, "revision", "", "40-character Hugging Face commit SHA")
	flags.StringVar(&options.SourceDir, "source-dir", "", "existing local HF weights; disables download")
	flags.StringVar(&options.ModelsDir, "models-dir", "", "published GGUF directory")
	flags.StringVar(&options.WorkDir, "work-dir", "", "external download and conversion directory")
	flags.StringVar(&options.Python, "python", options.Python, "conda Python executable")
	flags.StringVar(&options.Quantizer, "quantizer", options.Quantizer, "llama-quantize executable")
	flags.StringVar(&options.TokenFile, "token-file", "", "path to HF token file; contents never printed")

	return command
}
