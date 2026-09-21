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
