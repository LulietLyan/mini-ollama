package runner

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"time"
)

type Config struct {
	Path   string
	Args   []string
	Env    []string
	Stdin  io.Reader
	Stdout io.Writer
	Stderr io.Writer
}

type Process struct {
	command *exec.Cmd
}

func Start(ctx context.Context, config Config) (*Process, error) {
	if config.Path == "" {
		return nil, errors.New("runner path is empty")
	}

	command := exec.CommandContext(
		ctx,
		config.Path,
		config.Args...,
	)

	if config.Env != nil {
		command.Env = config.Env
	}

	command.Stdin = config.Stdin
	command.Stdout = config.Stdout
	command.Stderr = config.Stderr

	if err := command.Start(); err != nil {
		return nil, err
	}

	return &Process{
		command: command,
	}, nil
}

func (p *Process) Wait() error {
	if p == nil || p.command == nil {
		return errors.New("runner process is nil")
	}

	return p.command.Wait()
}

type ReadinessConfig struct {
	URL      string
	Client   *http.Client
	Timeout  time.Duration
	Interval time.Duration
}

func WaitForHTTPReady(
	ctx context.Context,
	config ReadinessConfig,
) error {
	if config.URL == "" {
		return errors.New("readiness URL is empty")
	}

	client := config.Client
	if client == nil {
		client = &http.Client{}
	}

	timeout := config.Timeout
	if timeout <= 0 {
		timeout = 2 * time.Minute
	}

	interval := config.Interval
	if interval <= 0 {
		interval = 200 * time.Millisecond
	}

	waitContext, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	var lastErr error

	for {
		request, err := http.NewRequestWithContext(
			waitContext,
			http.MethodGet,
			config.URL,
			nil,
		)
		if err != nil {
			return fmt.Errorf("create readiness request: %w", err)
		}

		response, err := client.Do(request)
		if err == nil {
			_, _ = io.Copy(io.Discard, response.Body)
			_ = response.Body.Close()

			if response.StatusCode >= 200 &&
				response.StatusCode < 300 {
				return nil
			}

			lastErr = fmt.Errorf(
				"readiness endpoint returned HTTP %d",
				response.StatusCode,
			)
		} else {
			lastErr = err
		}

		timer := time.NewTimer(interval)

		select {
		case <-waitContext.Done():
			if lastErr != nil {
				return fmt.Errorf(
					"wait for HTTP readiness: %w",
					lastErr,
				)
			}

			return waitContext.Err()

		case <-timer.C:
		}
	}
}
