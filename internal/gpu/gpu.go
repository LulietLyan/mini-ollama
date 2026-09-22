package gpu

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"math"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"time"
)

type Device struct {
	Index    int
	TotalMiB int64
	FreeMiB  int64
}

type Probe interface {
	List(context.Context) ([]Device, error)
}

type NVIDIAProbe struct {
	Command string
	Timeout time.Duration
}

func (p NVIDIAProbe) List(ctx context.Context) ([]Device, error) {
	command := p.Command
	if command == "" {
		command = "nvidia-smi"
	}

	timeout := p.Timeout
	if timeout <= 0 {
		timeout = 5 * time.Second
	}

	probeContext, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	output, err := exec.CommandContext(probeContext, command,
		"--query-gpu=index,memory.total,memory.free",
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		if probeContext.Err() != nil {
			return nil, fmt.Errorf("query NVIDIA GPU memory: %w", probeContext.Err())
		}
		return nil, fmt.Errorf("query NVIDIA GPU memory with %s: %w", command, err)
	}

	devices, err := Parse(strings.NewReader(string(output)))
	if err != nil {
		return nil, err
	}

	if len(devices) == 0 {
		return nil, fmt.Errorf("nvidia-smi returned no GPUs")
	}

	return devices, nil
}

func Parse(input io.Reader) ([]Device, error) {
	scanner := bufio.NewScanner(input)
	devices := make([]Device, 0)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		fields := strings.Split(line, ",")
		if len(fields) != 3 {
			return nil, fmt.Errorf("invalid nvidia-smi GPU row: %q", line)
		}
		index, err := strconv.ParseInt(strings.TrimSpace(fields[0]), 10, 32)
		if err != nil {
			return nil, fmt.Errorf("invalid GPU index %q: %w", fields[0], err)
		}
		total, err := strconv.ParseInt(strings.TrimSpace(fields[1]), 10, 64)
		if err != nil || total < 0 {
			return nil, fmt.Errorf("invalid total GPU memory %q", fields[1])
		}
		free, err := strconv.ParseInt(strings.TrimSpace(fields[2]), 10, 64)
		if err != nil || free < 0 || free > total {
			return nil, fmt.Errorf("invalid free GPU memory %q", fields[2])
		}
		devices = append(devices, Device{Index: int(index), TotalMiB: total, FreeMiB: free})
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read nvidia-smi output: %w", err)
	}
	sort.Slice(devices, func(i, j int) bool { return devices[i].Index < devices[j].Index })
	return devices, nil
}

func RequiredMemoryMiB(modelBytes int64, contextSize int, reserveMiB int64, fraction float64) (int64, error) {
	if modelBytes <= 0 {
		return 0, fmt.Errorf("model size must be positive")
	}
	if fraction <= 0 {
		fraction = 0.90
	}
	if fraction > 1 {
		return 0, fmt.Errorf("GPU memory fraction must be between 0 and 1")
	}
	if reserveMiB <= 0 {
		reserveMiB = 512
	}
	required := int64(math.Ceil(float64(modelBytes)/(1024*1024)*1.18)) + reserveMiB
	if contextSize > 0 {
		required += int64(contextSize) / 4
	}
	return int64(math.Ceil(float64(required) / fraction)), nil
}

type Allocation struct {
	Devices     []Device
	DeviceList  string
	TensorSplit string
	RequiredMiB int64
}

func Allocate(devices []Device, requested []int, requiredMiB int64) (Allocation, error) {
	if requiredMiB <= 0 {
		return Allocation{}, fmt.Errorf("required GPU memory must be positive")
	}
	byIndex := make(map[int]Device, len(devices))
	for _, device := range devices {
		if device.Index < 0 || device.TotalMiB <= 0 || device.FreeMiB < 0 || device.FreeMiB > device.TotalMiB {
			return Allocation{}, fmt.Errorf("invalid GPU memory information for device %d", device.Index)
		}
		byIndex[device.Index] = device
	}
	selected := make([]Device, 0)
	if len(requested) > 0 {
		for _, index := range requested {
			device, ok := byIndex[index]
			if !ok {
				return Allocation{}, fmt.Errorf("requested GPU %d was not found", index)
			}
			selected = append(selected, device)
		}
	} else {
		candidates := append([]Device(nil), devices...)
		sort.Slice(candidates, func(i, j int) bool {
			if candidates[i].FreeMiB == candidates[j].FreeMiB {
				return candidates[i].Index < candidates[j].Index
			}
			return candidates[i].FreeMiB > candidates[j].FreeMiB
		})
		var free int64
		for _, device := range candidates {
			if free >= requiredMiB {
				break
			}
			selected = append(selected, device)
			free += device.FreeMiB
		}
		if free < requiredMiB {
			return Allocation{}, fmt.Errorf("insufficient free GPU memory: need %d MiB", requiredMiB)
		}
		sort.Slice(selected, func(i, j int) bool { return selected[i].Index < selected[j].Index })
	}
	var free int64
	for _, device := range selected {
		free += device.FreeMiB
	}
	if free < requiredMiB {
		return Allocation{}, fmt.Errorf("selected GPUs provide %d MiB free memory, need %d MiB", free, requiredMiB)
	}
	deviceNames := make([]string, len(selected))
	ratios := make([]string, len(selected))
	for i, device := range selected {
		deviceNames[i] = strconv.Itoa(device.Index)
		ratios[i] = strconv.FormatFloat(float64(device.FreeMiB), 'f', 0, 64)
	}
	return Allocation{
		Devices: selected, DeviceList: strings.Join(deviceNames, ","),
		TensorSplit: strings.Join(ratios, ","), RequiredMiB: requiredMiB,
	}, nil
}
