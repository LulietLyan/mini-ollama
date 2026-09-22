package gpu

import (
	"strings"
	"testing"
)

func TestParse(t *testing.T) {
	devices, err := Parse(strings.NewReader("1, 24576, 20000\n0, 24576, 12000\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(devices) != 2 || devices[0].Index != 0 || devices[1].FreeMiB != 20000 {
		t.Fatalf("devices = %#v", devices)
	}
}

func TestAllocateUsesLargestFreeDevices(t *testing.T) {
	allocation, err := Allocate([]Device{
		{Index: 0, TotalMiB: 24000, FreeMiB: 4000},
		{Index: 1, TotalMiB: 24000, FreeMiB: 18000},
		{Index: 2, TotalMiB: 24000, FreeMiB: 12000},
	}, nil, 20000)
	if err != nil {
		t.Fatal(err)
	}
	if allocation.DeviceList != "1,2" {
		t.Fatalf("device list = %q", allocation.DeviceList)
	}
}

func TestAllocateRejectsInsufficientMemory(t *testing.T) {
	_, err := Allocate([]Device{{Index: 0, TotalMiB: 1000, FreeMiB: 500}}, nil, 501)
	if err == nil {
		t.Fatal("expected insufficient memory error")
	}
}
