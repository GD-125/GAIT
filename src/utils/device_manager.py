"""
Device Manager for CPU/GPU Selection
Provides intelligent device selection with user choice and system detection.
"""

import torch
import sys
from typing import Optional


class DeviceManager:
    """Manages device selection for training and inference."""

    def __init__(self):
        """Initialize device manager."""
        self.available_devices = self._detect_devices()
        self.selected_device = None

    def _detect_devices(self) -> dict:
        """
        Detect available computing devices.

        Returns:
            Dictionary with device information
        """
        devices = {
            'cpu': {
                'available': True,
                'name': 'CPU',
                'description': 'Central Processing Unit (Universal)'
            },
            'cuda': {
                'available': False,
                'name': 'GPU',
                'description': 'Not Available',
                'count': 0
            }
        }

        # Check CUDA/GPU availability
        if torch.cuda.is_available():
            devices['cuda']['available'] = True
            devices['cuda']['count'] = torch.cuda.device_count()
            devices['cuda']['description'] = torch.cuda.get_device_name(0)

            # Get GPU memory
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                devices['cuda']['memory_gb'] = f"{total_memory:.2f} GB"
            except:
                devices['cuda']['memory_gb'] = "Unknown"

        return devices

    def print_available_devices(self):
        """Print all available devices with details."""
        print("\n" + "="*80)
        print("AVAILABLE COMPUTING DEVICES")
        print("="*80)

        device_num = 1

        # CPU
        print(f"\n[{device_num}] CPU - {self.available_devices['cpu']['description']}")
        print(f"    Status: Always Available")
        print(f"    Best for: Small datasets, Testing, CPU-only environments")

        # GPU/CUDA
        device_num += 1
        if self.available_devices['cuda']['available']:
            print(f"\n[{device_num}] GPU (CUDA) - {self.available_devices['cuda']['description']}")
            print(f"    Status: Available ✓")
            print(f"    Memory: {self.available_devices['cuda'].get('memory_gb', 'Unknown')}")
            print(f"    GPU Count: {self.available_devices['cuda']['count']}")
            print(f"    Best for: Large datasets, Faster training, Production use")
            print(f"    Recommended: YES (Faster Performance)")
        else:
            print(f"\n[{device_num}] GPU (CUDA) - Not Available")
            print(f"    Status: No CUDA-capable GPU detected")
            print(f"    Note: Install CUDA toolkit for GPU support")

        print("\n" + "="*80 + "\n")

    def get_device(self,
                   device_choice: Optional[str] = None,
                   prefer_gpu: bool = True,
                   interactive: bool = False) -> torch.device:
        """
        Get computing device based on preferences.

        Args:
            device_choice: 'cpu', 'cuda', or None (auto-detect)
            prefer_gpu: If True and GPU available, use GPU (default: True)
            interactive: If True, prompt user for device selection

        Returns:
            torch.device object
        """
        # Interactive mode
        if interactive:
            return self._interactive_device_selection()

        # Explicit device choice
        if device_choice is not None:
            device_choice = device_choice.lower()

            if device_choice == 'cpu':
                device = torch.device('cpu')
                print(f"\n✓ Selected Device: CPU")

            elif device_choice in ['cuda', 'gpu']:
                if self.available_devices['cuda']['available']:
                    device = torch.device('cuda')
                    print(f"\n✓ Selected Device: GPU - {self.available_devices['cuda']['description']}")
                    print(f"  Memory: {self.available_devices['cuda'].get('memory_gb', 'Unknown')}")
                else:
                    print(f"\n⚠ Warning: GPU requested but not available. Falling back to CPU.")
                    device = torch.device('cpu')
            else:
                raise ValueError(f"Invalid device choice: {device_choice}. Use 'cpu' or 'cuda'.")

        # Auto-detection with GPU preference
        else:
            if prefer_gpu and self.available_devices['cuda']['available']:
                device = torch.device('cuda')
                print(f"\n✓ Auto-Selected Device: GPU - {self.available_devices['cuda']['description']}")
                print(f"  Memory: {self.available_devices['cuda'].get('memory_gb', 'Unknown')}")
                print(f"  GPU acceleration enabled for faster training")
            else:
                device = torch.device('cpu')
                reason = "CPU" if not prefer_gpu else "GPU not available"
                print(f"\n✓ Auto-Selected Device: CPU ({reason})")

        self.selected_device = device
        return device

    def _interactive_device_selection(self) -> torch.device:
        """
        Interactive device selection via user input.

        Returns:
            Selected torch.device
        """
        self.print_available_devices()

        # Determine available options
        options = ["1"]  # CPU always available
        if self.available_devices['cuda']['available']:
            options.append("2")

        while True:
            try:
                choice = input("Select device [1 for CPU, 2 for GPU, or 'auto' for automatic]: ").strip().lower()

                if choice == 'auto':
                    # Auto mode prefers GPU if available
                    if self.available_devices['cuda']['available']:
                        device = torch.device('cuda')
                        print(f"\n✓ Auto-Selected: GPU - {self.available_devices['cuda']['description']}")
                    else:
                        device = torch.device('cpu')
                        print(f"\n✓ Auto-Selected: CPU")
                    break

                elif choice == '1':
                    device = torch.device('cpu')
                    print(f"\n✓ Selected: CPU")
                    break

                elif choice == '2':
                    if self.available_devices['cuda']['available']:
                        device = torch.device('cuda')
                        print(f"\n✓ Selected: GPU - {self.available_devices['cuda']['description']}")
                        break
                    else:
                        print("❌ GPU not available. Please select CPU (1) or auto.")

                else:
                    print(f"❌ Invalid choice. Please enter {', '.join(options)} or 'auto'.")

            except KeyboardInterrupt:
                print("\n\n⚠ Interrupted. Using CPU by default.")
                device = torch.device('cpu')
                break

        self.selected_device = device
        return device

    def get_device_info(self) -> dict:
        """
        Get detailed information about current device.

        Returns:
            Dictionary with device information
        """
        if self.selected_device is None:
            return {"error": "No device selected"}

        info = {
            'device_type': str(self.selected_device),
            'device_name': None,
            'cuda_available': torch.cuda.is_available(),
        }

        if self.selected_device.type == 'cuda':
            info['device_name'] = torch.cuda.get_device_name(0)
            info['device_count'] = torch.cuda.device_count()
            info['cuda_version'] = torch.version.cuda

            # Memory info
            try:
                info['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info['allocated_memory_gb'] = torch.cuda.memory_allocated(0) / (1024**3)
                info['cached_memory_gb'] = torch.cuda.memory_reserved(0) / (1024**3)
            except:
                pass
        else:
            info['device_name'] = 'CPU'

        return info

    def print_device_summary(self):
        """Print summary of selected device."""
        if self.selected_device is None:
            print("No device selected yet.")
            return

        info = self.get_device_info()

        print("\n" + "="*80)
        print("DEVICE SUMMARY")
        print("="*80)
        print(f"Device Type: {info['device_type'].upper()}")
        print(f"Device Name: {info['device_name']}")

        if info['device_type'] == 'cuda':
            print(f"CUDA Available: {info['cuda_available']}")
            print(f"CUDA Version: {info.get('cuda_version', 'N/A')}")
            print(f"Total Memory: {info.get('total_memory_gb', 0):.2f} GB")
            print(f"PyTorch Version: {torch.__version__}")

        print("="*80 + "\n")


def get_device(device_choice: Optional[str] = None,
               prefer_gpu: bool = True,
               interactive: bool = False,
               verbose: bool = True) -> torch.device:
    """
    Convenience function to get device quickly.

    Args:
        device_choice: 'cpu', 'cuda', or None (auto)
        prefer_gpu: Prefer GPU if available (default: True)
        interactive: Show interactive selection menu
        verbose: Print device information

    Returns:
        torch.device object

    Examples:
        >>> device = get_device()  # Auto-select, prefer GPU
        >>> device = get_device(device_choice='cpu')  # Force CPU
        >>> device = get_device(interactive=True)  # Interactive selection
    """
    manager = DeviceManager()

    if verbose and not interactive:
        manager.print_available_devices()

    device = manager.get_device(
        device_choice=device_choice,
        prefer_gpu=prefer_gpu,
        interactive=interactive
    )

    if verbose:
        print()

    return device


if __name__ == "__main__":
    print("="*80)
    print("DEVICE MANAGER - Test Mode")
    print("="*80)

    # Test 1: Auto-detection (prefer GPU)
    print("\n[TEST 1] Auto-detection with GPU preference:")
    device1 = get_device(prefer_gpu=True)
    print(f"Result: {device1}")

    # Test 2: Force CPU
    print("\n[TEST 2] Force CPU:")
    device2 = get_device(device_choice='cpu')
    print(f"Result: {device2}")

    # Test 3: Request GPU
    print("\n[TEST 3] Request GPU:")
    device3 = get_device(device_choice='cuda')
    print(f"Result: {device3}")

    # Test 4: Device Manager class
    print("\n[TEST 4] Device Manager Class:")
    manager = DeviceManager()
    manager.print_available_devices()
    device4 = manager.get_device(prefer_gpu=True)
    manager.print_device_summary()

    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
