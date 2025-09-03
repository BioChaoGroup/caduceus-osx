"""MPS-optimized dataloaders for genomics datasets on Apple Silicon.

This module provides optimized data loading strategies specifically designed
for Apple Silicon MPS architecture to maximize GPU utilization and minimize
CPU-GPU data transfer overhead.
"""

import torch
from torch.utils.data import DataLoader
from typing import Any, List, Union, Optional
import numpy as np

from src.dataloaders.genomics import GenomicBenchmark
from src.dataloaders.fault_tolerant_sampler import RandomFaultTolerantSampler


class MPSOptimizedDataLoader(DataLoader):
    """MPS-optimized DataLoader that pre-allocates tensors on MPS device."""
    
    def __init__(self, dataset, batch_size, device='mps', **kwargs):
        # Allow configurable num_workers for better CPU utilization
        # Default to 0 for MPS, but can be overridden for better performance
        if 'num_workers' not in kwargs:
            kwargs['num_workers'] = 0
        
        # pin_memory can be beneficial for MPS with multiple workers
        if 'pin_memory' not in kwargs:
            kwargs['pin_memory'] = (kwargs['num_workers'] > 0)
        
        # Enable persistent workers if using multiple workers
        if kwargs.get('num_workers', 0) > 0:
            kwargs['persistent_workers'] = True
        
        super().__init__(dataset, batch_size=batch_size, **kwargs)
        self.device = torch.device(device)
        self._prefetch_factor = kwargs.get('prefetch_factor', 2)
        
    def __iter__(self):
        """Optimized iterator that moves data to MPS device efficiently."""
        for batch in super().__iter__():
            # Move batch to MPS device with non_blocking=True for better performance
            if isinstance(batch, (list, tuple)):
                batch = [self._move_to_device(item) for item in batch]
            else:
                batch = self._move_to_device(batch)
            yield batch
    
    def _move_to_device(self, tensor):
        """Move tensor to MPS device with optimizations."""
        if isinstance(tensor, torch.Tensor):
            # Use non_blocking transfer when possible
            return tensor.to(self.device, non_blocking=True)
        elif isinstance(tensor, dict):
            return {k: self._move_to_device(v) for k, v in tensor.items()}
        elif isinstance(tensor, (list, tuple)):
            return type(tensor)(self._move_to_device(item) for item in tensor)
        else:
            return tensor


class MPSOptimizedGenomicBenchmark(GenomicBenchmark):
    """MPS-optimized version of GenomicBenchmark dataloader."""
    
    _name_ = "mps_optimized_genomic_benchmark"
    
    def __init__(self, *args, **kwargs):
        # Preserve passed num_workers and pin_memory values for optimal performance
        # Only set defaults if not provided
        if 'num_workers' not in kwargs:
            kwargs['num_workers'] = 0  # Default to single worker for MPS
        if 'pin_memory' not in kwargs:
            kwargs['pin_memory'] = False  # Default disable pin_memory for MPS
        kwargs['drop_last'] = kwargs.get('drop_last', True)  # Drop last for consistent batch sizes
        
        # Add MPS-specific batch size adjustment
        if 'batch_size' in kwargs:
            original_batch_size = kwargs['batch_size']
            # Reduce batch size slightly for MPS memory efficiency
            kwargs['batch_size'] = max(1, int(original_batch_size * 0.8))
            print(f"Adjusted batch size from {original_batch_size} to {kwargs['batch_size']} for MPS optimization")
        
        super().__init__(*args, **kwargs)
        
        # MPS-specific settings
        self.mps_device = torch.device('mps')
        self.prefetch_to_device = True
        self.use_amp = True
        
    def setup(self, stage=None, device=None):
        """Setup with MPS-specific optimizations."""
        # Force device to MPS if available
        if device is None and torch.backends.mps.is_available():
            device = self.mps_device
            
        super().setup(stage=stage, device=device)
        
        # Pre-warm MPS device
        if device and device.type == 'mps':
            self._prewarm_mps_device()
    
    def _prewarm_mps_device(self):
        """Pre-warm MPS device with dummy operations."""
        try:
            # Create small tensors to initialize MPS
            dummy_tensor = torch.randn(10, 10, device=self.mps_device)
            _ = torch.matmul(dummy_tensor, dummy_tensor.T)
            del dummy_tensor
            torch.mps.empty_cache()
            print("MPS device pre-warmed successfully")
        except Exception as e:
            print(f"Warning: MPS pre-warming failed: {e}")
    
    def _create_optimized_dataloader(self, dataset, batch_size, shuffle=False, **kwargs):
        """Create MPS-optimized dataloader."""
        # Remove incompatible arguments for MPS
        kwargs.pop('sampler', None)
        
        # Use instance settings for num_workers and pin_memory
        kwargs['pin_memory'] = self.pin_memory
        kwargs['num_workers'] = self.num_workers
        kwargs['drop_last'] = True
        
        # Use custom MPS-optimized DataLoader
        return MPSOptimizedDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            device=self.mps_device,
            **kwargs
        )
    
    def train_dataloader(self, **kwargs: Any) -> DataLoader:
        """MPS-optimized train dataloader."""
        # Use fault-tolerant sampler if needed
        if self.shuffle and self.fault_tolerant:
            sampler = RandomFaultTolerantSampler(self.dataset_train)
            kwargs['sampler'] = sampler
            kwargs['shuffle'] = False
        else:
            kwargs['shuffle'] = self.shuffle
            
        return self._create_optimized_dataloader(
            self.dataset_train, 
            batch_size=self.batch_size,
            **kwargs
        )
    
    def val_dataloader(self, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """MPS-optimized validation dataloader."""
        kwargs["drop_last"] = False
        kwargs["shuffle"] = False
        return self._create_optimized_dataloader(
            self.dataset_val, 
            batch_size=self.batch_size_eval,
            **kwargs
        )
    
    def test_dataloader(self, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """MPS-optimized test dataloader."""
        kwargs["drop_last"] = False
        kwargs["shuffle"] = False
        return self._create_optimized_dataloader(
            self.dataset_test, 
            batch_size=self.batch_size_eval,
            **kwargs
        )


class MPSMemoryOptimizedBatch:
    """Memory-optimized batch container for MPS."""
    
    def __init__(self, batch_data, device='mps'):
        self.device = torch.device(device)
        self._data = self._optimize_batch(batch_data)
    
    def _optimize_batch(self, batch_data):
        """Optimize batch data for MPS memory usage."""
        if isinstance(batch_data, torch.Tensor):
            # Ensure contiguous memory layout
            tensor = batch_data.contiguous()
            # Move to MPS with appropriate dtype
            if tensor.dtype == torch.float64:
                tensor = tensor.to(torch.float32)  # MPS prefers float32
            return tensor.to(self.device, non_blocking=True)
        elif isinstance(batch_data, (list, tuple)):
            return type(batch_data)(self._optimize_batch(item) for item in batch_data)
        elif isinstance(batch_data, dict):
            return {k: self._optimize_batch(v) for k, v in batch_data.items()}
        else:
            return batch_data
    
    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)
    
    def to(self, device):
        """Move batch to specified device."""
        return MPSMemoryOptimizedBatch(self._data, device)


def create_mps_optimized_dataloader(dataset, batch_size, device='mps', **kwargs):
    """Factory function to create MPS-optimized dataloader."""
    return MPSOptimizedDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        device=device,
        **kwargs
    )


# Register the MPS-optimized dataloader
from src.dataloaders.base import SequenceDataset
SequenceDataset.registry["mps_optimized_genomic_benchmark"] = MPSOptimizedGenomicBenchmark
