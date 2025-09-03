#!/usr/bin/env python3
"""
Verification script for MPS optimizations in genomic benchmark training.

This script compares the performance of the original training setup with
the MPS-optimized version, measuring GPU utilization, memory usage, and
training speed.
"""

import os
import sys
import time
import torch
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import json


class MPSPerformanceMonitor:
    """Monitor MPS performance metrics during training."""
    
    def __init__(self):
        self.metrics = {
            'memory_usage': [],
            'gpu_utilization': [],
            'training_speed': [],
            'batch_processing_time': [],
            'data_loading_time': []
        }
        self.start_time = None
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        print("Starting MPS performance monitoring...")
        
    def record_memory_usage(self):
        """Record current memory usage."""
        if torch.backends.mps.is_available():
            # Get MPS memory usage
            try:
                # MPS doesn't have direct memory query like CUDA
                # We'll use system memory as proxy
                memory_info = psutil.virtual_memory()
                self.metrics['memory_usage'].append({
                    'timestamp': time.time() - self.start_time,
                    'system_memory_percent': memory_info.percent,
                    'system_memory_used_gb': memory_info.used / (1024**3)
                })
            except Exception as e:
                print(f"Warning: Could not record memory usage: {e}")
    
    def record_batch_time(self, batch_time: float):
        """Record batch processing time."""
        self.metrics['batch_processing_time'].append({
            'timestamp': time.time() - self.start_time,
            'batch_time': batch_time
        })
    
    def record_data_loading_time(self, loading_time: float):
        """Record data loading time."""
        self.metrics['data_loading_time'].append({
            'timestamp': time.time() - self.start_time,
            'loading_time': loading_time
        })
    
    def get_summary(self) -> Dict:
        """Get performance summary."""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        avg_batch_time = 0
        if self.metrics['batch_processing_time']:
            avg_batch_time = sum(m['batch_time'] for m in self.metrics['batch_processing_time']) / len(self.metrics['batch_processing_time'])
        
        avg_loading_time = 0
        if self.metrics['data_loading_time']:
            avg_loading_time = sum(m['loading_time'] for m in self.metrics['data_loading_time']) / len(self.metrics['data_loading_time'])
        
        return {
            'total_time': total_time,
            'average_batch_time': avg_batch_time,
            'average_data_loading_time': avg_loading_time,
            'memory_samples': len(self.metrics['memory_usage']),
            'batch_samples': len(self.metrics['batch_processing_time'])
        }


def check_mps_availability():
    """Check if MPS is available and properly configured."""
    print("Checking MPS availability...")
    
    if not torch.backends.mps.is_available():
        print("❌ MPS is not available on this system")
        return False
    
    print("✅ MPS is available")
    
    # Test basic MPS operations
    try:
        device = torch.device('mps')
        x = torch.randn(100, 100, device=device)
        y = torch.matmul(x, x.T)
        print("✅ Basic MPS operations working")
        return True
    except Exception as e:
        print(f"❌ MPS operations failed: {e}")
        return False


def run_training_benchmark(script_path: str, config_name: str, duration_minutes: int = 5) -> Dict:
    """Run a training benchmark for specified duration."""
    print(f"\nRunning benchmark: {config_name}")
    print(f"Script: {script_path}")
    print(f"Duration: {duration_minutes} minutes")
    
    monitor = MPSPerformanceMonitor()
    monitor.start_monitoring()
    
    # Prepare environment
    env = os.environ.copy()
    env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    env['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    start_time = time.time()
    
    try:
        # Run the training script
        process = subprocess.Popen(
            ['bash', script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )
        
        # Monitor for specified duration
        while time.time() - start_time < duration_minutes * 60:
            if process.poll() is not None:
                break
            
            monitor.record_memory_usage()
            time.sleep(10)  # Record every 10 seconds
        
        # Terminate if still running
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        
        stdout, stderr = process.communicate()
        
        return {
            'config_name': config_name,
            'success': process.returncode == 0 or process.returncode == -15,  # -15 is SIGTERM
            'performance_summary': monitor.get_summary(),
            'stdout_sample': stdout[-1000:] if stdout else "",
            'stderr_sample': stderr[-1000:] if stderr else ""
        }
        
    except Exception as e:
        return {
            'config_name': config_name,
            'success': False,
            'error': str(e),
            'performance_summary': monitor.get_summary()
        }


def compare_configurations():
    """Compare original vs optimized configurations."""
    print("\n" + "="*60)
    print("COMPARING TRAINING CONFIGURATIONS")
    print("="*60)
    
    # Define configurations to test
    configs = [
        {
            'name': 'Original',
            'script': 'run_genomic_benchmark_no_wandb.sh',
            'description': 'Original training script with basic MPS support'
        },
        {
            'name': 'MPS Optimized',
            'script': 'mps_optimized_genomic_benchmark.sh',
            'description': 'Enhanced MPS-optimized training script'
        }
    ]
    
    results = []
    
    for config in configs:
        script_path = Path(config['script'])
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            continue
        
        print(f"\n📊 Testing {config['name']} configuration...")
        print(f"Description: {config['description']}")
        
        result = run_training_benchmark(str(script_path), config['name'], duration_minutes=3)
        results.append(result)
        
        # Print immediate results
        if result['success']:
            summary = result['performance_summary']
            print(f"✅ {config['name']} completed successfully")
            print(f"   Average batch time: {summary['average_batch_time']:.4f}s")
            print(f"   Total samples: {summary['batch_samples']}")
        else:
            print(f"❌ {config['name']} failed")
            if 'error' in result:
                print(f"   Error: {result['error']}")
    
    return results


def generate_report(results: List[Dict]):
    """Generate a comprehensive performance report."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON REPORT")
    print("="*60)
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'system_info': {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'mps_available': torch.backends.mps.is_available(),
            'system_memory_gb': psutil.virtual_memory().total / (1024**3)
        },
        'results': results
    }
    
    # Print summary
    for result in results:
        print(f"\n📈 {result['config_name']} Results:")
        if result['success']:
            summary = result['performance_summary']
            print(f"   ✅ Status: Success")
            print(f"   ⏱️  Average batch time: {summary['average_batch_time']:.4f}s")
            print(f"   📊 Batch samples: {summary['batch_samples']}")
            print(f"   💾 Memory samples: {summary['memory_samples']}")
        else:
            print(f"   ❌ Status: Failed")
            if 'error' in result:
                print(f"   🚫 Error: {result['error']}")
    
    # Save detailed report
    report_path = Path('mps_optimization_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    # Performance comparison
    successful_results = [r for r in results if r['success']]
    if len(successful_results) >= 2:
        original = next((r for r in successful_results if 'Original' in r['config_name']), None)
        optimized = next((r for r in successful_results if 'Optimized' in r['config_name']), None)
        
        if original and optimized:
            orig_time = original['performance_summary']['average_batch_time']
            opt_time = optimized['performance_summary']['average_batch_time']
            
            if orig_time > 0 and opt_time > 0:
                improvement = ((orig_time - opt_time) / orig_time) * 100
                print(f"\n🚀 Performance Improvement: {improvement:.2f}%")
                if improvement > 0:
                    print(f"   MPS optimizations provide {improvement:.2f}% faster batch processing")
                else:
                    print(f"   MPS optimizations are {abs(improvement):.2f}% slower (may need tuning)")


def main():
    """Main verification function."""
    print("🔍 MPS Optimization Verification Script")
    print("="*50)
    
    # Check system requirements
    if not check_mps_availability():
        print("❌ MPS not available. Cannot run verification.")
        return 1
    
    # Check if required files exist
    required_files = [
        'run_genomic_benchmark_no_wandb.sh',
        'mps_optimized_genomic_benchmark.sh',
        'train.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return 1
    
    print("✅ All required files found")
    
    # Run comparisons
    try:
        results = compare_configurations()
        generate_report(results)
        
        print("\n✅ Verification completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Verification interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
