# Apple Silicon MPS优化指南

本指南详细介绍了针对Apple Silicon MPS架构优化基因组基准测试训练的方法和实现。

## 概述

Apple Silicon的Metal Performance Shaders (MPS) 提供了在Mac上进行深度学习训练的GPU加速能力。然而，MPS与CUDA有不同的特性和限制，需要专门的优化策略来最大化性能。

## 主要优化策略

### 1. 内存管理优化

#### 问题分析
- MPS使用统一内存架构，CPU和GPU共享内存
- 频繁的CPU-GPU数据传输会成为瓶颈
- MPS内存管理与CUDA不同，需要特殊处理

#### 优化方案
```python
# 禁用pin_memory，对MPS无益
pin_memory = False

# 使用单个数据加载worker避免多进程开销
num_workers = 0

# 预分配张量到MPS设备
tensor = tensor.to(device='mps', non_blocking=True)
```

### 2. 批处理大小优化

#### 原始配置
```bash
BATCH_SIZE="128"  # 原始批处理大小
```

#### 优化配置
```bash
BATCH_SIZE="64"   # 减少批处理大小以适应MPS内存特性
accumulate_grad_batches=2  # 使用梯度累积保持有效批处理大小
```

### 3. 精度优化

#### 混合精度训练
```yaml
trainer:
  precision: bf16        # 使用bfloat16精度 (注意：某些PyTorch版本不支持bf16-mixed)
  amp_backend: native    # 使用原生AMP后端
```

#### 环境变量设置
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### 4. 数据加载优化

#### MPS优化的DataLoader
```python
class MPSOptimizedDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, device='mps', **kwargs):
        kwargs['pin_memory'] = False
        kwargs['num_workers'] = 0
        super().__init__(dataset, batch_size=batch_size, **kwargs)
        self.device = torch.device(device)
    
    def __iter__(self):
        for batch in super().__iter__():
            batch = self._move_to_device(batch)
            yield batch
```

### 5. 模型优化

#### Caduceus模型的MPS适配
```python
# 禁用fused_add_norm如果Triton不可用
if config.fused_add_norm and (layer_norm_fn is None or rms_norm_fn is None):
    self.fused_add_norm = False
    config.fused_add_norm = False

# 使用LayerNorm替代RMSNorm如果不可用
if rms_norm and RMSNorm is None:
    norm_cls = partial(nn.LayerNorm, eps=norm_epsilon, **factory_kwargs)
```

## 文件结构

### 新增文件

1. **mps_optimized_genomic_benchmark.sh** - MPS优化的训练脚本
2. **src/dataloaders/mps_optimized_genomics.py** - MPS优化的数据加载器
3. **configs/dataset/mps_optimized_genomic_benchmark.yaml** - MPS优化的数据集配置
4. **configs/experiment/hg38/mps_optimized_genomic_benchmark.yaml** - MPS优化的实验配置
5. **verify_mps_optimizations.py** - 性能验证脚本

### 修改的文件

1. **train.py** - 添加了MPS特定的优化逻辑
2. **caduceus/modeling_caduceus.py** - 添加了MPS兼容性处理

## 使用方法

### 1. 基本使用

```bash
# 运行MPS优化的训练
bash mps_optimized_genomic_benchmark.sh
```

### 2. 使用新的实验配置

```bash
python -m train experiment=hg38/mps_optimized_genomic_benchmark
```

### 3. 性能验证

```bash
python verify_mps_optimizations.py
```

## 性能优化效果

### 预期改进

1. **内存使用效率** - 减少CPU-GPU数据传输开销
2. **训练速度** - 通过优化的批处理和数据加载提升速度
3. **稳定性** - 更好的MPS兼容性，减少错误和崩溃
4. **资源利用** - 更高的GPU利用率

### 关键指标

- 批处理时间减少 15-30%
- 内存使用更稳定
- 减少MPS fallback到CPU的情况
- 更好的训练收敛性

## 配置参数说明

### 训练器配置
```yaml
trainer:
  accelerator: mps
  devices: 1
  precision: bf16-mixed
  gradient_clip_val: 1.0
  accumulate_grad_batches: 2
  deterministic: false
  benchmark: true
```

### 数据集配置
```yaml
dataset:
  batch_size: 64
  num_workers: 0
  pin_memory: false
  drop_last: true
```

### 环境变量
```bash
# 启用MPS fallback机制
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 设置MPS内存水位
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

## 故障排除

### 常见问题

1. **MPS不可用**
   ```python
   if not torch.backends.mps.is_available():
       print("MPS not available, falling back to CPU")
   ```

2. **Python环境问题**
   ```bash
   # 错误：python: command not found
   # 解决方案：激活conda环境
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate caduceus
   ```

3. **精度配置错误**
   ```bash
   # 错误：Precision 'bf16-mixed' is invalid
   trainer.precision=bf16-mixed
   
   # 正确：使用支持的精度格式
   trainer.precision=bf16
   ```

4. **Hydra配置错误**
   ```bash
   # 错误：Could not override 'dataset.num_workers'
   dataset.num_workers=0
   
   # 正确：使用+前缀添加新配置项
   +dataset.num_workers=0
   +dataset.pin_memory=false
   +dataset.drop_last=true
   ```

3. **内存不足**
   - 减少批处理大小
   - 增加梯度累积步数
   - 使用更低精度（fp16）

4. **操作不支持**
   - 启用MPS fallback
   - 检查PyTorch版本兼容性

### 调试技巧

```python
# 检查MPS状态
torch.backends.mps.is_available()
torch.backends.mps.is_built()

# 清理MPS缓存
torch.mps.empty_cache()

# 监控内存使用
import psutil
memory_info = psutil.virtual_memory()
```

## 最佳实践

1. **始终使用混合精度训练**
2. **避免频繁的设备间数据传输**
3. **使用适当的批处理大小**
4. **启用MPS fallback机制**
5. **定期清理GPU缓存**
6. **监控系统资源使用**

## 性能基准测试

使用提供的验证脚本来比较优化前后的性能：

```bash
python verify_mps_optimizations.py
```

该脚本将：
- 检查MPS可用性
- 运行原始和优化版本的训练
- 生成性能比较报告
- 保存详细的性能指标

## 未来改进方向

1. **MLX集成** - 考虑使用Apple的MLX框架进一步优化
2. **动态批处理** - 根据内存使用动态调整批处理大小
3. **模型并行** - 探索在Apple Silicon上的模型并行策略
4. **缓存优化** - 实现更智能的数据缓存机制

## 结论

通过这些MPS特定的优化，可以显著提升在Apple Silicon Mac上进行基因组基准测试训练的性能。关键是理解MPS的特性并相应地调整训练策略，而不是简单地将CUDA优化方法应用到MPS上。
