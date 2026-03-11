# CUDA 编程问答记录

## 问题：add1cpu.cu 和 add2gpu.cu 这两个代码有什么区别？

**日期**: 2026-03-07

### 回答

这两个文件分别展示了向量加法（Vector Addition）的 CPU 串行实现和 GPU 并行实现。它们是 CUDA 编程入门最经典的对比示例。

#### 1. 核心区别总结

*   **`add1cpu.cu`**: 在 CPU 上运行，使用标准的 C++ 编写，通过 `for` 循环串行处理每个元素的加法。
*   **`add2gpu.cu`**: 在 GPU 上运行，使用 CUDA C++ 编写，通过启动大量的线程（Threads）并行处理加法，每个线程负责一个或多个元素的计算。

#### 2. 详细差异对比

| 特性 | add1cpu.cu (CPU 版本) | add2gpu.cu (GPU 版本) |
| :--- | :--- | :--- |
| **执行设备** | Host (CPU) | Device (GPU) |
| **函数定义** | 普通 C++ 函数 `void add(...)` | CUDA 核函数 `void __global__ add(...)` |
| **内存分配** | `malloc` (Host 内存) | `cudaMalloc` (Device 内存) + `malloc` (Host 内存) |
| **数据传输** | 不需要 | 需要 `cudaMemcpy` 在 Host 和 Device 之间传输数据 |
| **计算逻辑** | 使用 `for` 循环遍历所有元素 | 使用线程索引计算当前线程负责的元素 |
| **并行度** | 串行 (1 个线程) | 高度并行 (成千上万个线程) |
| **资源释放** | `free` | `cudaFree` (Device) + `free` (Host) |

#### 3. 关键代码对比

**A. 函数定义与执行**

*   **CPU**:
    ```cpp
    // 定义
    void add(const real *x, const real *y, real *z, const int N)
    {
        for (int n = 0; n < N; ++n) // 串行循环
        {
            z[n] = x[n] + y[n];
        }
    }

    // 调用
    add(x, y, z, N);
    ```

*   **GPU**:
    ```cpp
    // 定义 (__global__ 表明在设备上执行，从主机调用)
    void __global__ add(const real *x, const real *y, real *z, const int N)
    {
        // 计算全局线程索引
        const int n = blockDim.x * blockIdx.x + threadIdx.x;
        if (n < N) // 边界检查
        {
            z[n] = x[n] + y[n]; // 并行执行，无循环
        }
    }

    // 调用 (<<<grid_size, block_size>>> 配置执行参数)
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    ```

**B. 内存管理流程**

*   **CPU**: 分配 Host 内存 -> 初始化 -> 计算 -> 释放。
*   **GPU**: 分配 Host 内存 -> 初始化 -> **分配 Device 内存** -> **Host 到 Device 数据拷贝** -> 计算 -> **Device 到 Host 数据拷贝** -> 释放所有内存。

#### 4. 性能计时注意事项

在 `add2gpu.cu` 中，计时代码包裹了核函数调用：

```cpp
CHECK(cudaEventRecord(start));
add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
CHECK(cudaEventRecord(stop));
CHECK(cudaEventSynchronize(stop));
```

*   这里的计时**仅包含 GPU 核函数的执行时间**（以及启动开销）。
*   它**不包含**数据在 CPU 和 GPU 之间传输（`cudaMemcpy`）的时间。在实际应用中，数据传输往往是性能瓶颈之一，需要综合考虑。
