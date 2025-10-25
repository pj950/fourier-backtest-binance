# 架构深入探讨

[English](ARCHITECTURE.md) | [中文](ARCHITECTURE.zh-CN.md)

本文档提供币安傅里叶回测器架构、算法和设计决策的深入技术解释。

## 目录

1. [系统概述](#系统概述)
2. [数据层架构](#数据层架构)
3. [傅里叶变换方法](#傅里叶变换方法)
4. [频谱分析](#频谱分析)
5. [信号生成逻辑](#信号生成逻辑)
6. [回测器设计](#回测器设计)
7. [优化框架](#优化框架)
8. [投资组合管理](#投资组合管理)
9. [性能考虑](#性能考虑)

---

## 系统概述

该平台使用**分层架构**，严格分离关注点：

### 设计原则

1. **关注点分离**：每层都有单一、明确定义的职责
2. **依赖倒置**：核心逻辑依赖于抽象，而非实现
3. **类型安全**：全面的类型提示用于编译时错误检测
4. **可测试性**：所有组件设计用于单元测试，最小化模拟
5. **性能**：使用 NumPy 的向量化操作以提高计算效率

### 层通信流程

```
用户输入 → UI 层 → 核心分析 → 执行引擎 → 数据层
            ↓                      ↓
         结果 ←───── 指标 ←────── 存储
```

数据单向从数据层向上流经分析到 UI，结果向下流回存储/缓存。

---

## 数据层架构

### 币安 REST API 客户端

**位置**：`core/data/binance_client.py`

#### 设计决策

1. **httpx 而非 requests**：异步支持用于未来增强，更好的连接池
2. **tenacity 用于重试**：声明式重试逻辑，带指数退避
3. **速率限制**：客户端跟踪以避免 429 错误

#### 重试策略

```python
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((BinanceTransientError,)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def fetch_klines(...):
    # 获取逻辑
```

**理由**：币安 API 可能有瞬态故障。指数退避防止雷鸣般的群体效应，同时允许从临时问题中恢复。

#### 错误处理层次

```
BinanceError（基类）
├── BinanceRateLimitError (429) → 用户应等待
├── BinanceRequestError (400, 404) → 无效请求，不重试
└── BinanceTransientError (500, 503) → 带退避重试
```

### 缓存系统

**位置**：`core/data/cache.py`

#### 为什么选择 Parquet？

| 格式 | 读取速度 | 写入速度 | 压缩 | 模式 |
|--------|------------|-------------|-------------|--------|
| CSV | 慢 | 慢 | 差 | 无 |
| JSON | 慢 | 中等 | 差 | 无 |
| Pickle | 快 | 快 | 中等 | 无 |
| **Parquet** | **非常快** | **快** | **优秀** | **有** |

**Parquet 优势**：
- 列式格式：在时间戳列上快速过滤
- 内置压缩：比 CSV 小约 10 倍
- 模式保留：无需解析的类型安全
- pandas 原生支持：零拷贝读取

#### 缺口检测算法

```python
def detect_gaps(df: pd.DataFrame, interval: str) -> list[tuple[datetime, datetime]]:
    """
    1. 计算预期间隔持续时间（例如，1h = 3600s）
    2. 计算连续时间戳之间的差异
    3. 标记 > 2x 预期持续时间的缺口（允许轻微的时钟偏差）
    4. 返回 (gap_start, gap_end) 元组列表
    """
```

**为什么是 2x 阈值？** 允许轻微的币安 API 不一致性，同时捕获真实缺口。

#### 增量更新策略

```
1. 加载现有缓存（如果存在）
2. 查找缓存中的最大时间戳
3. 从 max_timestamp + interval 到现在请求数据
4. 将新数据与缓存连接
5. 删除重复项（以防重叠）
6. 按时间戳排序
7. 写回 Parquet
```

**优化**：仅获取增量，而非每次加载时的完整历史数据集。

---

## 傅里叶变换方法

### 为什么选择 DCT 而非 FFT？

| 方法 | 边界处理 | 频率分辨率 | 实数/复数 |
|--------|-------------------|---------------------|--------------|
| FFT | 周期性扩展 | 高 | 复数 |
| **DCT** | **镜像扩展** | **高** | **实数** |
| DFT | 周期性扩展 | 高 | 复数 |
| 小波 | 取决于小波 | 多分辨率 | 实数 |

**DCT 在金融数据中的优势**：
1. **无周期性假设**：价格不是周期性的，因此 FFT 的周期性扩展会产生伪影
2. **实值**：更容易解释，没有虚部
3. **平滑边界**：镜像填充避免边缘不连续

### DCT 低通滤波算法

**位置**：`core/analysis/fourier.py`

#### 数学基础

对于长度为 N 的信号 x[n]：

1. **镜像填充**：
   ```
   原始：  [x₀, x₁, x₂, ..., xₙ]
   填充：  [xₙ, ..., x₂, x₁, x₀, x₁, x₂, ..., xₙ, xₙ₋₁, ..., x₁]
   ```
   确保边界处的平滑过渡。

2. **DCT 变换**：
   ```
   X[k] = Σ x[n] * cos(πk(2n+1)/(2N))
   ```
   转换到频域。

3. **低通滤波器**：
   ```
   H[k] = 1                           如果 k < k_cutoff
   H[k] = 0.5 * (1 + cos(π(k - k_cutoff) / taper_width))  如果 k_cutoff ≤ k < k_cutoff + taper
   H[k] = 0                           如果 k ≥ k_cutoff + taper
   ```
   渐变截止防止振铃（吉布斯现象）。

4. **逆 DCT**：
   ```
   x_filtered[n] = IDCT(X[k] * H[k])
   ```

5. **取消填充**：从滤波填充信号的中心提取原始长度。

#### 截止频率选择

给定所需的最小周期 `P_min`（以柱数为单位）：
```python
cutoff_freq = 1.0 / P_min
```

**示例**：对于 1h 数据上的 24 小时趋势：
- `P_min = 24 柱`
- `cutoff_freq = 1/24 ≈ 0.042`

**截止缩放**：用于微调的用户乘数：
- 缩放 = 0.5：截止的一半（更多平滑）
- 缩放 = 1.0：精确截止
- 缩放 = 2.0：截止的两倍（更少平滑）

---

## 频谱分析

### FFT 功率谱

**位置**：`core/analysis/spectral.py`

#### 实现

```python
def compute_fft_spectrum(signal, sampling_rate=1.0):
    """
    1. 应用汉宁窗以减少频谱泄漏
    2. 使用 scipy.fft.rfft 计算 FFT（实 FFT，半谱）
    3. 计算功率：|X[k]|²
    4. 按信号长度归一化
    5. 将频率仓转换为周期
    """
```

**为什么选择汉宁窗？** 减少有限信号长度的频谱泄漏，为非周期信号提供更好的频率分辨率。

#### 主导峰值检测

```python
def find_dominant_peaks(freqs, power, min_distance=10):
    """
    1. 使用 scipy.signal.find_peaks 带显著性阈值
    2. 按功率排序峰值（降序）
    3. 返回前 N 个峰值及其对应周期
    """
```

**min_distance 参数**：防止将谐波峰值识别为单独的周期。

### Welch 方法（PSD）

#### 为什么选择 Welch 而非周期图？

| 方法 | 方差 | 频率分辨率 | 计算成本 |
|--------|----------|---------------------|-------------------|
| 周期图 | 高 | 高 | 低 |
| **Welch** | **低** | **中等** | **中等** |
| 多锥度 | 非常低 | 中等 | 高 |

**Welch 优势**：
- **方差减少**：平均多个窗口
- **良好的频率分辨率**：重叠窗口保留细节
- **实用**：对于实时 UI 足够快

#### Welch 算法

```
1. 将信号分成重叠段
   - 窗口长度：W（例如，256 柱）
   - 重叠：O = overlap_ratio * W（例如，50%）
   - 段数：(N - W) / (W - O) + 1

2. 对于每个段：
   a. 应用汉宁窗
   b. 计算 FFT
   c. 计算功率谱

3. 平均所有段的功率谱
```

**参数选择**：
- **窗口长度（W）**：
  - 太小：频率分辨率差
  - 太大：段数少，方差高
  - 经验法则：W = N/4 到 N/8
- **重叠**：
  - 50% 标准，75% 用于更平滑的估计

### 滑动窗口主导周期

```python
def compute_sliding_dominant_period(signal, window_length=256, overlap_ratio=0.5):
    """
    1. 带重叠的窗口在信号上滑动
    2. 对于每个位置：
       a. 提取窗口
       b. 计算 Welch PSD
       c. 查找主导频率（最大功率）
       d. 转换为周期
    3. 返回主导周期的时间序列
    """
```

**用例**：识别状态变化（趋势 vs. 均值回归）。

**解释**：
- 上升的主导周期：市场转向更长周期（趋势）
- 下降的主导周期：市场转向更短周期（震荡）
- 稳定周期：一致的状态

---

## 信号生成逻辑

**位置**：`core/analysis/signals.py`

### 趋势跟踪信号

#### 进入逻辑

```python
def generate_entry_signal(close, smoothed, slope_threshold, slope_lookback):
    """
    多头进入条件：
    1. close[t] > smoothed[t]  （价格高于趋势）
    2. slope(smoothed) > slope_threshold  （趋势上升）
    3. 当前未持仓
    
    斜率计算为：
    slope[t] = (smoothed[t] - smoothed[t - lookback]) / lookback
    """
```

**设计理由**：
1. **价格高于趋势**：确保动量确认
2. **正斜率**：避免在下降趋势中进入
3. **斜率回溯**：平滑斜率以避免噪音引起的进入

#### 退出逻辑

```python
def generate_exit_signal(close, smoothed, stop_levels):
    """
    多头退出条件：
    1. close[t] < smoothed[t]  （价格低于趋势）
    或
    2. close[t] < stop_levels[t]  （触及止损）
    
    在信号后的下一柱开盘退出。
    """
```

**为什么下一柱执行？** 防止前视偏差。信号在柱收盘时生成，成交发生在下一柱开盘时。

### 止损集成

提供两种止损方法：

#### 1. 基于 ATR 的止损

```python
stop_loss = close - k_stop * ATR(period)
take_profit = close + k_profit * ATR(period)
```

**优势**：
- 适应波动率
- 在波动期更宽的止损（更少的假突破）
- 在平静期更紧的止损（保护利润）

**参数范围**：
- `k_stop`：1.5-3.0（典型为 2.0）
- `k_profit`：2.0-4.0（典型为 3.0）
- `period`：10-20（典型为 14）

#### 2. 基于残差的止损

```python
residual = close - smoothed
sigma = rolling_std(residual, window)
stop_loss = smoothed - k_stop * sigma
take_profit = smoothed + k_profit * sigma
```

**优势**：
- 适应价格-趋势偏差
- 在稳定趋势中更紧
- 在均值回归期间更宽

**何时使用**：
- ATR：用于突破/动量策略
- 残差：用于带紧密控制的趋势跟踪

---

## 回测器设计

**位置**：`core/backtest/engine.py`

### 向量化 vs. 事件驱动

| 方法 | 速度 | 灵活性 | 复杂性 |
|----------|-------|-------------|------------|
| **向量化** | **非常快** | **中等** | **低** |
| 事件驱动 | 慢 | 高 | 高 |

**向量化优势**：
- 对于简单策略快 100-1000 倍
- 更易调试（无状态机）
- 对于无复杂状态的趋势跟踪足够

**事件驱动优势**：
- 处理复杂订单类型（限价、止损限价）
- 投资组合再平衡逻辑
- 实时执行模拟

**设计选择**：向量化以提高速度，带增强功能以实现真实成交。

### 回测算法

```python
def run_backtest(signals, open_prices, high, low, close, timestamps, config):
    """
    1. 初始化状态
       - equity = initial_capital
       - position = 0
       - trades = []
    
    2. 对于每个柱 t：
       a. 如果 signal[t] == 1 且 position == 0：
          - entry_price = open[t+1]  （下一柱开盘）
          - entry_price += slippage
          - position_size = compute_size(equity, volatility)
          - position = position_size / entry_price
          - equity -= position_size * (1 + fee_rate)
          
       b. 如果 signal[t] == -1 且 position > 0：
          - exit_price = open[t+1]
          - exit_price -= slippage
          - equity += position * exit_price * (1 - fee_rate)
          - record_trade(entry, exit)
          - position = 0
       
       c. 如果 position > 0：
          - 检查止损：如果 low[t+1] < stop[t]，在止损处退出
          - 检查止盈：如果 high[t+1] > profit[t]，在止盈处退出
       
       d. 更新权益曲线：
          - equity_curve[t] = cash + position * close[t]
    
    3. 从权益曲线和交易计算指标
    """
```

### 真实成交建模

#### 1. 下一柱成交

**为什么不是同柱？** 防止前视偏差。实际上，您无法在生成信号的收盘价交易。

```python
# 在柱 t 收盘时的信号
if signal[t] == 1:
    # 在柱 t+1 开盘时成交
    entry_price = open[t+1]
```

#### 2. 止损/止盈柱内检查

```python
# 如果持仓且 stop < high < profit
if low[t] <= stop <= high[t]:
    exit_price = stop  # 假设先触及止损
    
elif low[t] <= profit <= high[t]:
    exit_price = profit
```

**假设**：保守 - 假设在同一柱中先触及止损后触及止盈。

#### 3. 手续费和滑点

```python
# 进入
cost = quantity * entry_price * (1 + fee_rate) + slippage

# 退出
proceeds = quantity * exit_price * (1 - fee_rate) - slippage
```

**滑点建模**：固定每笔交易或百分比。更复杂的模型可以使用基于成交量的滑点。

### 仓位规模

**位置**：`core/analysis/sizing.py`

#### 基于波动率的规模

```python
def compute_volatility_size(equity, volatility_target, current_volatility):
    """
    目标波动率：所需的投资组合波动率（例如，每天 2%）
    当前波动率：资产的近期波动率（例如，ATR 或 sigma）
    
    position_size = equity * (volatility_target / current_volatility)
    
    在交易中标准化风险。
    """
```

**示例**：
- 权益：$10,000
- 目标波动率：2%
- 资产波动率：4%
- 仓位规模：$10,000 * (0.02 / 0.04) = $5,000（50% 资金）

如果波动率翻倍，仓位规模减半 → 恒定风险。

---

## 优化框架

**位置**：`core/optimization/`

### 搜索算法

#### 1. 网格搜索

```python
def grid_search(param_space, objective_func):
    """
    1. 生成所有参数组合的笛卡尔积
    2. 对于每个组合：
       a. 运行回测
       b. 计算目标指标
    3. 按目标排序结果
    4. 返回最佳配置
    """
```

**优势**：
- 详尽：探索所有组合
- 可重现：确定性结果
- 简单：易于理解和验证

**劣势**：
- 计算成本高：组合数呈指数增长
- 对于大参数空间不切实际

**何时使用**：2-3 个参数，每个参数 5-10 个值

#### 2. 随机搜索

```python
def random_search(param_space, n_iterations, objective_func, seed):
    """
    1. 从每个参数的分布中随机采样
    2. 运行 n_iterations 次回测
    3. 按目标排序结果
    4. 返回最佳配置
    """
```

**优势**：
- 可扩展：固定迭代，不受参数空间影响
- 探索：可以发现网格点之间的良好区域
- 并行化：独立试验

**劣势**：
- 非确定性：除非播种
- 可能错过最佳点

**何时使用**：大参数空间，有限时间

#### 3. 贝叶斯优化

```python
def bayesian_optimization(param_space, n_iterations, objective_func):
    """
    1. 随机初始采样（例如，5 个点）
    2. 拟合高斯过程替代模型
    3. 对于每次迭代：
       a. 使用获取函数选择下一个点（EI, UCB）
       b. 评估目标
       c. 更新替代模型
    4. 返回观察到的最佳点
    """
```

**优势**：
- 高效：在更少的评估中找到良好的区域
- 智能：平衡探索与利用
- 适用于昂贵的目标函数

**劣势**：
- 顺序：评估不能完全并行化
- 复杂：需要替代模型调优

**何时使用**：昂贵的评估，寻找全局最优

### 步进式分析

```python
def walk_forward_analysis(data, param_space, train_ratio, n_splits):
    """
    1. 将数据分成 n_splits 折
    2. 对于每次拆分：
       a. 训练集：前 train_ratio 的数据
       b. 测试集：剩余数据
       c. 在训练集上优化参数
       d. 在测试集上评估最佳参数
    3. 汇总样本外绩效
    """
```

**类型**：
- **锚定**：训练集从开始增长
- **滚动**：固定大小的训练窗口向前滑动

**何时使用**：
- 锚定：趋势市场，历史相关
- 滚动：状态变化市场，近期更相关

### 蒙特卡洛重采样

```python
def monte_carlo_resample(equity_curve, n_simulations, block_size):
    """
    1. 将权益曲线分成大小为 block_size 的块
    2. 对于每次模拟：
       a. 带替换地随机采样块
       b. 连接以形成新的权益曲线
       c. 计算指标
    3. 返回指标分布
    """
```

**块大小选择**：
- 太小：破坏自相关
- 太大：重采样不足
- 经验法则：平均交易持续时间的 2-5 倍

**用途**：评估指标的置信区间，检测过拟合

---

## 投资组合管理

**位置**：`core/portfolio/`

### 加权方案

#### 1. 等权

```python
weights = {symbol: 1/N for symbol in symbols}
```

**优势**：简单，多样化
**劣势**：忽略风险差异

#### 2. 波动率加权

```python
inv_vols = {symbol: 1/volatility[symbol] for symbol in symbols}
total = sum(inv_vols.values())
weights = {symbol: inv_vols[symbol]/total for symbol in symbols}
```

**优势**：标准化风险贡献
**劣势**：可能集中在低波动率资产

#### 3. 风险平价

```python
# 迭代求解：每个资产的风险贡献 = 总风险 / N
# 使用优化：minimize(sum((RC_i - target)^2))
```

**优势**：真正的风险平衡
**劣势**：需要协方差矩阵，计算成本高

#### 4. 市值加权

```python
weights = {symbol: market_cap[symbol] / total_market_cap for symbol in symbols}
```

**优势**：跟踪市场基准
**劣势**：集中在大型资产

### 再平衡

```python
def rebalance(current_weights, target_weights, threshold):
    """
    如果任何权重偏离超过阈值：
    1. 计算所需的交易
    2. 执行交易（有手续费）
    3. 更新持仓
    """
```

**频率**：
- 高（每日）：跟踪目标更接近，更多手续费
- 低（月度）：手续费更少，漂移更多

**阈值**：
- 5-10%：平衡控制与成本
- 更低：更频繁的再平衡
- 更高：手续费更少，但偏离目标

### 风险分析

#### 1. 相关性矩阵

```python
correlation_matrix = returns.corr()
```

**解释**：
- 高正相关（> 0.7）：资产同向移动，多样化较少
- 低/负相关（< 0.3）：良好的多样化

#### 2. 分散化比率

```python
diversification_ratio = (sum(weights * volatilities)) / portfolio_volatility
```

**解释**：
- DR > 1：组合效应，降低风险
- DR = 1：无多样化好处
- 更高更好

---

## 性能考虑

### 向量化优化

**使用 NumPy 而非循环**：

```python
# 慢：Python 循环
for i in range(len(prices)):
    returns[i] = (prices[i] - prices[i-1]) / prices[i-1]

# 快：NumPy 向量化
returns = np.diff(prices) / prices[:-1]
```

**加速**：10-100 倍

### 内存管理

**技巧**：
1. 使用 `float32` 而非 `float64`（如果精度允许）
2. 在不再需要时删除大数组
3. 使用生成器进行迭代
4. 对于大数据集使用 Dask/Vaex

### 并行化

**策略**：
1. **参数搜索**：并行评估（`joblib`、`multiprocessing`）
2. **多币种回测**：并行逐币种运行
3. **蒙特卡洛**：并行模拟

**示例**：

```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(run_backtest)(params) for params in param_grid
)
```

---

## 总结

该架构在速度、灵活性和可维护性之间平衡：

- **数据层**：高效缓存，带稳健的错误处理
- **分析层**：快速向量化操作，理论上合理的方法
- **执行层**：真实的回测，带实用的优化工具
- **UI 层**：交互式和响应式，支持会话持久化

**设计哲学**：简单胜于复杂，但不过于简单。每个组件都解决实际的交易研究需求，同时保持代码清晰和可测试。

---

有关参数详细信息，请参阅 [CONFIGURATION.zh-CN.md](CONFIGURATION.zh-CN.md)。
有关常见问题，请参阅 [FAQ.zh-CN.md](FAQ.zh-CN.md)。
