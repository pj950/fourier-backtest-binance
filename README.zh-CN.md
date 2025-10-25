# 币安傅里叶回测器

[English](README.md) | [中文](README.zh-CN.md)

一个综合性的 Python 3.11+ 交易策略平台，结合高级傅里叶分析、频谱平滑和算法回测，配备交互式 Streamlit UI。专为系统性交易研究、参数优化和投资组合风险管理而设计。

## 🎯 项目概述

本平台使量化交易者和研究人员能够：
- **分析** 使用傅里叶变换和频谱方法分析加密货币价格数据
- **开发** 带动态止损和多时间框架确认的趋势跟踪策略
- **回测** 使用真实的成交、手续费和滑点模型进行策略回测
- **优化** 使用网格搜索、随机搜索或贝叶斯优化进行参数调优
- **管理** 具有高级风险分析的多币种投资组合

### 主要应用场景
- 使用 FFT 和 Welch 频谱分析识别主导市场周期
- 使用基于 DCT 的低通平滑过滤价格噪音
- 构建具有基于波动率的仓位规模的稳健交易策略
- 步进式分析和蒙特卡洛验证以确保策略稳健性
- 使用风险平价、波动率缩放或市值加权构建投资组合

## ✨ 功能特性

### 📊 数据与分析
- **数据获取**：从币安 REST API 获取 30m、1h 和 4h K线数据，自动重试和限速
- **智能缓存**：基于 Parquet 的缓存，支持增量更新和自动缺口检测/回填
- **DCT 平滑**：基于离散余弦变换的低通平滑，带镜像填充和渐变截止
- **FFT 频谱分析**：全局功率谱，标记主导频率峰值（以柱数/小时为单位）
- **滑动窗口 PSD**：使用 Welch 方法随时间提取局部主导周期
- **频谱热图**：时频分析显示主导周期如何演变

### 📈 回测与交易
- **动态止损带**：基于 ATR 和残差的止损，可配置乘数
- **信号生成**：带斜率和波动率过滤器的趋势跟踪信号
- **多时间框架确认**：在 30m 上执行，使用 1h/4h 趋势过滤器获得更高概率的设置
- **高级退出**：基于时间的止损、部分止盈缩放、斜率反转确认
- **动态仓位规模**：基于波动率（ATR/sigma）、固定风险、可选加仓
- **做空/期货交易**：可选做空交易模式，支持按交易所配置手续费
- **向量化回测器**：快速、真实的回测，下一柱成交、手续费和滑点
- **绩效指标**：19 个指标，包括夏普比率、索提诺比率、胜率、盈利因子等
- **交易分析**：MAE/MFE 跟踪、权益曲线、完整的交易日志及退出原因

### 🔬 参数优化 (M8)
- **网格/随机/贝叶斯搜索**：用于参数调优的多种优化算法
- **步进式分析**：滚动或锚定验证，带训练/测试拆分
- **蒙特卡洛重采样**：块自助法进行稳健性评估
- **丰富的可视化**：热图、前沿图、参数重要性、进度跟踪
- **导出功能**：CSV/Parquet 导出最佳配置
- **可重现种子**：所有方法支持种子以实现可重现性
- **批处理**：基于排行榜的参数组合评估

### 📊 投资组合与风险管理 (M9)
- **多币种回测**：并行逐币种运行，投资组合聚合
- **加权方案**：等权、波动率缩放、风险平价、市值加权
- **动态再平衡**：可配置频率和基于阈值的再平衡
- **相关性分析**：静态和滚动相关性矩阵
- **风险分析**：分散化比率、集中度指标、风险贡献
- **敞口跟踪**：板块敞口和 beta 计算
- **投资组合指标**：全面的投资组合级别绩效和风险指标
- **交互式投资组合 UI**：币种篮子选择、权重可视化、权益曲线

### 🖥️ 基础设施
- **交互式 UI**：基于 Streamlit 的界面，实时参数调整
- **多种模式**：回测、优化和投资组合选项卡
- **预设管理**：保存/加载参数配置
- **会话持久化**：自动恢复上次会话状态
- **实时流**：基于 WebSocket 的实时价格更新（可选）
- **强大的错误处理**：指数退避重试，可配置限制
- **类型安全**：完整的类型提示，带 mypy 验证
- **全面测试**：150+ 测试覆盖所有组件

## 🏗️ 架构

该平台遵循分层架构，清晰分离关注点：

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI 层                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   回测标签   │  │   优化标签   │  │  投资组合    │      │
│  │              │  │              │  │    标签      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                    核心分析层                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    傅里叶    │  │    频谱      │  │    信号      │      │
│  │    平滑      │  │    分析      │  │    生成      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   止损带     │  │     MTF      │  │    退出      │      │
│  │ (ATR/残差)   │  │   对齐       │  │    策略      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐                                            │
│  │   仓位       │                                            │
│  │   规模       │                                            │
│  └──────────────┘                                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                执行与优化层                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   回测       │  │   优化       │  │  投资组合    │      │
│  │   引擎       │  │   算法       │  │   管理器     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                      数据层                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    币安      │  │    缓存      │  │   加载器     │      │
│  │   客户端     │  │  (Parquet)   │  │  统一 API    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐                                            │
│  │  WebSocket   │                                            │
│  │    流式      │                                            │
│  └──────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

### 架构组件

#### 1. 数据层
- **币安客户端** (`core/data/binance_client.py`)：REST API 客户端，带指数退避重试
- **缓存系统** (`core/data/cache.py`)：基于 Parquet 的存储，带缺口检测和增量更新
- **数据加载器** (`core/data/loader.py`)：统一的 `load_klines()` API，抽象获取和缓存逻辑
- **WebSocket 流式** (`core/data/streaming.py`)：通过 WebSocket 实时 K线 更新 (M6)

#### 2. 核心分析模块
- **傅里叶变换** (`core/analysis/fourier.py`)：基于 DCT 的平滑，带镜像填充
- **频谱分析** (`core/analysis/spectral.py`)：FFT、Welch PSD、滑动窗口主导周期
- **信号生成** (`core/analysis/signals.py`)：带斜率/波动率过滤器的趋势跟踪逻辑
- **止损带** (`core/analysis/stops.py`)：基于 ATR 和残差的动态止损
- **多时间框架** (`core/analysis/mtf.py`)：时间框架对齐和趋势确认 (M7)
- **退出策略** (`core/analysis/exits.py`)：基于时间、部分止盈、斜率反转 (M7)
- **仓位规模** (`core/analysis/sizing.py`)：基于波动率、固定风险、加仓 (M7)

#### 3. 执行与优化层
- **回测引擎** (`core/backtest/engine.py`)：向量化回测，真实成交和手续费
- **优化** (`core/optimization/`)：网格/随机/贝叶斯搜索、步进式、蒙特卡洛 (M8)
- **投资组合管理器** (`core/portfolio/`)：多币种执行、加权、风险分析 (M9)

#### 4. UI 层
- **主标签** (`app/ui/main.py`)：回测界面，带图表和控件
- **优化标签** (`app/ui/optimization_tab.py`)：参数调优和稳健性测试 (M8)
- **投资组合标签** (`app/ui/portfolio_tab.py`)：多币种投资组合管理 (M9)
- **实时模式** (`app/ui/live.py`)：实时流式和增量计算 (M6)

## 🚀 技术栈

- **语言**：Python 3.11+
- **UI 框架**：Streamlit 1.28+
- **数据处理**：pandas, numpy
- **分析**：scipy (FFT, Welch)，自定义 DCT 实现
- **可视化**：plotly, matplotlib, seaborn
- **HTTP 客户端**：httpx 带 tenacity 重试
- **存储**：Parquet (pyarrow)，DuckDB 用于查询
- **配置**：pydantic-settings, python-dotenv
- **WebSocket**：websocket-client
- **测试**：pytest (150+ 测试)
- **代码质量**：ruff (linting), mypy (type checking)
- **依赖管理**：Poetry
- **容器化**：Docker

## 📦 安装

### 前置要求

- **Python 3.11 或更高版本**（使用 `python --version` 检查）
- **Poetry**（推荐）或 pip
- **Git** 用于克隆仓库
- **Docker**（可选，用于容器化部署）

### 选项 1：使用 Poetry（推荐）

Poetry 自动处理虚拟环境和依赖锁定。

```bash
# 如果没有 Poetry，先安装它
curl -sSL https://install.python-poetry.org | python3 -

# 克隆仓库
git clone <repository-url>
cd binance-fourier-backtester

# 安装依赖（自动创建虚拟环境）
poetry install

# 激活虚拟环境
poetry shell

# 验证安装
python -c "import streamlit; print('OK')"
```

### 选项 2：使用 pip

```bash
# 克隆仓库
git clone <repository-url>
cd binance-fourier-backtester

# 创建虚拟环境
python3.11 -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import streamlit; print('OK')"
```

### 选项 3：使用 Docker

```bash
# 克隆仓库
git clone <repository-url>
cd binance-fourier-backtester

# 构建 Docker 镜像
docker build -t binance-backtester .

# 运行容器（挂载数据目录以持久化）
docker run -p 8501:8501 -v $(pwd)/data:/app/data binance-backtester

# 在浏览器访问 UI：http://localhost:8501
```

## ⚙️ 配置

### 环境变量

复制示例环境文件并根据需要自定义：

```bash
cp .env.example .env
```

`.env` 中的关键配置选项：

```bash
# 数据路径
BASE_PATH=./data
CACHE_DIR=./data/cache

# 币安 API 设置
BINANCE_BASE_URL=https://api.binance.com
BINANCE_RATE_LIMIT_PER_MINUTE=1200
BINANCE_REQUEST_TIMEOUT=30

# 交易默认值
DEFAULT_FEE_RATE=0.001          # 0.1% 交易手续费
DEFAULT_SLIPPAGE_BPS=5.0        # 5 个基点滑点

# 重试设置
MAX_RETRY_ATTEMPTS=5
RETRY_INITIAL_WAIT=1.0          # 初始重试等待时间（秒）
RETRY_MAX_WAIT=60.0             # 最大重试等待时间（秒）

# UI 预设存储
PRESET_STORAGE_PATH=./data/presets/presets.yaml
LAST_SESSION_STATE_PATH=./data/presets/last_state.yaml
```

### 首次设置

1. **创建数据目录**（首次运行时自动完成）：
   ```bash
   mkdir -p data/cache data/presets
   ```

2. **配置 API 访问**（可选）：
   - 公共市场数据无需 API 密钥
   - 速率限制：默认 1200 次/分钟
   - 要获得更高限制，请创建币安账户并将 API 密钥添加到 `.env`

3. **测试配置**：
   ```bash
   poetry run python -c "from config.settings import settings; print(f'缓存目录: {settings.cache_dir}')"
   ```

## 🎮 首次运行与使用

### 启动应用程序

```bash
# 使用 Poetry
poetry run streamlit run app/ui/main.py

# 或在 Poetry shell 中
streamlit run app/ui/main.py

# 使用 Docker
docker run -p 8501:8501 -v $(pwd)/data:/app/data binance-backtester
```

UI 将在浏览器中打开，地址为 `http://localhost:8501`。

### 分步使用指南

#### 1. **数据加载**（侧边栏）

**币种选择：**
- 从支持的币种中选择：BTCUSDT、ETHUSDT 等
- 币种可以在 `core/data/loader.py` 中扩展

**时间间隔选择：**
- `30m`：30 分钟 K线（短期交易）
- `1h`：1 小时 K线（日内策略）
- `4h`：4 小时 K线（波段交易）

**日期范围：**
- 开始日期：要获取的数据起始日期（历史数据从 2020 年开始）
- 结束日期：数据范围结束日期（最多到当前日期）
- 推荐：从 3-6 个月开始测试

**加载数据按钮：**
- 首次点击：从币安 API 获取数据并在本地缓存
- 后续点击：使用缓存数据（快速）
- 强制刷新：绕过缓存并获取新数据（较慢）

**数据获取流程：**
```
点击"加载数据" → 缓存检查 → API 获取（如需要）→ 缺口检测 → 回填 → 显示
```

#### 2. **策略参数**（侧边栏）

**傅里叶平滑：**
- **最小趋势周期（小时）**：要保留的最小周期
  - 较低值（12-24h）：更敏感，更多交易
  - 较高值（48-96h）：更平滑的趋势，更少交易
- **截止缩放**：平滑激进程度（0.5-2.0）
  - 较低：较少平滑，紧跟价格
  - 较高：更多平滑，忽略短期噪音

**止损配置：**
- **止损类型**：
  - `ATR`：基于平均真实范围（波动率）
  - `Residual`：基于价格-趋势偏差
- **ATR 周期**：ATR 计算的回溯期（典型为 14）
- **残差窗口**：残差标准差的回溯期（典型为 20）
- **K 止损**：止损乘数（1.5-3.0）
  - 较低：更紧的止损，更多退出
  - 较高：更宽的止损，更少退出
- **K 止盈**：止盈乘数（2.0-4.0）
  - 应 > K 止损以获得正风险/收益比

**信号生成：**
- **斜率阈值**：进入的最小趋势斜率（0.0 = 任何方向）
- **斜率回溯**：计算斜率的柱数（1-5）

**回测配置：**
- **初始资金**：起始投资组合价值（典型为 $10,000）
- **手续费率**：交易佣金（0.1% = 10 个基点）
- **滑点**：每笔交易的预期滑点（典型为 0.05%）

#### 3. **运行回测**

加载数据后点击 **"运行回测"**。系统将：
1. 对价格序列应用 DCT 平滑
2. 计算止损/止盈带
3. 生成进入/退出信号
4. 使用真实成交模拟交易
5. 计算绩效指标
6. 显示结果

#### 4. **理解图表**

**价格 + 平滑趋势图：**
- **K线**：OHLC 价格数据
- **蓝线**：DCT 平滑趋势
- **绿色标记 (▲)**：多头进入信号
- **红色标记 (▼)**：多头退出信号
- **解读**：当价格突破趋势上方且斜率为正时进入

**FFT 频谱：**
- **X 轴**：频率（每柱周期数）
- **Y 轴**：功率（幅度）
- **峰值**：数据中的主导周期
- **标签**：以柱数和小时为单位的周期
- **解读**：识别主要市场周期（如 24h 日周期、周周期）

**滑动主导周期：**
- **X 轴**：时间（柱）
- **Y 轴**：主导周期（柱）
- **线**：每个时间点最强大的周期
- **解读**：显示市场状态如何变化（趋势 vs. 震荡）

**Welch PSD 热图：**
- **X 轴**：时间
- **Y 轴**：周期（柱）
- **颜色**：功率密度（红色 = 高，蓝色 = 低）
- **解读**：时频图显示周期演变

**权益曲线：**
- **线**：投资组合价值随时间变化
- **阴影区域**：回撤期
- **解读**：策略绩效和风险的视觉评估

#### 5. **绩效指标术语表**

**收益：**
- `总收益`：绝对盈亏（货币）
- `累计收益`：期间总收益率（%）
- `年化收益`：外推至 1 年的收益率（%）

**风险指标：**
- `最大回撤`：最大峰谷跌幅（%）
- `最大回撤 $`：最大峰谷跌幅（货币）
- `夏普比率`：风险调整收益（收益 / 波动率）
  - < 0：亏损策略
  - 0-1：差到可接受
  - 1-2：良好
  - \> 2：优秀
- `索提诺比率`：收益 / 下行偏差（忽略上行波动率）
  - 越高越好，仅惩罚下行
- `Calmar 比率`：年化收益 / 最大回撤
  - 衡量每单位最坏情况风险的收益

**交易统计：**
- `交易次数`：完成的往返交易总数
- `胜率`：盈利交易的百分比
- `盈利因子`：总盈利 / 总亏损
  - < 1.0：整体亏损
  - 1.0-1.5：边际
  - 1.5-2.0：良好
  - \> 2.0：强劲
- `平均盈利`：每笔盈利交易的平均利润
- `平均亏损`：每笔亏损交易的平均损失
- `平均持仓柱数`：平均交易持续时间

**执行质量：**
- `MAE`（最大不利偏移）：交易期间的平均最差价格
- `MFE`（最大有利偏移）：交易期间的平均最佳价格
- `MAE/MFE 分析`：帮助优化止损位置

#### 6. **交易日志与导出**

**交易详情表：**
- 进入/退出时间和价格
- 每笔交易的盈亏
- 持续时间（持仓柱数）
- 退出原因（止损、止盈、信号）

**CSV 下载：**
- 点击"下载交易 CSV"导出完整交易日志
- 包含所有列用于外部分析
- 兼容 Excel、Python、R

#### 7. **参数预设**

**保存预设：**
1. 按您的喜好配置所有参数
2. 展开"💾 预设与持久化"
3. 输入预设名称
4. 点击"保存当前配置"
5. 预设存储在 YAML 中供将来使用

**加载预设：**
1. 展开"💾 预设与持久化"
2. 从下拉菜单中选择预设
3. 点击"加载所选预设"
4. 所有参数更新为保存的值

**会话持久化：**
- 退出时自动保存上次使用的参数
- 下次启动应用时恢复
- 通过清除 `LAST_SESSION_STATE_PATH` 禁用

### 多时间框架策略（高级）

使用 **多时间框架** 部分过滤交易：

1. 启用"使用多时间框架确认"
2. 加载更高时间框架数据（例如，如果交易 30m，则加载 1h 或 4h）
3. 为每个时间框架设置平滑参数
4. 仅当所有时间框架都同意方向时才执行交易

**示例：**
- 在 30m 图表上交易
- 用 1h 趋势确认：仅在 1h 趋势向上时做多
- 用 4h 趋势确认：仅在 4h 趋势向上时做多
- 结果：更少但更高概率的交易

### 优化标签 (M8)

通过侧边栏访问：**"优化"**

**功能：**
- **网格搜索**：对参数网格进行详尽搜索
- **随机搜索**：采样 N 个随机组合
- **贝叶斯优化**：使用高斯过程的智能搜索
- **步进式分析**：滚动窗口验证
- **蒙特卡洛**：块自助法重采样以确保稳健性

**工作流程：**
1. 加载数据（与主标签相同）
2. 选择优化方法
3. 定义参数范围
4. 选择目标指标（夏普、收益等）
5. 点击"运行优化"
6. 查看热图、前沿图、参数重要性
7. 将顶级配置导出到 CSV

详见 [IMPLEMENTATION_M8.md](IMPLEMENTATION_M8.md)。

### 投资组合标签 (M9)

通过侧边栏访问：**"投资组合"**

**功能：**
- 多币种篮子选择
- 加权方法：等权、波动率、风险平价、市值
- 相关性矩阵和风险分析
- 投资组合级别指标和权益曲线

**工作流程：**
1. 从列表中选择多个币种
2. 选择加权方法
3. 设置再平衡频率
4. 配置投资组合级别参数
5. 点击"运行投资组合回测"
6. 查看聚合结果和逐币种细分

详见 [IMPLEMENTATION_M9.md](IMPLEMENTATION_M9.md)。

## 🐛 故障排查

### 常见错误

#### 1. **币安速率限制错误**

**错误消息：**
```
BinanceRateLimitError: Rate limit exceeded. Please wait 60s.
```

**解决方案：**
- 等待建议的时间后重试
- 减少日期范围以获取更少数据
- 使用缓存数据（取消选中"强制刷新"）
- 避免并行请求多个币种
- 考虑添加币安 API 密钥以获得更高限制

**预防：**
- 让缓存在多个会话中逐步填充
- 最初使用较窄的日期范围
- 仅在必要时启用"强制刷新"

#### 2. **缓存损坏**

**错误消息：**
```
ArrowInvalid: Failed to read parquet file
```

**解决方案：**
- 删除损坏的缓存文件：
  ```bash
  rm data/cache/BTCUSDT_1h.parquet
  ```
- 或清除整个缓存：
  ```bash
  rm -rf data/cache/*
  ```
- 选中"强制刷新"重新加载数据

**预防：**
- 确保有足够的磁盘空间
- 避免手动编辑缓存文件
- 使用正确的关闭方式（Ctrl+C）以避免不完整的写入

#### 3. **数据缺口**

**症状：** 图表中缺少数据或回测结果异常

**解决方案：**
- 启用"强制刷新"以触发缺口检测和回填
- 检查币安 API 状态（可能有历史中断）
- 缩小日期范围以排除问题时段

**检测：**
- 系统自动检测 > 2x 间隔持续时间的缺口
- 记录缺口检测：检查终端输出

#### 4. **依赖冲突**

**错误消息：**
```
ModuleNotFoundError: No module named 'streamlit'
```

**解决方案：**
- 确保虚拟环境已激活：
  ```bash
  poetry shell
  ```
- 重新安装依赖：
  ```bash
  poetry install
  ```
- 检查 Python 版本：
  ```bash
  python --version  # 应为 3.11+
  ```

**Poetry 问题：**
- 清除锁定文件并重新安装：
  ```bash
  rm poetry.lock
  poetry install
  ```

#### 5. **内存不足**

**症状：** 大型回测或优化期间崩溃

**解决方案：**
- 减少日期范围
- 减少优化迭代
- 使用较小的频谱分析窗口长度
- 增加系统交换空间
- 分批运行优化

**内存使用估计：**
- 1 年的 1h 数据：约 50MB
- 网格搜索（1000 个组合）：约 500MB-2GB，取决于数据
- 10 个币种的投资组合：约 200MB-1GB

#### 6. **Streamlit 端口已被占用**

**错误消息：**
```
Address already in use
```

**解决方案：**
- 终止现有的 Streamlit 进程：
  ```bash
  pkill -f streamlit
  ```
- 使用不同端口：
  ```bash
  streamlit run app/ui/main.py --server.port 8502
  ```

#### 7. **WebSocket 连接失败**（实时模式）

**错误消息：**
```
WebSocket connection failed
```

**解决方案：**
- 检查互联网连接
- 验证币安 WebSocket 端点可访问
- 禁用实时模式并仅使用历史数据
- 检查防火墙设置

### 性能问题

**数据加载慢：**
- 首次加载正常（API 获取）
- 后续加载应很快（缓存）
- 如果初始加载超时，请减少日期范围

**回测执行慢：**
- 大型数据集（1+ 年）正常
- 启用向量化操作（默认）
- 减少频谱分析窗口大小
- 使用更高的时间框架（4h vs 30m）

**优化慢：**
- 贝叶斯（迭代）预期较慢
- 使用随机搜索以获得更快结果
- 减少参数空间
- 使用较少折数的步进式

### 日志和调试

**启用调试日志：**
```python
# 在 config/settings.py 中
import logging
logging.basicConfig(level=logging.DEBUG)
```

**检查缓存状态：**
```bash
ls -lh data/cache/
```

**验证预设文件：**
```bash
cat data/presets/presets.yaml
cat data/presets/last_state.yaml
```

**测试数据加载：**
```python
from core.data.loader import load_klines
from datetime import datetime, UTC

df = load_klines(
    symbol="BTCUSDT",
    interval="1h",
    start=datetime(2024, 1, 1, tzinfo=UTC),
    end=datetime(2024, 2, 1, tzinfo=UTC)
)
print(f"已加载 {len(df)} 行")
```

## 📁 项目结构

```
.
├── app/                            # 用户界面层
│   └── ui/
│       ├── main.py                 # 主 Streamlit UI，带回测标签
│       ├── optimization_tab.py     # 参数优化界面 (M8)
│       ├── portfolio_tab.py        # 投资组合管理界面 (M9)
│       └── live.py                 # 实时流式协调器 (M6)
│
├── core/                           # 核心业务逻辑
│   ├── analysis/                   # 分析和信号生成
│   │   ├── fourier.py              # 基于 DCT 的平滑函数
│   │   ├── spectral.py             # FFT/Welch PSD 分析和可视化
│   │   ├── signals.py              # 趋势跟踪信号生成
│   │   ├── stops.py                # 动态止损/止盈带
│   │   ├── mtf.py                  # 多时间框架对齐 (M7)
│   │   ├── exits.py                # 高级退出策略 (M7)
│   │   └── sizing.py               # 仓位规模算法 (M7)
│   │
│   ├── backtest/                   # 回测引擎
│   │   └── engine.py               # 向量化回测，带 M7 增强
│   │
│   ├── data/                       # 数据获取和管理
│   │   ├── binance_client.py       # 币安 REST API 客户端
│   │   ├── cache.py                # Parquet 缓存，带缺口检测
│   │   ├── loader.py               # 统一数据加载 API
│   │   ├── streaming.py            # WebSocket 流式 (M6)
│   │   └── exceptions.py           # 自定义异常类型
│   │
│   ├── optimization/               # 参数优化框架 (M8)
│   │   ├── search.py               # 网格/随机/贝叶斯搜索引擎
│   │   ├── params.py               # 参数空间定义
│   │   ├── walkforward.py          # 步进式分析
│   │   ├── monte_carlo.py          # 蒙特卡洛重采样
│   │   ├── runner.py               # 优化编排
│   │   └── visualization.py        # 优化结果可视化
│   │
│   ├── portfolio/                  # 投资组合管理 (M9)
│   │   ├── portfolio.py            # 主投资组合管理器
│   │   ├── weights.py              # 加权方案（等权、风险平价等）
│   │   ├── analytics.py            # 风险和相关性分析
│   │   └── executor.py             # 并行回测执行
│   │
│   └── utils/                      # 实用函数
│       └── time.py                 # UTC 时间工具
│
├── config/                         # 配置管理
│   ├── settings.py                 # Pydantic 设置，带 .env 支持
│   └── presets.py                  # 预设和会话状态管理
│
├── tests/                          # 测试套件（150+ 测试）
│   ├── test_backtest.py            # 回测引擎测试
│   ├── test_backtest_enhanced.py   # 增强回测测试 (M7)
│   ├── test_data_fetch_cache.py    # 数据层测试
│   ├── test_fourier.py             # 傅里叶平滑测试
│   ├── test_spectral.py            # 频谱分析测试
│   ├── test_mtf.py                 # 多时间框架测试 (M7)
│   ├── test_exits.py               # 退出策略测试 (M7)
│   ├── test_sizing.py              # 仓位规模测试 (M7)
│   ├── test_optimization.py        # 优化测试 (M8)
│   ├── test_portfolio.py           # 投资组合测试 (M9)
│   ├── test_portfolio_weights.py   # 加权方案测试 (M9)
│   ├── test_portfolio_analytics.py # 分析测试 (M9)
│   └── test_strategy_integration.py # 集成测试 (M7)
│
├── examples/                       # 示例脚本
│   ├── mtf_strategy_example.py     # 完整 MTF 策略示例 (M7)
│   ├── optimization_example.py     # 优化工作流示例 (M8)
│   └── portfolio_example.py        # 投资组合管理示例 (M9)
│
├── docs/                           # 附加文档
│   ├── ARCHITECTURE.md             # 傅里叶方法和设计深入探讨
│   ├── CONFIGURATION.md            # 所有参数，带默认值和范围
│   └── FAQ.md                      # 常见问题
│
├── IMPLEMENTATION_M7.md            # M7 功能文档
├── IMPLEMENTATION_M8.md            # M8 功能文档
├── IMPLEMENTATION_M9.md            # M9 功能文档
├── IMPLEMENTATION_SUMMARY.md       # M1 实现摘要
├── M7_SUMMARY.md                   # M7 里程碑摘要
├── M9_SUMMARY.md                   # M9 里程碑摘要
│
├── Dockerfile                      # Docker 容器配置
├── docker-compose.yml              # Docker Compose 设置（如果存在）
├── pyproject.toml                  # Poetry 依赖和工具配置
├── requirements.txt                # Pip 依赖（从 pyproject.toml 生成）
├── .env.example                    # 环境变量模板
├── .gitignore                      # Git 忽略规则
├── .ruff.toml                      # Ruff linter 配置
├── mypy.ini                        # Mypy 类型检查器配置
└── README.md                       # 此文件
```

### 模块说明

**核心模块：**
- `fourier.py`：实现基于 DCT 的低通滤波，带镜像填充以避免边缘效应
- `spectral.py`：FFT 和 Welch PSD 用于频率分析，识别主导市场周期
- `signals.py`：带斜率和波动率过滤的趋势跟踪进入/退出逻辑
- `stops.py`：计算基于 ATR（波动率）和残差（偏差）的止损带
- `engine.py`：向量化回测器在下一柱开盘时模拟成交，带手续费和滑点

**高级功能：**
- `mtf.py`：对齐多个时间框架并检查趋势一致性
- `exits.py`：基于时间的退出、部分止盈、斜率反转确认
- `sizing.py`：波动率缩放仓位规模以标准化每笔交易的风险
- `optimization/`：参数调优框架，带多种搜索算法
- `portfolio/`：多币种管理，带风险分析和再平衡

## 🧪 测试

运行全面的测试套件：

```bash
# 运行所有测试
poetry run pytest

# 带覆盖率运行
poetry run pytest --cov=core --cov=config --cov=app

# 运行特定测试文件
poetry run pytest tests/test_backtest.py

# 带详细输出运行
poetry run pytest -v

# 仅运行优化测试
poetry run pytest tests/test_optimization.py

# 使用标记运行（如果定义）
poetry run pytest -m "not slow"
```

### 测试覆盖率

- **150+ 测试**覆盖所有模块
- **核心分析**：傅里叶、频谱、信号、止损
- **回测**：引擎逻辑、增强功能
- **数据层**：API 客户端、缓存、缺口检测
- **优化**：搜索算法、步进式、蒙特卡洛
- **投资组合**：加权、分析、执行
- **集成**：端到端策略工作流程

## 🛠️ 开发

### 代码质量工具

```bash
# 格式化代码（自动修复）
poetry run ruff format .

# 检查代码（尽可能自动修复）
poetry run ruff check . --fix

# 不自动修复的检查
poetry run ruff check .

# 类型检查
poetry run mypy .

# 运行所有检查
poetry run ruff format . && poetry run ruff check . && poetry run mypy . && poetry run pytest
```

### Pre-commit 设置（可选）

```bash
# 安装 pre-commit
poetry add --group dev pre-commit

# 安装钩子
poetry run pre-commit install

# 手动运行
poetry run pre-commit run --all-files
```

### 添加新币种

编辑 `core/data/loader.py`：

```python
SUPPORTED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"}  # 在此添加币种
```

### 添加新时间间隔

编辑 `core/data/loader.py`：

```python
SUPPORTED_INTERVALS = {"30m", "1h", "4h", "1d"}  # 在此添加时间间隔
```

确保币安 API 支持该时间间隔。

### 扩展策略

在 `core/analysis/signals.py` 中创建新的信号生成器：

```python
def generate_mean_reversion_signals(
    close: np.ndarray,
    smoothed: np.ndarray,
    threshold: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """生成均值回归信号。"""
    deviation = (close - smoothed) / smoothed
    entries = deviation < -threshold  # 超卖时买入
    exits = deviation > threshold     # 超买时卖出
    return entries, exits
```

在 `app/ui/main.py` 中集成到 UI。

## 📚 API 参考

有关详细的 API 文档，请参阅：
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**：深入的技术设计和算法
- **[CONFIGURATION.md](docs/CONFIGURATION.md)**：完整的参数参考，带范围
- **[FAQ.md](docs/FAQ.md)**：常见问题和答案

### 快速 API 示例

```python
from datetime import datetime, UTC
from core.data.loader import load_klines
from core.analysis.fourier import smooth_price_series
from core.analysis.stops import compute_atr_stops
from core.analysis.signals import generate_signals_with_stops
from core.backtest.engine import BacktestConfig, run_backtest

# 加载数据
df = load_klines(
    symbol="BTCUSDT",
    interval="1h",
    start=datetime(2024, 1, 1, tzinfo=UTC),
    end=datetime(2024, 6, 1, tzinfo=UTC)
)

# 提取 OHLCV
close = df["close"].values
high = df["high"].values
low = df["low"].values
open_prices = df["open"].values
timestamps = df["open_time"]

# 平滑价格
smoothed = smooth_price_series(
    prices=close,
    min_period_bars=24,  # 24 小时趋势
    cutoff_scale=1.0
)

# 计算止损带
long_stop, long_profit, _, _ = compute_atr_stops(
    close=close,
    high=high,
    low=low,
    atr_period=14,
    k_stop=2.0,
    k_profit=3.0
)

# 生成信号
signals = generate_signals_with_stops(
    close=close,
    smoothed=smoothed,
    stop_levels=long_stop,
    slope_threshold=0.0
)

# 运行回测
config = BacktestConfig(
    initial_capital=10000.0,
    fee_rate=0.001,
    slippage=0.0005
)

result = run_backtest(
    signals=signals,
    open_prices=open_prices,
    high_prices=high,
    low_prices=low,
    close_prices=close,
    timestamps=timestamps,
    config=config
)

# 打印结果
print(f"总收益: {result.metrics['total_return']:.2f}")
print(f"夏普比率: {result.metrics['sharpe_ratio']:.2f}")
print(f"胜率: {result.metrics['win_rate']:.2%}")
print(f"最大回撤: {result.metrics['max_drawdown_pct']:.2%}")
print(f"交易次数: {result.metrics['n_trades']}")
```

## 🤝 贡献

欢迎贡献！以下是开始的方法：

1. **Fork 仓库**
2. **创建功能分支**：`git checkout -b feature/your-feature`
3. **按现有代码风格进行更改**
4. **运行测试**：`poetry run pytest`
5. **运行检查**：`poetry run ruff check .`
6. **运行类型检查**：`poetry run mypy .`
7. **提交更改**：`git commit -m "添加功能"`
8. **推送到分支**：`git push origin feature/your-feature`
9. **打开 Pull Request**

### 代码风格指南

- 遵循 PEP 8，行长度限制为 100 个字符
- 为所有函数签名使用类型提示
- 为公共函数编写文档字符串
- 为新功能添加测试
- 保持函数专注和模块化
- 使用描述性变量名

### 提交消息格式

```
<类型>: <主题>

<正文>

<页脚>
```

类型：`feat`、`fix`、`docs`、`style`、`refactor`、`test`、`chore`

示例：
```
feat: 添加布林带信号生成器

- 实现标准差带
- 与现有信号管道集成
- 包含边缘情况测试

Closes #123
```

## 📄 许可证

MIT 许可证

Copyright (c) 2024 Binance Fourier Backtester Contributors

特此免费授予任何获得本软件副本和相关文档文件（"软件"）的人不受限制地处理软件的权利，包括但不限于使用、复制、修改、合并、发布、分发、再许可和/或出售软件副本的权利，以及允许获得软件的人这样做，但须符合以下条件：

上述版权声明和本许可声明应包含在软件的所有副本或主要部分中。

软件按"原样"提供，不提供任何形式的明示或暗示保证，包括但不限于对适销性、特定用途适用性和非侵权的保证。在任何情况下，作者或版权持有人均不对任何索赔、损害或其他责任负责，无论是在合同诉讼、侵权行为还是其他方面，由软件或软件的使用或其他交易引起、由此产生或与之相关。

## 🙏 致谢

- **币安** 提供免费的市场数据 API
- **Streamlit** 提供优秀的 UI 框架
- **SciPy** 提供强大的 FFT 和信号处理工具
- **Poetry** 提供现代 Python 依赖管理

## 📞 支持与资源

- **文档**：查看 [docs/](docs/) 文件夹
- **问题**：通过 GitHub Issues 报告错误或请求功能
- **讨论**：使用 GitHub Discussions 提问
- **示例**：查看 [examples/](examples/) 文件夹获取完整工作流程

## 🗺️ 路线图

**已完成：**
- ✅ M1: 数据层和缓存
- ✅ M2: 傅里叶分析和平滑
- ✅ M3: 基本回测引擎
- ✅ M4: 可视化和 UI
- ✅ M5: 动态止损和信号
- ✅ M6: 实时流式（可选）
- ✅ M7: 多时间框架和高级功能
- ✅ M8: 参数优化
- ✅ M9: 投资组合管理

**未来增强：**
- [ ] 机器学习集成（从傅里叶进行特征工程）
- [ ] 期权策略（跨式、价差）
- [ ] 通过币安 API 自动交易执行
- [ ] Telegram/Discord 通知
- [ ] 自定义指标库
- [ ] 多交易所支持（FTX、Coinbase）
- [ ] 实时风险监控仪表板

---

**祝交易愉快！🚀📈**

有关详细的技术文档，请参阅：
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - 系统设计和算法
- [CONFIGURATION.md](docs/CONFIGURATION.md) - 完整参数参考
- [FAQ.md](docs/FAQ.md) - 常见问题
