## 项目说明：多语种毒性数据 + 抑郁概率标注

本项目使用 Hugging Face 上的多语种毒性二分类数据集 `malexandersalazar/toxicity-multilingual-binary-classification-dataset`，  
并调用 DeepSeek 大模型，对每条文本：

- **第 1 步**：判断文本语言（输出语言代码）
- **第 2 步**：估计文本表达抑郁症状的概率（0–1 之间的实数）

处理后会在原始数据基础上增加两列：

- **`language`**：语言代码（如 `en`、`de`、`fr`、`es`、`it`、`pt`、`hi`、`th` 等）
- **`depression_prob`**：抑郁概率（0–1）

数据集介绍详见 Hugging Face 数据卡片：  
[`malexandersalazar/toxicity-multilingual-binary-classification-dataset`](https://huggingface.co/datasets/malexandersalazar/toxicity-multilingual-binary-classification-dataset)。

---

## 一、创建 conda 虚拟环境（不要在这里运行，只是命令示例）

1. 在终端切换到项目目录：

```bash
cd "/Users/bird/HKU课程/DATA8013"
```

2. 根据 `environment.yml` 创建虚拟环境（**命令供你在本地手动运行**）：

```bash
conda env create -f environment.yml
```

3. 激活虚拟环境：

```bash
conda activate toxicity_env
```

---

## 二、配置 DeepSeek API Key

出于安全原因，脚本不会在代码里硬编码 API key，而是通过环境变量读取：

- 环境变量名称：**`DEEPSEEK_API_KEY`**

你可以在终端中这样设置（请把示例里的 key 换成你自己的真实 key）：

```bash
export DEEPSEEK_API_KEY="你的_deepseek_api_key"
```

如果你更习惯用 `.env` 文件，也可以在项目根目录创建一个 `.env`，内容类似：

```bash
DEEPSEEK_API_KEY=你的_deepseek_api_key
```

脚本会自动通过 `python-dotenv` 加载 `.env`。

（如有需要，你也可以设置可选环境变量 `DEEPSEEK_API_BASE` 和 `DEEPSEEK_MODEL` 来指定 DeepSeek API 的 Base URL 和模型名。）

---

## 三、脚本使用方法

脚本文件：`toxicity_depression_inference.py`

在激活了 `toxicity_env` 环境，并且已经设置好 `DEEPSEEK_API_KEY` 后，可以在项目目录运行：

```bash
python toxicity_depression_inference.py \
  --split train \
  --output toxicity_with_depression.csv \
  --max-rows 1000
```

- **`--split`**：可选 `train` / `val` / `test`，默认 `train`
- **`--output`**：输出文件路径，支持 `.csv` 或 `.parquet`，默认 `toxicity_with_depression.csv`
- **`--max-rows`**：可选参数，用于限制处理的样本数量（调试时建议先设为几百或几千，避免过多 API 请求）

脚本会执行以下流程：

1. 从 Hugging Face 加载指定 split 的数据集（包含 `text` 和 `label` 两列）。
2. 对每条 `text`：
   - 调用 DeepSeek 模型，识别语言（一步 prompt）。
   - 再调用 DeepSeek 模型，估计抑郁概率（第二步 prompt）。
3. 将结果汇总到一个 Pandas DataFrame，并按列顺序保存为：
   - `text`
   - `label`
   - `language`
   - `depression_prob`

---

## 四、重要提醒（伦理与隐私）

- 数据集中包含仇恨言论等有害内容，以及与心理健康相关的敏感文本，使用时请注意伦理与隐私问题。  
- 深度模型对“抑郁概率”的估计**不是**临床诊断，只能用于研究和模型评估，不应用于对个体做真实医疗判断。  
- 请谨慎保存和传播带有敏感内容的推理结果文件，避免泄露和误用。


