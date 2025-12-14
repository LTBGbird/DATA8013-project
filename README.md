## Project Description: Multilingual Toxicity Data + Depression Probability Annotations

This project uses the multilingual toxicity binary classification dataset  
`malexandersalazar/toxicity-multilingual-binary-classification-dataset` from Hugging Face,  
and calls the DeepSeek large language model to process each text sample:

- **Step 1**: Detect the language of the text (output language code).
- **Step 2**: Estimate the probability that the text expresses depressive symptoms (a real number between 0 and 1).

After processing, two new columns are added to the original dataset:

- **`language`**: Language code (e.g., `en`, `de`, `fr`, `es`, `it`, `pt`, `hi`, `th`, etc.).
- **`depression_prob`**: Depression probability (0–1).

For details about the dataset, see the Hugging Face dataset card:  
[`malexandersalazar/toxicity-multilingual-binary-classification-dataset`](https://huggingface.co/datasets/malexandersalazar/toxicity-multilingual-binary-classification-dataset)。

---

## 1. Create Conda Virtual Environment (do not run here, commands are examples)

1. In the terminal, change to the project directory:

```bash
cd "/Users/bird/HKU课程/DATA8013"
```

2. Create the virtual environment from `environment.yml` (**run these commands locally on your machine**):

```bash
conda env create -f environment.yml
```

3. Activate the virtual environment:

```bash
conda activate toxicity_env
```

---

## 2. Configure DeepSeek API Key

For security reasons, the scripts do not hard-code the API key in the code.  
Instead, they read it from environment variables:

- Environment variable name: **`DEEPSEEK_API_KEY`**

You can set it in the terminal like this (replace the example key with your real key):

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

If you prefer using a `.env` file, you can create one in the project root directory with contents like:

```bash
DEEPSEEK_API_KEY=your_deepseek_api_key
```

The scripts will automatically load `.env` using `python-dotenv`.

(If needed, you can also set the optional environment variables `DEEPSEEK_API_BASE` and `DEEPSEEK_MODEL`  
to specify the DeepSeek API base URL and model name.)

---

## 3. How to Use the Inference Script

Main script file: `toxicity_depression_inference.py`

After you have activated the `toxicity_env` environment and set `DEEPSEEK_API_KEY`,  
you can run the following command in the project directory:

```bash
python toxicity_depression_inference.py \
  --split train \
  --output toxicity_with_depression.csv \
  --max-rows 1000
```

- **`--split`**: Optional values are `train` / `val` / `test`, default is `train`.
- **`--output`**: Output file path, supports `.csv` or `.parquet`, default is `toxicity_with_depression.csv`.
- **`--max-rows`**: Optional argument to limit the number of samples processed  
  (for debugging, it is recommended to first set this to a few hundred or thousand to avoid too many API requests).

The script will perform the following steps:

1. Load the specified split of the dataset from Hugging Face (containing `text` and `label` columns).
2. For each `text`:
   - Call the DeepSeek model to detect the language (first prompt).
   - Call the DeepSeek model again to estimate the depression probability (second prompt).
3. Aggregate the results into a Pandas DataFrame and save them with the following column order:
   - `text`
   - `label`
   - `language`
   - `depression_prob`

---

## 4. Important Notes (Ethics and Privacy)

- The dataset contains harmful content such as hate speech, as well as sensitive texts related to mental health.  
  Please pay attention to ethical and privacy issues when using it.
- The model’s estimation of “depression probability” is **not** a clinical diagnosis.  
  It should only be used for research and model evaluation, not for making real medical judgments about individuals.
- Be cautious when storing and sharing inference result files that contain sensitive content,  
  to avoid data leakage and misuse.


