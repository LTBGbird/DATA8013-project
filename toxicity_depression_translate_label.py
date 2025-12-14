import os
import time
import argparse
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import threading

import requests
import pandas as pd
from dotenv import load_dotenv


DEEPSEEK_API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


def _format_seconds(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    if seconds < 0:
        seconds = 0
    seconds_int = int(seconds)
    h = seconds_int // 3600
    m = (seconds_int % 3600) // 60
    s = seconds_int % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        api_base: str = DEEPSEEK_API_BASE,
        model: str = DEEPSEEK_MODEL,
        max_retries: int = 1,
        timeout: int = 40,
        sleep_between_requests: float = 0.2,
        max_rps: Optional[float] = None,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.sleep_between_requests = sleep_between_requests
        # Simple global rate limiting (across threads) with min interval between requests
        self.max_rps = max_rps
        self._min_interval = 1.0 / max_rps if max_rps and max_rps > 0 else None
        self._rate_lock = threading.Lock()
        self._last_request_time = 0.0

    def _apply_rate_limit(self) -> None:
        """
        Apply a basic global rate limit across threads.
        If max_rps is set, enforce at most max_rps requests per second
        by waiting between requests.
        """
        if self._min_interval is None:
            return
        with self._rate_lock:
            now = time.time()
            wait = self._min_interval - (now - self._last_request_time)
            if wait > 0:
                time.sleep(wait)
                now = time.time()
            self._last_request_time = now

    def _chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call DeepSeek chat-completions API (OpenAI-compatible) and return assistant content as string.
        """
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "stream": False,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                # Apply global rate limit before each request
                self._apply_rate_limit()
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    time.sleep(self.sleep_between_requests)
                    return content.strip()
                else:
                    # Backoff on non-200 responses (e.g. rate limit)
                    time.sleep(self.sleep_between_requests * attempt)
            except requests.RequestException:
                time.sleep(self.sleep_between_requests * attempt)

        raise RuntimeError("DeepSeek API request failed after max retries.")

    def translate_to_english(self, text: str) -> str:
        system_prompt = (
            "You are a translation assistant. Translate the given text into English. The input may be in English, German, French, Italian, Portuguese, Thai, Hindi, Spanish languages. Respond ONLY with the English translation, without quotes or any explanation."
        )
        user_prompt = f"Text:\n{text}"
        result = self._chat_completion(system_prompt, user_prompt)
        return result.strip()

    def translate_to_chinese(self, text: str) -> str:
        system_prompt = (
            "你是一个翻译助手，请把输入文本翻译成中文。输入可能是英语、德语、法语、意大利语、葡萄牙语、泰语、印地语、西班牙语或其他语言。只输出中文翻译内容，不要带引号或任何解释。"
        )
        user_prompt = f"原文：\n{text}"
        result = self._chat_completion(system_prompt, user_prompt)
        return result.strip()

    def depression_probability(self, text: str) -> float:
        system_prompt = (
            "You are a classifier that estimates the probability that a text is toxic. The text may be any language. Return ONLY a single number between 0 and 1 (inclusive) representing the probability that the text is toxic, with up to 2 decimal places."
        )
        user_prompt = f"The text to be estimated is: {text}"
        result = self._chat_completion(system_prompt, user_prompt)
        token = result.strip().split()[0]
        try:
            prob = float(token)
        except ValueError:
            prob = 0.0
        if prob < 0.0:
            prob = 0.0
        if prob > 1.0:
            prob = 1.0
        return prob


def process_csv_translate_and_label(
    input_csv: str,
    output_path: str,
    api_key: str,
    batch_size: int = 64,
    num_workers: int = 4,
    max_req_per_sec: Optional[float] = None,
) -> None:
    """
    For a small CSV (like selected_v2.csv) that has columns:
        text, label, language, toxicity_prob_en
    (either with a header or as 4 unnamed columns),
    use DeepSeek to:
      - translate text to English and Chinese (new columns: text_en, text_zh)
      - run toxicity classification on EN and ZH versions
        (new columns: depression_label_en, depression_label_zh)
    """
    print(f"Loading existing CSV: {input_csv}")
    try:
        df = pd.read_csv(input_csv)
        if len(df.columns) == 4 and set(df.columns) != {"text", "label", "language", "toxicity_prob_en"}:
            # Probably no header
            df = pd.read_csv(
                input_csv,
                header=None,
                names=["text", "label", "language", "toxicity_prob_en"],
            )
    except Exception:
        # Fallback robust read
        df = pd.read_csv(
            input_csv,
            header=None,
            names=["text", "label", "language", "toxicity_prob_en"],
        )

    if "text" not in df.columns:
        raise RuntimeError("Input CSV must have a 'text' column (or be 4 columns without header).")

    # Keep a copy of the original dataframe so we can later write skipped rows
    # with all original columns preserved.
    df_original = df.copy()

    texts = df["text"].tolist()
    total_len = len(texts)
    print(f"Total rows to process: {total_len}")

    client = DeepSeekClient(api_key=api_key, max_rps=max_req_per_sec)

    text_en_all: list[str] = []
    text_zh_all: list[str] = []
    prob_en_all: list[float] = []
    prob_zh_all: list[float] = []
    # Keep track of successfully processed row indices (0-based) so we can
    # drop failed rows entirely from the main output, and also track skipped
    # rows in a separate CSV.
    success_indices: list[int] = []
    skipped_indices: list[int] = []

    start_time = time.time()
    current = 0

    while current < total_len:
        end = min(current + batch_size, total_len)
        batch_texts = texts[current:end]

        batch_en: list[str] = []
        batch_zh: list[str] = []
        batch_prob_en: list[float] = []
        batch_prob_zh: list[float] = []

        def _process_row(text: str):
            """
            Process a single row: translate + classify.
            If any DeepSeek call fails (including timeout), return (False, exc)
            so that the caller can skip this row entirely.
            """
            try:
                en = client.translate_to_english(text)
                zh = client.translate_to_chinese(text)
                prob_en = client.depression_probability(en)
                prob_zh = client.depression_probability(zh)
                return True, (en, zh, prob_en, prob_zh)
            except Exception as exc:  # noqa: BLE001 - we want to skip on any failure
                return False, exc

        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
            futures = [executor.submit(_process_row, t) for t in batch_texts]

            for local_offset, future in enumerate(futures):
                # global index (0-based) in the original dataframe
                global_zero_idx = current + local_offset
                global_display_idx = global_zero_idx + 1  # for logging (1-based)

                try:
                    ok, payload = future.result()
                except Exception as exc:  # pragma: no cover - very rare
                    ok, payload = False, exc

                if not ok:
                    # Skip this row entirely on failure (e.g. timeout, sensitive content, etc.)
                    print(
                        f"\n[SKIP] Row {global_display_idx}/{total_len} "
                        f"skipped due to DeepSeek error: {payload}"
                    )
                    skipped_indices.append(global_zero_idx)
                    # Even if skipped, count it as "processed" for ETA purposes.
                    processed = global_display_idx
                else:
                    en, zh, prob_en, prob_zh = payload
                    batch_en.append(en)
                    batch_zh.append(zh)
                    batch_prob_en.append(prob_en)
                    batch_prob_zh.append(prob_zh)
                    success_indices.append(global_zero_idx)
                    processed = global_display_idx

                elapsed = time.time() - start_time
                if processed > 0 and elapsed > 0:
                    est_total_time = elapsed * total_len / processed
                    eta_seconds = est_total_time - elapsed
                    eta_str = _format_seconds(eta_seconds)
                    speed = processed / elapsed
                    status = (
                        f"[{processed}/{total_len}] Translating & classifying... "
                        f"{speed:.2f} samples/s, ETA ~ {eta_str}"
                    )
                else:
                    status = f"[{processed}/{total_len}] Translating & classifying..."
                print(status, end="\r")

        text_en_all.extend(batch_en)
        text_zh_all.extend(batch_zh)
        prob_en_all.extend(batch_prob_en)
        prob_zh_all.extend(batch_prob_zh)

        print(f"\nFinished batch {current}-{end - 1}")
        current = end

    # If some rows failed (e.g. sensitive content or repeated timeouts),
    # drop those rows entirely from the main output so that all new columns
    # align in length with the dataframe. Skipped rows will be written to a
    # separate CSV that preserves the original columns.
    if success_indices:
        df_ok = df_original.iloc[success_indices].reset_index(drop=True)
    else:
        # If nothing succeeded, just create an empty dataframe with the same columns.
        df_ok = df_original.iloc[0:0].copy()

    # Attach new columns for successfully processed rows
    df_ok["text_en"] = text_en_all
    df_ok["text_zh"] = text_zh_all
    df_ok["toxicity_prob_en"] = prob_en_all
    df_ok["toxicity_prob_zh"] = prob_zh_all

    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".csv"
    skipped_output_path = base + "_skipped" + ext

    # Save main processed output
    if ext.lower() == ".csv":
        df_ok.to_csv(output_path, index=False)
    elif ext.lower() == ".parquet":
        df_ok.to_parquet(output_path, index=False)
    else:
        df_ok.to_csv(output_path, index=False)

    # Save skipped rows (with original columns only) into a separate CSV if any
    if skipped_indices:
        df_skipped = df_original.iloc[skipped_indices].reset_index(drop=True)
        if ext.lower() == ".csv":
            df_skipped.to_csv(skipped_output_path, index=False)
        elif ext.lower() == ".parquet":
            df_skipped.to_parquet(skipped_output_path, index=False)
        else:
            df_skipped.to_csv(skipped_output_path, index=False)
        skipped_msg = f" and skipped rows to: {skipped_output_path} (rows: {len(df_skipped)})"
    else:
        skipped_msg = " and no rows were skipped."

    print(
        f"Saved CSV with new columns 'text_en', 'text_zh', "
        f"'toxicity_prob_en', 'toxicity_prob_zh' to: {output_path} "
        f"(total processed rows: {len(df_ok)}){skipped_msg}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate selected toxicity samples into English and Chinese, and "
            "predict toxicity labels on the translated texts using DeepSeek."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to input CSV (e.g., selected_v2.csv) with columns: text,label,language,toxicity_prob_en.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="selected_v2_with_translations_and_labels.csv",
        help="Output CSV path with added translation and toxicity label columns.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of examples per batch (default: 100).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=50,
        help="Number of parallel worker threads for DeepSeek API calls (default: 50).",
    )
    parser.add_argument(
        "--max-req-per-sec",
        type=float,
        default=120.0,
        help="Approximate global max DeepSeek API requests per second across all threads (default: 120.0).",
    )
    return parser.parse_args()


def main() -> None:
    # Load .env if present, so user can put DEEPSEEK_API_KEY there
    load_dotenv()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing DEEPSEEK_API_KEY. Please set it in your environment, for example:\n"
            "  export DEEPSEEK_API_KEY='your_api_key_here'"
        )

    args = parse_args()
    process_csv_translate_and_label(
        input_csv=args.input_csv,
        output_path=args.output,
        api_key=api_key,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_req_per_sec=args.max_req_per_sec,
    )


if __name__ == "__main__":
    main()


