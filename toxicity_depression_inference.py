import os
import time
import argparse
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import threading

import requests
import pandas as pd
from datasets import load_dataset
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
        max_retries: int = 3,
        timeout: int = 30,
        sleep_between_requests: float = 0.5,
        max_rps: Optional[float] = None,
        enable_thinking: bool = False,
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
        # Whether to enable DeepSeek's server-side "deep thinking" mode
        self.enable_thinking = enable_thinking

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
        # Enable DeepSeek's deep thinking mode at API level if requested
        if self.enable_thinking:
            payload["thinking"] = {"type": "enabled"}

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

    def detect_language(self, text: str) -> str:
        system_prompt = (
            "You are a language identification assistant. Given a short text, you must detect the type of the language and respond with a short language code. Use ISO 639-1 codes where possible, such as: en, de, fr, es, it, pt, hi, th, zh, etc. You should carefully detect the language of the text itself, completely unaffected by the content of the language. Respond ONLY with the language code, no extra words. "
        )
        user_prompt = f"The text to be detected for language type is: {text}"
        result = self._chat_completion(system_prompt, user_prompt)
        # Just in case the model returns extra tokens, take first token-like piece
        return result.split()[0]

    def toxicity_probability(self, text: str) -> float:
        """
        Generic (English-instructed) toxicity probability classifier.
        """
        system_prompt = (
            "You are a classifier that estimates the probability that a text is toxic. The text may be any language. Return ONLY a single number between 0 and 1 (inclusive) representing the probability that the text is toxic, with up to 2 decimal places."
        )
        user_prompt = f"The text to be estimated is: {text}"
        return self._prob_from_prompts(system_prompt, user_prompt)

    def _prob_from_prompts(self, system_prompt: str, user_prompt: str) -> float:
        """
        Helper to run a chat completion and parse a 0-1 probability from the response.
        """
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

    def toxicity_probability_multilingual(self, text: str, lang: str) -> float:
        """
        Language-specific toxicity probability, using prompts written in the target language.
        Currently supports: en, de, fr, it, pt, th, hi, es. Falls back to English prompt otherwise.
        """
        lang = (lang or "").lower()

        if lang == "en":
            system_prompt = (
                "You are a classifier that estimates the probability that a text is toxic. The text may be any language. Return ONLY a single number between 0 and 1 (inclusive) representing the probability that the text is toxic, with up to 2 decimal places."
            )
            user_prompt = f"The text to be estimated is: {text}"
        elif lang == "de":
            system_prompt = (
                "Sie sind ein Klassifikator, der die Wahrscheinlichkeit schätzt, dass ein Text toxisch ist. Der Text kann in jeder Sprache verfasst sein. Geben Sie NUR eine einzige Zahl zwischen 0 und 1 (einschließlich) zurück, die die Wahrscheinlichkeit darstellt, dass der Text toxisch ist, mit bis zu 2 Dezimalstellen."
            )
            user_prompt = f"Der zu bewertende Text ist: {text}"
        elif lang == "fr":
            system_prompt = (
                "Vous êtes un classificateur qui estime la probabilité qu'un texte soit toxique. Le texte peut être dans n'importe quelle langue. Retournez SEULEMENT un seul nombre entre 0 et 1 (inclus) représentant la probabilité que le texte soit toxique, avec jusqu'à 2 décimales."
            )
            user_prompt = f"Le texte à évaluer est : {text}"
        elif lang == "it":
            system_prompt = (
                "Sei un classificatore che stima la probabilità che un testo sia tossico. Il testo può essere in qualsiasi lingua. Restituisci SOLO un singolo numero tra 0 e 1 (inclusi) che rappresenta la probabilità che il testo sia tossico, con fino a 2 decimali."
            )
            user_prompt = f"Il testo da valutare è: {text}"
        elif lang == "pt":
            system_prompt = (
                "Você é um classificador que estima a probabilidade de um texto ser tóxico. O texto pode estar em qualquer idioma. Retorne APENAS um único número entre 0 e 1 (inclusive) representando a probabilidade de o texto ser tóxico, com até 2 casas decimais."
            )
            user_prompt = f"O texto a ser avaliado é: {text}"
        elif lang == "th":
            system_prompt = (
                "คุณเป็นตัวจำแนกที่ประเมินความน่าจะเป็นที่ข้อความจะเป็นพิษ ข้อความอาจเป็นภาษาใดก็ได้ ให้คืนค่าเพียงตัวเลขเดียวระหว่าง 0 และ 1 (รวม) ที่แสดงถึงความน่าจะเป็นที่ข้อความจะเป็นพิษ โดยมีทศนิยมไม่เกิน 2 ตำแหน่ง"
            )
            user_prompt = f"ข้อความที่จะประเมินคือ: {text}"
        elif lang == "hi":
            system_prompt = (
                "आप एक वर्गीकारक हैं जो अनुमान लगाता है कि कोई पाठ विषाक्त होने की क्या संभावना है। पाठ किसी भी भाषा में हो सकता है। केवल 0 और 1 (समावेशी) के बीच एक संख्या लौटाएं जो इस बात की संभावना दर्शाती है कि पाठ विषाक्त है, 2 दशमलव स्थानों तक।"
            )
            user_prompt = f"मूल्यांकन किया जाने वाला पाठ है: {text}"
        elif lang == "es":
            system_prompt = (
                "Eres un clasificador que estima la probabilidad de que un texto sea tóxico. El texto puede estar en cualquier idioma. Devuelve SOLO un único número entre 0 y 1 (inclusive) que represente la probabilidad de que el texto sea tóxico, con hasta 2 decimales."
            )
            user_prompt = f"El texto a evaluar es: {text}"
        else:
            # Fallback to the generic English-instructed classifier
            print(f"[Warning]Falling back to English-instructed classifier for language: {lang}")
            system_prompt = (
                "You are a classifier that estimates the probability that a text is toxic or harmful.\n"
                "The text may be any language. "
                "Return ONLY a single number between 0 and 1 (inclusive), with up to 3 decimal places, "
                "representing the probability that the text is toxic."
            )
            user_prompt = f"The text to be estimated is: {text}"

        return self._prob_from_prompts(system_prompt, user_prompt)

    def toxicity_probability_ur_context(self, text: str) -> float:
        """
        Urdu-specific toxicity probability using an ENGLISH prompt that explicitly
        describes the typical background of Urdu speakers, as provided by the user.
        This version is used together with DeepSeek's deep-thinking mode enabled.
        """
        system_prompt = (
            "You are a classifier that estimates the probability that a text is toxic or harmful.\n"
            "The text is in Urdu. Urdu speakers often have the following background:\n"
            "Values: Their core faith is Islam, and they care deeply about and sympathize with the fate "
            "of Muslim communities in places such as Palestine and Kashmir.\n"
            "Traditions: It is very common to quote the Quran, Hadith, or religious figures in history "
            "(such as Imam Husayn) to support their views. They frequently use metaphors, symbols, and "
            "historical allusions to strengthen their expressions.\n"
            "Socio-cultural context: Solidarity of the Muslim community is highly valued. They are very "
            "sensitive to any form of oppression experienced by Muslims. Media and social networks are "
            "full of reports and discussions about geopolitical conflicts.\n"
            "First, reason deeply and carefully about whether the text is toxic in this context, "
            "but in your final answer you must output ONLY the probability number, without any of your reasoning.\n"
            "Return ONLY a single number between 0 and 1 (inclusive), with up to 3 decimal places, "
            "representing the probability that the text is toxic."
        )
        user_prompt = f"Text:\n{text}"
        return self._prob_from_prompts(system_prompt, user_prompt)

    def toxicity_probability_ur_ctx_A(self, text: str) -> float:
        """
        Urdu-specific toxicity probability, background A only:
        "Socio-cultural context: Solidarity of the Muslim community is highly valued."
        (no DeepSeek deep-thinking parameter; plain classification prompt)
        """
        system_prompt = (
            "You are a classifier that estimates the probability that a text is toxic or harmful.\n"
            "The text is in Urdu. Urdu speakers often have the following background:\n"
            "Socio-cultural context: Solidarity of the Muslim community is highly valued.\n"
            "Return ONLY a single number between 0 and 1 (inclusive), with up to 3 decimal places, "
            "representing the probability that the text is toxic."
        )
        user_prompt = f"Text:\n{text}"
        return self._prob_from_prompts(system_prompt, user_prompt)

    def toxicity_probability_ur_ctx_B(self, text: str) -> float:
        """
        Urdu-specific toxicity probability, background B only:
        "They are very sensitive to any form of oppression experienced by Muslims."
        (no DeepSeek deep-thinking parameter; plain classification prompt)
        """
        system_prompt = (
            "You are a classifier that estimates the probability that a text is toxic or harmful.\n"
            "The text is in Urdu. Urdu speakers often have the following background:\n"
            "They are very sensitive to any form of oppression experienced by Muslims.\n"
            "Return ONLY a single number between 0 and 1 (inclusive), with up to 3 decimal places, "
            "representing the probability that the text is toxic."
        )
        user_prompt = f"Text:\n{text}"
        return self._prob_from_prompts(system_prompt, user_prompt)

    def toxicity_probability_ur_ctx_C(self, text: str) -> float:
        """
        Urdu-specific toxicity probability, background C only:
        "Media and social networks are full of reports and discussions about geopolitical conflicts."
        (no DeepSeek deep-thinking parameter; plain classification prompt)
        """
        system_prompt = (
            "You are a classifier that estimates the probability that a text is toxic or harmful.\n"
            "The text is in Urdu. Urdu speakers often have the following background:\n"
            "Media and social networks are full of reports and discussions about geopolitical conflicts.\n"
            "Return ONLY a single number between 0 and 1 (inclusive), with up to 3 decimal places, "
            "representing the probability that the text is toxic."
        )
        user_prompt = f"Text:\n{text}"
        return self._prob_from_prompts(system_prompt, user_prompt)

    def toxicity_probability_hi_ctx_full(self, text: str) -> float:
        """
        Hindi-specific toxicity probability, using ALL three background paragraphs
        (Values + Traditions + Socio-cultural context) together, as provided by the user.
        """
        system_prompt = (
            "You are a classifier that estimates the probability that a text is toxic or harmful. The text is in Hindi. "
            "Hindi speakers often have the following background:\n\n"
            "Values: While India is incredibly diverse, a significant portion of Hindi speakers identify with Hinduism. "
            "Core values often include Dharma (righteous conduct), Karma, family honor, respect for elders, and "
            "hospitality. National pride and patriotism are also highly valued.\n\n"
            "Traditions: It is very common to quote from Hindu scriptures, ancient texts, epics, or common proverbs. "
            "References to historical figures (e.g., Mahatma Gandhi, Shivaji) or mythological figures (e.g., Rama, "
            "Krishna) are frequent.\n\n"
            "Socio-cultural context: India is a highly diverse and often politically charged society. Discussions "
            "frequently revolve around national identity, religious harmony/discord, caste dynamics, regional issues, "
            "and geopolitical events. There is high sensitivity around religious sentiments, national symbols, and "
            "historical narratives.\n\n"
            "Return ONLY a single number between 0 and 1 (inclusive) representing the probability that the text is "
            "toxic, with up to 2 decimal places."
        )
        user_prompt = f"Text:\n{text}"
        return self._prob_from_prompts(system_prompt, user_prompt)

    def toxicity_probability_hi_ctx_values(self, text: str) -> float:
        """
        Hindi-specific toxicity probability, using ONLY the 'Values' background paragraph.
        """
        system_prompt = (
            "You are a classifier that estimates the probability that a text is toxic or harmful. The text is in Hindi. Hindi speakers often have the following background:\n\n"
            "Values: While India is incredibly diverse, a significant portion of Hindi speakers identify with Hinduism. Core values often include Dharma (righteous conduct), Karma, family honor, respect for elders, and hospitality. National pride and patriotism are also highly valued.\n\n"
            "Return ONLY a single number between 0 and 1 (inclusive) representing the probability that "
            "the text is toxic, with up to 2 decimal places."
        )
        user_prompt = f"Text:\n{text}"
        return self._prob_from_prompts(system_prompt, user_prompt)

    def toxicity_probability_hi_ctx_traditions(self, text: str) -> float:
        """
        Hindi-specific toxicity probability, using ONLY the 'Traditions' background paragraph.
        """
        system_prompt = (
            "You are a classifier that estimates the probability that a text is toxic or harmful. The text is in Hindi. Hindi speakers often have the following background:\n\n"
            "Traditions: It is very common to quote from Hindu scriptures, ancient texts, epics, or common proverbs. References to historical figures (e.g., Mahatma Gandhi, Shivaji) or mythological figures (e.g., Rama, Krishna) are frequent.\n\n"
            "Return ONLY a single number between 0 and 1 (inclusive) representing the probability that the text is toxic, with up to 2 decimal places."
        )
        user_prompt = f"Text:\n{text}"
        return self._prob_from_prompts(system_prompt, user_prompt)

    def toxicity_probability_hi_ctx_socio(self, text: str) -> float:
        """
        Hindi-specific toxicity probability, using ONLY the 'Socio-cultural context' background paragraph.
        """
        system_prompt = (
            "You are a classifier that estimates the probability that a text is toxic or harmful. The text is in Hindi. Hindi speakers often have the following background:\n\n"
            "Socio-cultural context: India is a highly diverse and often politically charged society. Discussions frequently revolve around national identity, religious harmony/discord, caste dynamics, regional issues, and geopolitical events. There is high sensitivity around religious sentiments, national symbols, and historical narratives.\n\n"
            "Return ONLY a single number between 0 and 1 (inclusive) representing the probability that the text is toxic, with up to 2 decimal places."
        )
        user_prompt = f"Text:\n{text}"
        return self._prob_from_prompts(system_prompt, user_prompt)

    def toxicity_probability_th_ctx_full(self, text: str) -> float:
        """
        Thai-specific toxicity probability, using ALL three background paragraphs
        (Values + Traditions + Socio-cultural context) together, as provided by the user.
        """
        system_prompt = (
            "You are a classifier that estimates the probability that a text is toxic or harmful. The text is in Thai. "
            "Thai speakers often have the following background:\n\n"
            "Values: Their core faith is Theravada Buddhism, and they hold deep reverence for the Monarchy. "
            "National identity is often strongly linked to 'Nation, Religion, King.' Respect for elders and "
            "hierarchical social structures is also highly valued.\n\n"
            "Traditions: It is very common to quote Buddhist teachings (Dharma), proverbs, or sayings from respected "
            "monks. References to historical figures from Thai history, particularly royalty, are frequent. "
            "Communication often values indirectness and 'Kreng Jai', a sense of deference and not wanting to impose "
            "or offend, so direct aggression can be perceived as highly toxic.\n\n"
            "Socio-cultural context: There is extreme sensitivity around the Monarchy; any criticism or perceived insult "
            "can be deeply offensive and is subject to strict lèse-majesté laws. Respect for Buddhist institutions and "
            "figures is paramount. There is strong national pride and sensitivity to anything perceived as denigrating "
            "Thailand or its culture.\n\n"
            "Return ONLY a single number between 0 and 1 (inclusive) representing the probability that the text is toxic, "
            "with up to 2 decimal places."
        )
        user_prompt = f"Text:\n{text}"
        return self._prob_from_prompts(system_prompt, user_prompt)

    def toxicity_probability_th_ctx_values(self, text: str) -> float:
        """
        Thai-specific toxicity probability, using ONLY the 'Values' background paragraph.
        """
        system_prompt = (
            "You are a classifier that estimates the probability that a text is toxic or harmful. The text is in Thai. "
            "Thai speakers often have the following background:\n\n"
            "Values: Their core faith is Theravada Buddhism, and they hold deep reverence for the Monarchy. "
            "National identity is often strongly linked to 'Nation, Religion, King.' Respect for elders and "
            "hierarchical social structures is also highly valued.\n\n"
            "Return ONLY a single number between 0 and 1 (inclusive) representing the probability that the text is toxic, "
            "with up to 2 decimal places."
        )
        user_prompt = f"Text:\n{text}"
        return self._prob_from_prompts(system_prompt, user_prompt)

    def toxicity_probability_th_ctx_traditions(self, text: str) -> float:
        """
        Thai-specific toxicity probability, using ONLY the 'Traditions' background paragraph.
        """
        system_prompt = (
            "You are a classifier that estimates the probability that a text is toxic or harmful. The text is in Thai. "
            "Thai speakers often have the following background:\n\n"
            "Traditions: It is very common to quote Buddhist teachings (Dharma), proverbs, or sayings from respected "
            "monks. References to historical figures from Thai history, particularly royalty, are frequent. "
            "Communication often values indirectness and 'Kreng Jai', a sense of deference and not wanting to impose or "
            "offend, so direct aggression can be perceived as highly toxic.\n\n"
            "Return ONLY a single number between 0 and 1 (inclusive) representing the probability that the text is toxic, "
            "with up to 2 decimal places."
        )
        user_prompt = f"Text:\n{text}"
        return self._prob_from_prompts(system_prompt, user_prompt)

    def toxicity_probability_th_ctx_socio(self, text: str) -> float:
        """
        Thai-specific toxicity probability, using ONLY the 'Socio-cultural context' background paragraph.
        """
        system_prompt = (
            "You are a classifier that estimates the probability that a text is toxic or harmful. The text is in Thai. "
            "Thai speakers often have the following background:\n\n"
            "Socio-cultural context: There is extreme sensitivity around the Monarchy; any criticism or perceived insult "
            "can be deeply offensive and is subject to strict lèse-majesté laws. Respect for Buddhist institutions and "
            "figures is paramount. There is strong national pride and sensitivity to anything perceived as denigrating "
            "Thailand or its culture.\n\n"
            "Return ONLY a single number between 0 and 1 (inclusive) representing the probability that the text is toxic, "
            "with up to 2 decimal places."
        )
        user_prompt = f"Text:\n{text}"
        return self._prob_from_prompts(system_prompt, user_prompt)


def process_dataset(
    split: str,
    output_path: str,
    api_key: str,
    max_rows: Optional[int] = None,
    batch_size: int = 512,
    resume: bool = False,
    num_workers: int = 1,
    max_req_per_sec: Optional[float] = None,
) -> None:
    """
    Load the multilingual toxicity dataset, enrich with language and depression probability,
    and save to a file with two extra columns.

    Supports batch-wise incremental saving and simple resume for CSV outputs.
    """
    print(f"Loading dataset split='{split}' from HuggingFace...")
    ds = load_dataset("malexandersalazar/toxicity-multilingual-binary-classification-dataset", split=split)

    total_len = len(ds)
    if max_rows is not None:
        total_len = min(max_rows, total_len)

    ext = os.path.splitext(output_path)[1].lower()
    if resume and ext != ".csv":
        raise RuntimeError("Resume is only supported for CSV outputs. Please use a .csv output file.")

    start_idx = 0
    if resume and os.path.exists(output_path):
        # Determine how many rows have already been processed
        existing_df = pd.read_csv(output_path)
        processed_n = len(existing_df)
        if processed_n >= total_len:
            print(f"All {processed_n} rows already processed, nothing to do.")
            return
        start_idx = processed_n
        print(f"Resuming from index {start_idx} (already processed {processed_n}/{total_len} rows).")
    else:
        # If not resuming and file exists, overwrite it
        if os.path.exists(output_path):
            print(f"Output file '{output_path}' already exists and resume=False, overwriting it.")
            os.remove(output_path)

    client = DeepSeekClient(api_key=api_key, max_rps=max_req_per_sec)

    texts = ds["text"]
    labels = ds["label"]

    # Process in batches and append to CSV
    current = start_idx
    start_time = time.time()
    while current < total_len:
        end = min(current + batch_size, total_len)
        batch_texts = texts[current:end]
        batch_labels = labels[current:end]

        batch_languages = []
        batch_depression_probs = []

        def _process_text(text: str):
            lang = client.detect_language(text)
            prob = client.toxicity_probability(text)
            return lang, prob

        # Parallelize API calls within the batch using threads (I/O-bound)
        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
            futures = [executor.submit(_process_text, t) for t in batch_texts]

            for local_offset, future in enumerate(futures):
                lang, prob = future.result()

                global_idx = current + 1 + local_offset
                batch_languages.append(lang)
                batch_depression_probs.append(prob)

                # Per-sample progress + ETA (like deep learning training logs)
                processed_since_start = global_idx - start_idx
                elapsed = time.time() - start_time
                if processed_since_start > 0 and elapsed > 0:
                    est_total_time = elapsed * (total_len - start_idx) / processed_since_start
                    eta_seconds = est_total_time - elapsed
                    eta_str = _format_seconds(eta_seconds)
                    speed = processed_since_start / elapsed
                    status = (
                        f"[{global_idx}/{total_len}] Processing example... "
                        f"{speed:.2f} samples/s, ETA ~ {eta_str}"
                    )
                else:
                    status = f"[{global_idx}/{total_len}] Processing example..."
                print(status, end="\r")

        # Build batch DataFrame
        df_batch = pd.DataFrame(
            {
                "text": batch_texts,
                "label": batch_labels,
                "language": batch_languages,
                "depression_prob": batch_depression_probs,
            }
        )

        # Decide output format by file extension
        if ext in [".csv"]:
            # Append mode with header only if file does not exist
            header = not os.path.exists(output_path)
            df_batch.to_csv(output_path, mode="a", header=header, index=False)
        elif ext in [".parquet"]:
            # For parquet, we currently do not support incremental append;
            # instead, accumulate all results would be required. For large
            # runs, prefer CSV if you need resume/checkpointing.
            if current == start_idx:
                df_batch.to_parquet(output_path, index=False)
            else:
                # Load existing, append, and overwrite (less efficient but keeps API simple)
                existing = pd.read_parquet(output_path)
                combined = pd.concat([existing, df_batch], ignore_index=True)
                combined.to_parquet(output_path, index=False)
        else:
            # Default to CSV if unknown extension
            header = not os.path.exists(output_path)
            df_batch.to_csv(output_path, mode="a", header=header, index=False)

        print(f"\nSaved batch {current}-{end - 1} to: {output_path}")
        current = end

    print(f"Finished all {total_len} rows. Output saved to: {output_path}")


def sample_languages_hf_dataset(
    split: str,
    output_path: str,
    api_key: str,
    batch_size: int = 256,
    num_workers: int = 4,
    max_req_per_sec: Optional[float] = None,
) -> None:
    """
    Detect languages on the original HuggingFace dataset until we have at least
    10 language types and EACH has at least 100 examples. Results are saved
    incrementally every 1000 samples.

    Output columns:
        - text
        - label
        - language
    """
    print(f"[LANG] Loading HF dataset split='{split}' for language detection...")
    ds = load_dataset(
        "malexandersalazar/toxicity-multilingual-binary-classification-dataset",
        split=split,
    )

    texts = ds["text"]
    labels = ds["label"]
    total_len = len(texts)
    print(f"[LANG] Total available rows in split '{split}': {total_len}")

    # If output already exists, overwrite to avoid mixing old and new runs.
    # We also create a separate file for skipped rows.
    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".csv"
    ok_path = base + ext
    skip_path = base + "_skipped" + ext
    for p in (ok_path, skip_path):
        if os.path.exists(p):
            print(f"[LANG] Output file '{p}' exists, overwriting it.")
            os.remove(p)

    client = DeepSeekClient(api_key=api_key, max_rps=max_req_per_sec)

    from collections import Counter

    lang_counts: Counter[str] = Counter()
    ok_buffer: list[dict[str, Any]] = []
    skip_buffer: list[dict[str, Any]] = []
    processed = 0
    last_save_at = 0
    last_print_time = time.time()
    has_written_ok = False
    has_written_skip = False

    def flush_ok() -> None:
        nonlocal has_written_ok
        if not ok_buffer:
            return
        df = pd.DataFrame(ok_buffer)
        if ext.lower() == ".parquet":
            if not has_written_ok:
                df.to_parquet(ok_path, index=False)
            else:
                existing = pd.read_parquet(ok_path)
                combined = pd.concat([existing, df], ignore_index=True)
                combined.to_parquet(ok_path, index=False)
        else:
            df.to_csv(ok_path, mode="a", header=not has_written_ok, index=False)
        has_written_ok = True
        print(
            f"[LANG] Saved {len(ok_buffer)} processed rows to '{ok_path}' "
            f"(total processed so far: {processed})"
        )
        ok_buffer.clear()

    def flush_skip() -> None:
        nonlocal has_written_skip
        if not skip_buffer:
            return
        df = pd.DataFrame(skip_buffer)
        if ext.lower() == ".parquet":
            if not has_written_skip:
                df.to_parquet(skip_path, index=False)
            else:
                existing = pd.read_parquet(skip_path)
                combined = pd.concat([existing, df], ignore_index=True)
                combined.to_parquet(skip_path, index=False)
        else:
            df.to_csv(skip_path, mode="a", header=not has_written_skip, index=False)
        has_written_skip = True
        print(
            f"[LANG] Saved {len(skip_buffer)} skipped rows to '{skip_path}' "
            f"(total processed so far: {processed})"
        )
        skip_buffer.clear()

    def print_stats(force: bool = False) -> None:
        nonlocal last_print_time
        now = time.time()
        if not force and (now - last_print_time) < 10.0:
            return
        if lang_counts:
            num_langs = len(lang_counts)
            min_count = min(lang_counts.values())
            counts_str = ", ".join(f"{k}:{v}" for k, v in sorted(lang_counts.items()))
            print(
                f"\n[LANG STATS] types={num_langs}, min_count={min_count}, "
                f"counts={{ {counts_str} }}\n"
            )
        else:
            print("\n[LANG STATS] No languages detected yet.\n")
        last_print_time = now

    stop = False
    current = 0
    start_time = time.time()
    while current < total_len and not stop:
        end = min(current + batch_size, total_len)
        batch_texts = texts[current:end]
        batch_labels = labels[current:end]

        def _detect(text: str) -> str:
            return client.detect_language(text)

        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
            futures = [executor.submit(_detect, t) for t in batch_texts]

            for local_idx, (t, lbl, fut) in enumerate(
                zip(batch_texts, batch_labels, futures, strict=False)
            ):
                try:
                    # If a single language-detection request takes more than 40 seconds
                    # (including any internal retries inside DeepSeekClient), treat it
                    # as failed and put this row into the "skipped" file.
                    lang = fut.result(timeout=40.0)
                    lang_counts[lang] += 1
                    ok_buffer.append(
                        {
                            "text": t,
                            "label": lbl,
                            "language": lang,
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    print(
                        f"\n[LANG SKIP] row {processed + 1}/{total_len} skipped due to "
                        f"error or timeout: {e}"
                    )
                    skip_buffer.append(
                        {
                            "text": t,
                            "label": lbl,
                        }
                    )
                processed += 1

                # Every 1000 samples, flush to disk
                if processed - last_save_at >= 1000:
                    flush_ok()
                    flush_skip()
                    last_save_at = processed

                # Periodic stats
                print_stats(force=False)

                # Stopping condition: at least 10 languages and each has >= 100 samples
                if len(lang_counts) >= 10 and min(lang_counts.values()) >= 100:
                    stop = True
                    break

        current = end

    # Final flush and stats
    flush_ok()
    flush_skip()
    print_stats(force=True)
    elapsed = time.time() - start_time
    print(
        f"[LANG] Finished language sampling. Processed {processed} examples in "
        f"{_format_seconds(elapsed)}.\n"
        f"[LANG] Processed rows saved to: {ok_path}\n"
        f"[LANG] Skipped rows (errors/timeouts) saved to: {skip_path}"
    )


def process_existing_csv_th_context_abc(
    input_csv: str,
    output_path: str,
    api_key: str,
    batch_size: int = 512,
    num_workers: int = 1,
    max_req_per_sec: Optional[float] = None,
) -> None:
    """
    Re-run toxicity classification ONLY for rows where language == 'th' in an
    existing CSV, using FOUR different Thai-context English prompts:

        - Full background (Values + Traditions + Socio-cultural context)
        - Values-only background
        - Traditions-only background
        - Socio-cultural context-only background

    For each Thai row we compute four probabilities:
        - toxicity_prob_th_ctx_full
        - toxicity_prob_th_ctx_values
        - toxicity_prob_th_ctx_traditions
        - toxicity_prob_th_ctx_socio

    Only Thai rows are written to the outputs. Rows that fail (error or taking
    more than 30 seconds for all four calls together) are saved into a separate
    *_skipped file, preserving their original columns.
    """
    print(f"Loading existing CSV (TH-only context Full/Values/Traditions/Socio re-run): {input_csv}")
    df = pd.read_csv(input_csv)

    if "text" not in df.columns or "language" not in df.columns:
        raise RuntimeError("Input CSV must contain 'text' and 'language' columns.")

    # Identify Thai rows
    lang_series = df["language"].astype(str).str.lower()
    th_mask = lang_series == "th"
    th_indices = df.index[th_mask].tolist()
    total_len = len(th_indices)
    print(f"Total TH rows to re-process (context Full/Values/Traditions/Socio): {total_len}")

    if total_len == 0:
        print("No rows with language == 'th' found. Nothing to do.")
        return

    # Work only on the Thai subset for output
    th_df = df.loc[th_mask].copy()
    th_texts = th_df["text"].tolist()

    # For this TH-only re-run we EXPLICITLY DISABLE deep-thinking mode
    client = DeepSeekClient(api_key=api_key, max_rps=max_req_per_sec, enable_thinking=False)

    new_probs_full: list[float] = []
    new_probs_values: list[float] = []
    new_probs_traditions: list[float] = []
    new_probs_socio: list[float] = []

    start_time = time.time()
    current = 0
    while current < total_len:
        end = min(current + batch_size, total_len)
        batch_texts = th_texts[current:end]

        batch_probs_full: list[float] = []
        batch_probs_values: list[float] = []
        batch_probs_traditions: list[float] = []
        batch_probs_socio: list[float] = []

        def _process_th_text_all(text: str) -> tuple[float, float, float, float]:
            """
            Run four separate TH context prompts (Full / Values / Traditions / Socio) for the same text.
            """
            p_full = client.toxicity_probability_th_ctx_full(text)
            p_values = client.toxicity_probability_th_ctx_values(text)
            p_traditions = client.toxicity_probability_th_ctx_traditions(text)
            p_socio = client.toxicity_probability_th_ctx_socio(text)
            return p_full, p_values, p_traditions, p_socio

        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
            futures = [executor.submit(_process_th_text_all, t) for t in batch_texts]

            for local_offset, future in enumerate(futures):
                try:
                    # If a single sample takes more than 30 seconds (including any internal retries),
                    # treat it as failed and leave its Full/Values/Traditions/Socio probabilities empty (NaN).
                    p_full, p_values, p_traditions, p_socio = future.result(timeout=30.0)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"\n[SKIP TH] TH row {current + 1 + local_offset}/{total_len} "
                        f"skipped in Full/Values/Traditions/Socio due to error or timeout: {e}"
                    )
                    p_full = float("nan")
                    p_values = float("nan")
                    p_traditions = float("nan")
                    p_socio = float("nan")

                batch_probs_full.append(p_full)
                batch_probs_values.append(p_values)
                batch_probs_traditions.append(p_traditions)
                batch_probs_socio.append(p_socio)

                processed = current + 1 + local_offset
                elapsed = time.time() - start_time
                if processed > 0 and elapsed > 0:
                    est_total_time = elapsed * total_len / processed
                    eta_seconds = est_total_time - elapsed
                    eta_str = _format_seconds(eta_seconds)
                    speed = processed / elapsed
                    status = (
                        f"[{processed}/{total_len}] Re-processing TH example (Full/Values/Traditions/Socio)... "
                        f"{speed:.2f} samples/s, ETA ~ {eta_str}"
                    )
                else:
                    status = (
                        f"[{processed}/{total_len}] Re-processing TH example (Full/Values/Traditions/Socio)..."
                    )
                print(status, end="\r")

        new_probs_full.extend(batch_probs_full)
        new_probs_values.extend(batch_probs_values)
        new_probs_traditions.extend(batch_probs_traditions)
        new_probs_socio.extend(batch_probs_socio)
        print(f"\nFinished TH batch {current}-{end - 1}")
        current = end

    # Build a DataFrame containing ONLY the Thai rows, with four new columns
    th_df["toxicity_prob_th_ctx_full"] = new_probs_full
    th_df["toxicity_prob_th_ctx_values"] = new_probs_values
    th_df["toxicity_prob_th_ctx_traditions"] = new_probs_traditions
    th_df["toxicity_prob_th_ctx_socio"] = new_probs_socio

    # Split into successfully processed vs skipped (any NaN in any of the four considered skipped)
    ok_mask = ~(
        th_df["toxicity_prob_th_ctx_full"].isna()
        | th_df["toxicity_prob_th_ctx_values"].isna()
        | th_df["toxicity_prob_th_ctx_traditions"].isna()
        | th_df["toxicity_prob_th_ctx_socio"].isna()
    )
    df_ok = th_df[ok_mask].copy()
    df_skip = th_df[~ok_mask].copy()

    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".csv"
    ok_path = base + ext
    skip_path = base + "_skipped" + ext

    # Save processed TH rows (with all four context-based probabilities) into one CSV
    if ext.lower() == ".parquet":
        df_ok.to_parquet(ok_path, index=False)
        df_skip.to_parquet(skip_path, index=False)
    else:
        df_ok.to_csv(ok_path, index=False)
        df_skip.to_csv(skip_path, index=False)

    print(
        "Saved TH-only context-based results to separate files:\n"
        f"  OK rows: {ok_path} (rows: {len(df_ok)})\n"
        f"  Skipped rows (timeout/error): {skip_path} (rows: {len(df_skip)})"
    )

def process_existing_csv_hi_context_abc(
    input_csv: str,
    output_path: str,
    api_key: str,
    batch_size: int = 512,
    num_workers: int = 1,
    max_req_per_sec: Optional[float] = None,
) -> None:
    """
    Re-run toxicity classification ONLY for rows where language == 'hi' in an
    existing CSV, using FOUR different Hindi-context English prompts:

        - Full background (Values + Traditions + Socio-cultural context)
        - Values-only background
        - Traditions-only background
        - Socio-cultural context-only background

    For each Hindi row we compute four probabilities:
        - toxicity_prob_hi_ctx_full
        - toxicity_prob_hi_ctx_values
        - toxicity_prob_hi_ctx_traditions
        - toxicity_prob_hi_ctx_socio

    Only Hindi rows are written to the outputs. Rows that fail (error or taking
    more than 30 seconds for all three calls together) are saved into a separate
    *_skipped file, preserving their original columns.
    """
    print(f"Loading existing CSV (HI-only context A/B/C re-run): {input_csv}")
    df = pd.read_csv(input_csv)

    if "text" not in df.columns or "language" not in df.columns:
        raise RuntimeError("Input CSV must contain 'text' and 'language' columns.")

    # Identify Hindi rows
    lang_series = df["language"].astype(str).str.lower()
    hi_mask = lang_series == "hi"
    hi_indices = df.index[hi_mask].tolist()
    total_len = len(hi_indices)
    print(f"Total HI rows to re-process (context Values/Traditions/Socio): {total_len}")

    if total_len == 0:
        print("No rows with language == 'hi' found. Nothing to do.")
        return

    # Work only on the Hindi subset for output
    hi_df = df.loc[hi_mask].copy()
    hi_texts = hi_df["text"].tolist()

    # For this HI-only re-run we EXPLICITLY DISABLE deep-thinking mode
    client = DeepSeekClient(api_key=api_key, max_rps=max_req_per_sec, enable_thinking=False)

    new_probs_full: list[float] = []
    new_probs_values: list[float] = []
    new_probs_traditions: list[float] = []
    new_probs_socio: list[float] = []

    start_time = time.time()
    current = 0
    while current < total_len:
        end = min(current + batch_size, total_len)
        batch_texts = hi_texts[current:end]

        batch_probs_full: list[float] = []
        batch_probs_values: list[float] = []
        batch_probs_traditions: list[float] = []
        batch_probs_socio: list[float] = []

        def _process_hi_text_all(text: str) -> tuple[float, float, float, float]:
            """
            Run four separate HI context prompts (Full / Values / Traditions / Socio) for the same text.
            """
            p_full = client.toxicity_probability_hi_ctx_full(text)
            p_values = client.toxicity_probability_hi_ctx_values(text)
            p_traditions = client.toxicity_probability_hi_ctx_traditions(text)
            p_socio = client.toxicity_probability_hi_ctx_socio(text)
            return p_full, p_values, p_traditions, p_socio

        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
            futures = [executor.submit(_process_hi_text_all, t) for t in batch_texts]

            for local_offset, future in enumerate(futures):
                try:
                    # If a single sample takes more than 30 seconds (including any internal retries),
                    # treat it as failed and leave its Full/Values/Traditions/Socio probabilities empty (NaN).
                    p_full, p_values, p_traditions, p_socio = future.result(timeout=30.0)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"\n[SKIP HI] HI row {current + 1 + local_offset}/{total_len} "
                        f"skipped in Full/Values/Traditions/Socio due to error or timeout: {e}"
                    )
                    p_full = float("nan")
                    p_values = float("nan")
                    p_traditions = float("nan")
                    p_socio = float("nan")

                batch_probs_full.append(p_full)
                batch_probs_values.append(p_values)
                batch_probs_traditions.append(p_traditions)
                batch_probs_socio.append(p_socio)

                processed = current + 1 + local_offset
                elapsed = time.time() - start_time
                if processed > 0 and elapsed > 0:
                    est_total_time = elapsed * total_len / processed
                    eta_seconds = est_total_time - elapsed
                    eta_str = _format_seconds(eta_seconds)
                    speed = processed / elapsed
                    status = (
                        f"[{processed}/{total_len}] Re-processing HI example (Full/Values/Traditions/Socio)... "
                        f"{speed:.2f} samples/s, ETA ~ {eta_str}"
                    )
                else:
                    status = (
                        f"[{processed}/{total_len}] Re-processing HI example (Full/Values/Traditions/Socio)..."
                    )
                print(status, end="\r")

        new_probs_full.extend(batch_probs_full)
        new_probs_values.extend(batch_probs_values)
        new_probs_traditions.extend(batch_probs_traditions)
        new_probs_socio.extend(batch_probs_socio)
        print(f"\nFinished HI batch {current}-{end - 1}")
        current = end

    # Build a DataFrame containing ONLY the Hindi rows, with four new columns
    hi_df["toxicity_prob_hi_ctx_full"] = new_probs_full
    hi_df["toxicity_prob_hi_ctx_values"] = new_probs_values
    hi_df["toxicity_prob_hi_ctx_traditions"] = new_probs_traditions
    hi_df["toxicity_prob_hi_ctx_socio"] = new_probs_socio

    # Split into successfully processed vs skipped (any NaN in any of the four considered skipped)
    ok_mask = ~(
        hi_df["toxicity_prob_hi_ctx_full"].isna()
        | hi_df["toxicity_prob_hi_ctx_values"].isna()
        | hi_df["toxicity_prob_hi_ctx_traditions"].isna()
        | hi_df["toxicity_prob_hi_ctx_socio"].isna()
    )
    df_ok = hi_df[ok_mask].copy()
    df_skip = hi_df[~ok_mask].copy()

    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".csv"
    ok_path = base + ext
    skip_path = base + "_skipped" + ext

    # Save processed HI rows (with all three context-based probabilities) into one CSV
    if ext.lower() == ".parquet":
        df_ok.to_parquet(ok_path, index=False)
        df_skip.to_parquet(skip_path, index=False)
    else:
        df_ok.to_csv(ok_path, index=False)
        df_skip.to_csv(skip_path, index=False)

    print(
        "Saved HI-only context-based results to separate files:\n"
        f"  OK rows: {ok_path} (rows: {len(df_ok)})\n"
        f"  Skipped rows (timeout/error): {skip_path} (rows: {len(df_skip)})"
    )

def process_toxicity_only_csv(
    input_csv: str,
    output_path: str,
    api_key: str,
    batch_size: int = 512,
    num_workers: int = 1,
    max_req_per_sec: Optional[float] = None,
) -> None:
    """
    Run ONLY the generic toxicity classifier (using `toxicity_probability`, which
    now represents toxicity probability) on an existing CSV.

    Expected columns (minimum):
        - text

    If 'label' and 'language' are present they are kept, but not required.

    New column added:
        - toxicity_prob_en
    """
    print(f"Loading input CSV for toxicity-only inference: {input_csv}")
    df = pd.read_csv(input_csv)

    if "text" not in df.columns:
        raise RuntimeError("Input CSV must contain a 'text' column.")

    texts = df["text"].astype(str).tolist()
    total_len = len(texts)
    print(f"Total rows to process (toxicity-only): {total_len}")

    client = DeepSeekClient(api_key=api_key, max_rps=max_req_per_sec)

    new_probs: list[float] = []
    start_time = time.time()

    current = 0
    while current < total_len:
        end = min(current + batch_size, total_len)
        batch_texts = texts[current:end]

        batch_probs: list[float] = []

        def _process_text(text: str) -> float:
            return client.toxicity_probability(text)

        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
            futures = [executor.submit(_process_text, t) for t in batch_texts]

            for local_offset, future in enumerate(futures):
                prob = future.result()
                batch_probs.append(prob)

                global_idx = current + 1 + local_offset
                processed = global_idx
                elapsed = time.time() - start_time
                if processed > 0 and elapsed > 0:
                    est_total_time = elapsed * total_len / processed
                    eta_seconds = est_total_time - elapsed
                    eta_str = _format_seconds(eta_seconds)
                    speed = processed / elapsed
                    status = (
                        f"[{global_idx}/{total_len}] Processing example (toxicity-only)... "
                        f"{speed:.2f} samples/s, ETA ~ {eta_str}"
                    )
                else:
                    status = f"[{global_idx}/{total_len}] Processing example (toxicity-only)..."
                print(status, end="\r")

        new_probs.extend(batch_probs)
        print(f"\nFinished batch {current}-{end - 1}")
        current = end

    df["toxicity_prob_en"] = new_probs

    ext = os.path.splitext(output_path)[1].lower()
    if ext in [".csv"]:
        df.to_csv(output_path, index=False)
    elif ext in [".parquet"]:
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(
        f"Saved CSV with new column 'toxicity_prob_en' (generic toxicity probability) to: {output_path} "
        f"(total rows: {len(df)})"
    )

def process_existing_csv(
    input_csv: str,
    output_path: str,
    api_key: str,
    batch_size: int = 512,
    num_workers: int = 1,
    max_req_per_sec: Optional[float] = None,
) -> None:
    """
    Re-run the multilingual toxicity probability classifier on an existing CSV that already
    has text and previous annotations. The expected columns are:
        text, label, language, toxicity_prob_en, text_en, text_zh, toxicity_prob_zh
    If these column names are not present, the CSV is assumed to have no header
    and 4 unnamed columns in that order.

    A new column 'toxicity_prob_lang' is appended as the last column.
    """
    print(f"Loading existing CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    # text,label,language,toxicity_prob_en,text_en,text_zh,toxicity_prob_zh
    expected_cols = {"text", "label", "language", "toxicity_prob_en", "text_en", "text_zh", "toxicity_prob_zh"}
    if not expected_cols.issubset(set(df.columns)):
        # Assume no header and exactly 4 columns in the expected order
        df = pd.read_csv(
            input_csv,
            header=None,
            names=["text", "label", "language", "toxicity_prob_en", "text_en", "text_zh", "toxicity_prob_zh"],
        )

    client = DeepSeekClient(api_key=api_key, max_rps=max_req_per_sec)

    # Drop completely empty rows (e.g. blank lines) to avoid NaNs
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    texts = df["text"].tolist()
    langs = df["language"].tolist()

    total_len = len(texts)
    print(f"Total rows to re-process: {total_len}")

    # Initialize the new column with NaN values
    df["toxicity_prob_lang"] = float('nan')

    start_time = time.time()

    current = 0
    while current < total_len:
        end = min(current + batch_size, total_len)
        batch_texts = texts[current:end]
        batch_langs = langs[current:end]

        batch_probs_lang: list[float] = []

        def _process_text_multilingual(text: str, lang: str) -> float:
            prob_lang = client.toxicity_probability_multilingual(text, lang)
            return prob_lang

        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
            futures = [
                executor.submit(_process_text_multilingual, t, l)
                for t, l in zip(batch_texts, batch_langs, strict=False)
            ]

            for local_offset, future in enumerate(futures):
                prob_lang = future.result()
                batch_probs_lang.append(prob_lang)

                global_idx = current + 1 + local_offset
                processed = global_idx
                elapsed = time.time() - start_time
                if processed > 0 and elapsed > 0:
                    est_total_time = elapsed * total_len / processed
                    eta_seconds = est_total_time - elapsed
                    eta_str = _format_seconds(eta_seconds)
                    speed = processed / elapsed
                    status = (
                        f"[{global_idx}/{total_len}] Processing example (multilingual)... "
                        f"{speed:.2f} samples/s, ETA ~ {eta_str}"
                    )
                else:
                    status = f"[{global_idx}/{total_len}] Processing example (multilingual)..."
                print(status, end="\r")

        # Update the dataframe with batch results
        df.loc[current:end-1, "toxicity_prob_lang"] = batch_probs_lang
        
        # Save after each batch
        ext = os.path.splitext(output_path)[1].lower()
        if ext in [".csv"]:
            df.to_csv(output_path, index=False)
        elif ext in [".parquet"]:
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        
        print(f"\nFinished batch {current}-{end - 1}, saved to {output_path}")
        current = end

    print(
        f"Completed processing. Final CSV with column 'toxicity_prob_lang' saved to: {output_path} "
        f"(total rows: {len(df)})"
    )







def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use DeepSeek to annotate the multilingual toxicity dataset with language and depression probability.\n\n"
            "Dataset: malexandersalazar/toxicity-multilingual-binary-classification-dataset\n"
            "See: https://huggingface.co/datasets/malexandersalazar/toxicity-multilingual-binary-classification-dataset"
        )
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split of the dataset to process (default: train).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="toxicity_with_depression.csv",
        help="Output file path (.csv or .parquet). Default: toxicity_with_depression.csv",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of rows to process (useful for testing). If omitted, process all rows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Number of examples per batch before saving to disk (default: 512).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel worker threads per batch for DeepSeek API calls (default: 1).",
    )
    parser.add_argument(
        "--max-req-per-sec",
        type=float,
        default=None,
        help="Approximate global max DeepSeek API requests per second across all threads (default: unlimited).",
    )
    parser.add_argument(
        "--lang-sample",
        action="store_true",
        help=(
            "When set (and without --input-csv), run language detection on the HF dataset "
            "and stop once at least 10 language types are found and each has at least 100 examples. "
            "Saves 'text,label,language' and flushes to disk every 1000 rows."
        ),
    )
    parser.add_argument(
        "--toxicity-only",
        action="store_true",
        help=(
            "When used with --input-csv, run ONLY the generic toxicity classifier on the CSV "
            "and append a 'toxicity_prob_en' column. The CSV must contain at least a 'text' column."
        ),
    )
    parser.add_argument(
        "--hi-context-abc",
        action="store_true",
        help=(
            "When used with --input-csv, only recompute toxicity for rows with language == 'hi' "
            "using four different Hindi-context English prompts (Full / Values / Traditions / Socio), "
            "adding columns 'toxicity_prob_hi_ctx_full', 'toxicity_prob_hi_ctx_values', "
            "'toxicity_prob_hi_ctx_traditions', and 'toxicity_prob_hi_ctx_socio', and splitting successful vs. timed-out rows "
            "into separate output files."
        ),
    )
    parser.add_argument(
        "--th-context-abc",
        action="store_true",
        help=(
            "When used with --input-csv, only recompute toxicity for rows with language == 'th' "
            "using four different Thai-context English prompts (Full / Values / Traditions / Socio), "
            "adding columns 'toxicity_prob_th_ctx_full', 'toxicity_prob_th_ctx_values', "
            "'toxicity_prob_th_ctx_traditions', and 'toxicity_prob_th_ctx_socio', and splitting successful vs. timed-out rows "
            "into separate output files."
        ),
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help=(
            "Optional path to an existing CSV file to re-run depression probability on. "
            "If set, the script will ignore the HF dataset and operate on this CSV instead, "
            "adding new toxicity probability columns."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing CSV output file, continuing from the last processed row.",
    )
    parser.add_argument(
        "--ur-context-only",
        action="store_true",
        help=(
            "When used with --input-csv, only recompute toxicity for rows with language == 'ur' "
            "using a context-aware English prompt, adding 'toxicity_prob_ur_ctx' column."
        ),
    )
    parser.add_argument(
        "--ur-context-abc",
        action="store_true",
        help=(
            "When used with --input-csv, only recompute toxicity for rows with language == 'ur' "
            "using three different Urdu-context English prompts (A/B/C), adding columns "
            "'toxicity_prob_ur_ctx_A', 'toxicity_prob_ur_ctx_B', 'toxicity_prob_ur_ctx_C'."
        ),
    )
    parser.add_argument(
        "--ur-context-abc-deep",
        action="store_true",
        help=(
            "When used with --input-csv, only recompute toxicity for rows with language == 'ur' "
            "using three different Urdu-context English prompts (A/B/C) with DeepSeek deep-thinking "
            "ENABLED, adding columns 'toxicity_prob_ur_ctx_A_deep', 'toxicity_prob_ur_ctx_B_deep', "
            "'toxicity_prob_ur_ctx_C_deep'."
        ),
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

    if args.input_csv:
        # Re-run toxicity probability on an existing CSV file
        if args.th_context_abc:
            process_existing_csv_th_context_abc(
                input_csv=args.input_csv,
                output_path=args.output,
                api_key=api_key,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_req_per_sec=args.max_req_per_sec,
            )
        elif args.hi_context_abc:
            process_existing_csv_hi_context_abc(
                input_csv=args.input_csv,
                output_path=args.output,
                api_key=api_key,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_req_per_sec=args.max_req_per_sec,
            )
        elif args.toxicity_only:
            process_toxicity_only_csv(
                input_csv=args.input_csv,
                output_path=args.output,
                api_key=api_key,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_req_per_sec=args.max_req_per_sec,
            )
        else:
            process_existing_csv(
                input_csv=args.input_csv,
                output_path=args.output,
                api_key=api_key,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_req_per_sec=args.max_req_per_sec,
            )
    else:
        # Default path: process HuggingFace dataset
        if args.lang_sample:
            sample_languages_hf_dataset(
                split=args.split,
                output_path=args.output,
                api_key=api_key,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_req_per_sec=args.max_req_per_sec,
            )
        else:
            process_dataset(
                split=args.split,
                output_path=args.output,
                api_key=api_key,
                max_rows=args.max_rows,
                batch_size=args.batch_size,
                resume=args.resume,
                num_workers=args.num_workers,
                max_req_per_sec=args.max_req_per_sec,
            )


if __name__ == "__main__":
    main()


