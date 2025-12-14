# en prompt, original text
cd "/Users/bird/HKU课程/DATA8013"
source .venv/bin/activate

python toxicity_depression_inference.py \
  --input-csv toxicity_8_lang.csv \
  --output toxicity_8_lang_toxicity.csv \
  --batch-size 1000 \
  --num-workers 50 \
  --max-req-per-sec 120 \
  --toxicity-only

  

# corresponding prompt for each language
cd "/Users/bird/HKU课程/DATA8013" && source .venv/bin/activate && export DEEPSEEK_API_KEY="sk-4635a19de4ef48ee8a7995a68ca1f7da" && python toxicity_depression_inference.py --input-csv toxicity_8_lang.csv --output toxicity_8_lang_toxicity_multi.csv --batch-size 1000 --num-workers 50 --max-req-per-sec 150