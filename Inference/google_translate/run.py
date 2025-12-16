import argparse
from bs4 import BeautifulSoup
import os
import requests
import time
from tqdm import tqdm

def translate_sentence(text, source_lang, target_lang):
    while True:
        try:
            response = requests.get(
                url="http://translate.google.com/m",
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'},
                params={
                    "q": text,
                    "sl": source_lang,
                    "tl": target_lang
                },
                timeout=60)
            if response.status_code == 200:
                return BeautifulSoup(response.text, "html.parser").select_one(".result-container").text
            else:
                # print(f"Error translating text: {text}. Status code: {response.status_code}")
                # print(f"Status code: {response.status_code}")
                raise ValueError("Request failed")
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="Translate text from a file using a RESTful API. For language codes, consult http://cloud.google.com/translate/docs/languages?hl=en")
    parser.add_argument("--sleep", type=float, help="Time to sleep between web requests (Default: 0.5)", default=0.5)
    parser.add_argument("--source_lang", help="Source language, set to 'auto' for Google's Language Detection (Default: auto)", default="auto")
    parser.add_argument("--target_lang", help="Target language")
    parser.add_argument("--input_file", help="Path of input text file, 1 sentence per line")
    parser.add_argument("--output_file", help="Path of output text file")
    parser.add_argument("--max_lines", type=int, help="Translate the first 'max_lines' number of lines", default=int(1e9))
    parser.add_argument("--resume", help="Resume a previously interrupted translation", action="store_true", default=False)
    
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as input_file:
        if args.resume and os.path.exists(args.output_file):
            start_from = len(open(args.output_file, 'r', encoding='utf-8').readlines())
            sentences = input_file.readlines()[start_from:args.max_lines]
            print(f"Skipping {start_from} sentences (already translated) ...")
            with open(args.output_file, 'a', encoding='utf-8') as output_file:
                for sentence in tqdm(sentences, desc="Requesting Google Translate"):
                    translated_sentence = translate_sentence(sentence.strip(), args.source_lang, args.target_lang)
                    if translated_sentence:
                        output_file.write(translated_sentence + '\n')
                        output_file.flush()
                    time.sleep(args.sleep)
        else:
            sentences = input_file.readlines()[:args.max_lines]
            with open(args.output_file, 'w', encoding='utf-8') as output_file:
                for sentence in tqdm(sentences, desc="Requesting Google Translate"):
                    translated_sentence = translate_sentence(sentence.strip(), args.source_lang, args.target_lang)
                    if translated_sentence:
                        output_file.write(translated_sentence + '\n')
                        output_file.flush()
                    time.sleep(args.sleep)

if __name__ == "__main__":
    main()
