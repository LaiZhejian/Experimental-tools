import sys
import re

def process_log_file(file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if "INFO | test" in line:
                    # Split the line by '|'
                    fields = line.split('|')

                    # Extract relevant fields
                    spearman_match = re.search(r'spearman\s+([\d.]+)', line)
                    pearson_match = re.search(r'pearson\s+([\d.]+)', line)
                    mcc_match = re.search(r'mt_tag_mcc\s+([\d.]+)', line)
                    f1_ok_match = re.search(r'mt_tag_f1_ok\s+([\d.]+)', line)
                    f1_bad_match = re.search(r'mt_tag_f1_bad\s+([\d.]+)', line)
                    f1_mult_math = re.search(r'mt_tag_f1_mult\s+([\d.]+)', line)

                    if spearman_match and pearson_match and mcc_match and f1_ok_match and f1_bad_match and f1_mult_math:
                        spearman = float(spearman_match.group(1))
                        pearson = float(pearson_match.group(1))
                        mcc = float(mcc_match.group(1))
                        f1_ok = float(f1_ok_match.group(1))
                        f1_bad = float(f1_bad_match.group(1))
                        f1_mult = float(f1_mult_math.group(1))

                        # Print the output
                        print("Spearman\tPearson\tMCC\tF1-Mult=F1_OK`*F1BAD")
                        print(f"{spearman}\t{pearson}\t{mcc}\t{f1_mult:.2f}={f1_ok:.2f}*{f1_bad:.2f}")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
    else:
        process_log_file(sys.argv[1])
