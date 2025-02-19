import html
import sys
import os

if __name__ == '__main__':

    src_path = sys.argv[1]
    out_path = sys.argv[2]
    with open(src_path) as src_f, open(out_path, "w") as out_f:
    	for line in src_f:
    		len1 = len(line.split())
    		line = html.unescape(line)
    		len2 = len(line.split())
    		if len1 != len2:
    			print(line)
    			sys.exit(0)
    		out_f.write(line)