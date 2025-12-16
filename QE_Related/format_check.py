
import sys

def check_data(file_path, output_path):

    with open(file_path) as f1, open(output_path) as f2:
        for line1, line2 in zip(f1, f2):
            if len(line1.split()) != len(line2.split()):
                print(line1)
                break


if __name__ == '__main__':

    src_path = sys.argv[1]
    out_path = sys.argv[2]

    check_data(src_path, out_path)


    print("all finished")