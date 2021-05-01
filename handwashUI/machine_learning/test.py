import sys
import os
import time

def main(argv):
    time.sleep(5)
    print(f"This the file you pass in: {argv}")

if __name__ == "__main__":
    main(sys.argv)