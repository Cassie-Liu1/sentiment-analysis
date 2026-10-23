"""Train/evaluate the shared sentiment model on prepared Amazon CSV splits."""

import sys

from run_imdb import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--train-file", "amazon_train.csv", "--eval-file", "amazon_test.csv"]
    main()
