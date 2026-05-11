"""
Unified preprocessing entry point for all FOMO26 downstream tasks.

Usage:
  python run_preprocessing.py --task 1 --raw_dir /data/raw/task1 --out_dir /data/preprocessed/task1
  python run_preprocessing.py --task 2 --raw_dir /data/raw/task2 --out_dir /data/preprocessed/task2
  python run_preprocessing.py --task 3 --raw_dir /data/raw/task3 --out_dir /data/preprocessed/task3
  python run_preprocessing.py --task 4 --raw_dir /data/raw/task4 --out_dir /data/preprocessed/task4
  python run_preprocessing.py --task 5 --raw_dir /data/raw/task5 --out_dir /data/preprocessed/task5
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",    type=int, required=True, choices=[1,2,3,4,5])
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_jobs",  type=int, default=8)
    args = parser.parse_args()

    if args.task == 1:
        from data.preprocess.fomo1 import preprocess_task1  # noqa: PLC0415
        preprocess_task1(args.raw_dir, args.out_dir, n_jobs=args.n_jobs)
    elif args.task == 2:
        from data.preprocess.fomo2 import preprocess_task2  # noqa: PLC0415
        preprocess_task2(args.raw_dir, args.out_dir, n_jobs=args.n_jobs)
    elif args.task == 3:
        from data.preprocess.fomo3 import preprocess_task3  # noqa: PLC0415
        preprocess_task3(args.raw_dir, args.out_dir, n_jobs=args.n_jobs)
    elif args.task == 4:
        from data.preprocess.fomo4 import preprocess_task4  # noqa: PLC0415
        preprocess_task4(args.raw_dir, args.out_dir, n_jobs=args.n_jobs)
    elif args.task == 5:
        from data.preprocess.fomo5 import preprocess_task5  # noqa: PLC0415
        preprocess_task5(args.raw_dir, args.out_dir, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
