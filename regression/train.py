import argparse
from RegressionTrainer import RegressionTrainer


# python regression/train.py -c config.yml -f 0 -d 140


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', "-c", type=str, required=True, help='Path to config.yml')
    parser.add_argument('--fold', "-f", type=int, required=True, help='Fold index')
    parser.add_argument('--dataset_id', "-d", type=int, required=True, help='Dataset ID')

    args = parser.parse_args()

    trainer = RegressionTrainer(config_path=args.config, fold=args.fold, dataset=args.dataset_id)
    trainer.run()


if __name__ == '__main__':
    main()
