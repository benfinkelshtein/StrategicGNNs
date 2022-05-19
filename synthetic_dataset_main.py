from utils.basic_classes import DataSet
from core.arguments import get_synthetic_dataset_parser
from core.experiment import Experiment

if __name__ == '__main__':
    parser = get_synthetic_dataset_parser()
    args = parser.parse_args()

    # constant args
    setattr(args, 'dataset_name', DataSet.SYNTHETIC)
    setattr(args, 'num_layers', 1)
    Experiment(args).run()
