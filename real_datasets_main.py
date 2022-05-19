from core.arguments import get_real_datasets_parser
from core.experiment import Experiment

if __name__ == '__main__':
    parser = get_real_datasets_parser()
    args = parser.parse_args()
    Experiment(args).run()
