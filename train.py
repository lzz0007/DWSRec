import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data import data_preparation, create_samplers, get_dataloader
from recbole.utils import init_seed, init_logger, get_trainer, set_color

from model.dwsrec import DWSRec

from data.dataset import UniSRecDataset

from data.dataloader import CustomizedTrainDataLoader, CustomizedFullSortEvalDataLoader
from trainer import CustomizedTrainer


def finetune(dataset, **kwargs):
    # configurations initialization
    props = ['props/general.yaml', 'props/DWSRec.yaml']
    print(props)

    # configurations initialization
    config = Config(model=DWSRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = UniSRecDataset(config)
    logger.info(dataset)

    # dataset splitting
    built_datasets = dataset.build()
    train_dataset, valid_dataset, test_dataset = built_datasets
    train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)

    train_data = CustomizedTrainDataLoader(config, train_dataset, train_sampler, shuffle=True)
    valid_data = CustomizedFullSortEvalDataLoader(config, valid_dataset, valid_sampler, shuffle=False)
    test_data = CustomizedFullSortEvalDataLoader(config, test_dataset, test_sampler, shuffle=False)

    # model loading and initialization
    model = DWSRec(config, train_data.dataset).to(config['device'])

    logger.info(model)

    # trainer loading and initialization
    trainer = CustomizedTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, test_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Toys', help='dataset name')
    parser.add_argument('--transform', type=str, default=None)
    parser.add_argument('--engine', type=str, default='svd')
    parser.add_argument('--group', type=str, default=4)
    parser.add_argument('--plm_size', type=int, default=768)
    parser.add_argument('--layer_choice', type=str, default='mlp')
    parser.add_argument('--fusion_type', type=str, default='sum')
    args, unparsed = parser.parse_known_args()
    print(args)

    finetune(args.dataset, plm_size=args.plm_size, transform=args.transform, layer_choice=args.layer_choice, engine=args.engine,
             group=args.group, fusion_type=args.fusion_type)

