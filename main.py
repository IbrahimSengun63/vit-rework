from data import CreateLMDB
from data import LMDBImageNetDataLoader
import warnings
from model import VIT
from train import Train, train_load_checkpoint, Evaluate

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    create_lmdb = CreateLMDB()
    create_lmdb.create_train()
    create_lmdb.create_val()

    data_loader = LMDBImageNetDataLoader()

    # Create train and val loaders
    train_loader = data_loader.create_train_loader()
    val_loader = data_loader.create_val_loader()

    model = VIT()

    trainer = Train(model, train_loader, val_loader)
    trainer.start_training()

    # train_load_checkpoint(trainer, "models/vit_latest.pth", device=trainer.device)
    # trainer.start_training()

    evaluator = Evaluate(trainer.get_trained_model(), val_loader)
    evaluator.get_result()


if __name__ == "__main__":
    main()
