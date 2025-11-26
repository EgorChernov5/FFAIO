from sklearn.metrics import accuracy_score

from src.models import LinearClassificator
from src.losses import PerceptronLoss
from src.optimizers import GDOptimizer
from src.data_loaders import ShuffleLoader, datasets


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = datasets.load_mushroom_dataset()

    model = LinearClassificator(in_features=X_train.shape[1])
    loss = PerceptronLoss()

    regularizer = None
    lr = 0.001
    batch_size = 32
    data_loader = ShuffleLoader(batch_size)
    optimizer = GDOptimizer(model.get_layers(), data_loader, lr)

    n_epochs = 100
    count_metric = accuracy_score
    verbose_n_batch_multiple = 10
    model.train_model(
        n_epochs,
        X_train, y_train, X_val, y_val,
        loss, optimizer, regularizer,
        count_metric=count_metric,
        verbose_n_batch_multiple=verbose_n_batch_multiple, verbose_statistic='EMA'
    )
