from cgi import test
from datasets import load_dataset
#load dataset
def load_cifar10(train_size, test_size, val_prct):    
    train_ds, test_ds = load_dataset('cifar10', split=[f"train[:{train_size}]", f"test[:{test_size}]"])
    splits = train_ds.train_test_split(test_size=float(val_prct))
    train_ds = splits['train']
    val_ds = splits['test']
    return train_ds, val_ds, test_ds
