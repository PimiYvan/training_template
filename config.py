import argparse

args = argparse.Namespace(
    lr=1e-4,
    bs=8,
    train_size=0.8,
    # epoch=30,
    path="./data/Images",
    metadata="./data/metadata_ok.csv"
)

batch_size= 16
train_size= 0.8
num_epochs= 30
