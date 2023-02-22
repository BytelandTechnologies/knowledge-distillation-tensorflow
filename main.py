from modules.dataset import Dataset
from modules.trainer import Trainer

dataset = Dataset()
trainer = Trainer(dataset.ds_num_classes)

history = trainer.train_model(dataset.train_ds, dataset.val_ds)
trainer.save_model("output_model")
trainer.evaluate_model(dataset.test_ds)
print("Foi!")
