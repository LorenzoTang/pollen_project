from rfdetr import RFDETRNano

model = RFDETRNano()

model.train(
    dataset_dir="./mini_dataset",
    epochs=3,
    batch_size=1,
    grad_accum_steps=4
)