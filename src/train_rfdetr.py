from rfdetr import RFDETRNano


model = RFDETRNano()


model.train(
    dataset_dir=
    r"C:\Users\22577\Desktop\pollen_project\data\processed\pollen_detection",

    epochs=50,

    batch_size=2,

    grad_accum_steps=4,

    lr=1e-4,

    output_dir=
    r"C:\Users\22577\Desktop\pollen_project\checkpoints"
)