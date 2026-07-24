import os
import multiprocessing

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
multiprocessing.set_start_method('spawn', force=True) 

if __name__ == '__main__':

    multiprocessing.freeze_support()
    
    from rfdetr import RFDETRMedium
    
    model = RFDETRMedium()
    
    model.train(
        dataset_dir=r"D:\Development\__Data\训练数据\Coco_dataset",
        epochs=50,
        batch_size=2,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=r"output",
        num_workers=0,
        log_every_n_steps=1,
    )

