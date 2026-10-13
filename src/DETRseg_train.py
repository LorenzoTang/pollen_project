import os
import multiprocessing

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
multiprocessing.set_start_method('spawn', force=True) 

if __name__ == '__main__':

    multiprocessing.freeze_support()
    
    from rfdetr import RFDETRSegMedium
    modelName = "RFDETRSegMedium"
    model = RFDETRSegMedium(
        num_classes=12,
        pretrain_weights=r"D:\Development\pollen_project_dev\Pretrain_para\rf-detr-seg-m-ft.pth"
    )
    
    model.train(
        dataset_dir=r"D:\Development\pollen_project_dev\COCO_Dataset_Raw2_Raw3",
        epochs=70,
        batch_size=2,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=fr"{modelName}output-01",
        num_workers=0,

        tensorboard=False,
        eval_interval=1,
        eval_max_dets=500,
        log_per_class_metrics=True,
        best_model_metric="map",
        run_test=False,
        resume=r"D:\Development\pollen_project_dev\RFDETRSegMediumoutput-01\last.ckpt",
    )

