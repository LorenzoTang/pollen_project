import os
import multiprocessing

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
multiprocessing.set_start_method('spawn', force=True) 

if __name__ == '__main__':

    multiprocessing.freeze_support()
    
    from rfdetr import RFDETRSegMedium
    modelName = "RFDETRSegMedium"
    model = RFDETRSegMedium()
    
    model.train(
        dataset_dir=r"D:\Development\RF-DETR\__Data\训练数据\dataset",  # 修改为修改为数据集地址
        epochs=50,
        batch_size=2,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=fr"{modelName}output",   # 可指定权重文件输出地址
        num_workers=0,
        log_every_n_steps=1,
    )

