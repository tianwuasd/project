import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


CLASS_NAMES = {'LV':'heart_ventricle_left', 'RV':'heart_ventricle_right',
    'LA':'heart_atrium_left', 'RA':'heart_atrium_right', 'Myo':'heart_myocardium',
    'AO':'aorta', 'PA':'pulmonary_artery'}


class TotalSegmentatorAdapter:
    def __init__(self, config, device, boundary):
        self.config, self.device, self.boundary = config, device, boundary
        os.environ['TOTALSEG_HOME_DIR'] = config['backend']['weights_dir']
        os.environ['TOTALSEG_WEIGHTS_PATH'] = str(Path(config['backend']['weights_dir'])/'nnunet/results')
        os.environ['nnUNet_compile'] = 'false'
        from totalsegmentator.config import setup_nnunet, has_valid_license_offline
        setup_nnunet()
        status, _ = has_valid_license_offline()
        if status != 'yes':
            raise PermissionError('Total心腔权重需要有效官方许可证：'+str(status))
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.utilities.file_path_utilities import get_output_folder
        self.predictors = {}
        for task, trainer, step in ((297,'nnUNetTrainer_4000epochs_NoMirroring',.8),(301,'nnUNetTrainer',.5)):
            boundary('model_initialization', task_id=task)
            folder = Path(get_output_folder(task,trainer,'nnUNetPlans','3d_fullres'))
            if not (folder/'fold_0'/'checkpoint_final.pth').is_file():
                raise FileNotFoundError('官方模型尚未下载：'+str(folder))
            predictor = nnUNetPredictor(tile_step_size=step, use_gaussian=True,
                use_mirroring=False, perform_everything_on_device=False,
                device=torch.device(device), verbose=False, verbose_preprocessing=False, allow_tqdm=True)
            predictor.initialize_from_trained_model_folder(str(folder),use_folds=[0],checkpoint_name='checkpoint_final.pth')
            self.predictors[task] = predictor

    def stage(self, image, task, spacing, crop=None):
        from ...observability.events import Stage
        from totalsegmentator.alignment import as_closest_canonical, undo_canonical
        from totalsegmentator.cropping import crop_to_mask, undo_crop
        from totalsegmentator.resampling import change_spacing
        from totalsegmentator.postprocessing import remove_auxiliary_labels
        original = image
        image = nib.Nifti1Image(image.get_fdata(),image.affine)
        if crop is not None:
            image, bbox = crop_to_mask(image,crop,addon=[20,20,20],dtype=np.int32,verbose=False)
        canonical = as_closest_canonical(image)
        sampled = canonical
        if spacing is not None:
            sampled = change_spacing(canonical,[spacing]*3,order=1,dtype=np.int32,
                nr_cpus=1,use_gpu=self.device.startswith('cuda'))
        self.boundary('official_prediction',task_id=task)
        values = np.asanyarray(sampled.dataobj).transpose(2,1,0)[None].astype(np.float32)
        properties = dict(spacing=sampled.header.get_zooms()[::-1])
        with Stage(self.config['run_dir'],'official_predictor',heartbeat_seconds=self.config['logging']['heartbeat_seconds'],task_id=task):
            segmentation = self.predictors[task].predict_single_npy_array(values,properties,
                segmentation_previous_stage=None,output_file_truncated=None,save_or_return_probabilities=False)
        prediction = nib.Nifti1Image(segmentation.transpose(2,1,0).astype(np.uint8),sampled.affine)
        del values, segmentation
        prediction = remove_auxiliary_labels(prediction,'total' if task==297 else 'heartchambers_highres')
        if spacing is not None:
            prediction = change_spacing(prediction,[spacing]*3,canonical.shape,order=0,dtype=np.uint8,
                nr_cpus=1,force_affine=canonical.affine,
                use_gpu=self.device.startswith('cuda'))
        prediction = undo_canonical(prediction,original)
        if crop is not None:
            prediction = undo_crop(prediction,original,bbox)
        return nib.Nifti1Image(np.asarray(prediction.dataobj,dtype=np.uint8),prediction.affine)

    def predict(self, image, labels):
        from totalsegmentator.map_to_binary import class_map
        from totalsegmentator.postprocessing import remove_outside_of_mask
        lowres = self.stage(image,297,3.)
        lowdata = np.asanyarray(lowres.dataobj)
        total = {name:index for index,name in class_map['total'].items()}
        heart = lowdata == total['heart']
        if not heart.any():
            raise RuntimeError('官方297未检出心脏裁剪区域，未生成虚假成功结果')
        outside = np.isin(lowdata,[total[name] for name in ('heart','aorta','inferior_vena_cava')])
        self.boundary('heart_crop_completed')
        prediction = self.stage(image,301,None,nib.Nifti1Image(heart.astype(np.uint8),image.affine))
        dilation = int(10 / np.mean(image.header.get_zooms()))
        values = remove_outside_of_mask(np.asanyarray(prediction.dataobj),outside,addon=dilation)
        mapping = {name:index for index,name in class_map['heartchambers_highres'].items()}
        result = np.zeros(image.shape,dtype=np.uint8)
        covered = []
        for name, label in labels.items():
            if name in CLASS_NAMES:
                result[values==mapping[CLASS_NAMES[name]]] = label
                covered.append(label)
        return result, dict(covered_classes=covered, uncovered_classes=[v for k,v in labels.items() if v and k not in CLASS_NAMES],
            lifecycle='297/301 predictors initialized once per worker',task='heartchambers_highres',
            logits_accumulation='fixed CPU',network_device=self.device,compile=False)
