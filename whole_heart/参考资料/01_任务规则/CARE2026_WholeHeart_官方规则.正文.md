# CARE2026_WholeHeart_官方规则

> 来源：https://zmic.org.cn/care_2026/track_wholeheart/
> 此文件为正文转换快照；外链及远程图片需要联网，公式和排版以原始资料为准。

# CARE-Whole Heart

Whole Heart Segmentation





## Registration

Registration for CARE 2026 has been closed.

## Motivation

Cardiovascular diseases (CVDs), as the leading cause of death globally, necessitate precise morphological and pathological quantification through segmentation of crucial cardiac structures from medical images. However, whole heart segmentation (WHS) faces challenges including heart shape variability during the cardiac cycle, clinical artifacts like motion and poor contrast-to-noise ratio, as well as domain shifts in multi-center data and the distinct modalities of CT and MRI. CARE-Whole Heart serves to inspire innovative solutions in the realms of biomedical imaging and computer vision, striving to overcome these challenges and advance automated WHS for enhanced understanding and treatment of CVDs.

## Task

![](https://zmic.org.cn/care_2026/assets/img/whs.png)


Figure 1. Overview of CARE-Whole Heart

The objective of this track is to achieve precise segmentation of seven substructures of the whole heart, with robustness against domain shifts (see Fig. 1).  
The specific substructures, each associated with a unique label value, are:

1. **Left Ventricular Blood Cavity (LV)** - Label value: 500
2. **Right Ventricular Blood Cavity (RV)** - Label value: 600
3. **Left Atrial Blood Cavity (LA)** - Label value: 420
4. **Right Atrial Blood Cavity (RA)** - Label value: 550
5. **Myocardium of the Left Ventricle (Myo)** - Label value: 205
6. **Ascending Aorta (AO)** - Label value: 820; defined as the aortic trunk from the aortic valve to the superior level of the atria.
7. **Pulmonary Artery (PA)** - Label value: 850; defined as the initial segment from the pulmonary valve to the bifurcation point.

**Note on Great Vessels:** The great vessels of interest, comprising the ascending aorta and pulmonary artery, are specifically defined due to variations in the fields of view across different scans. This uniform definition is crucial for ensuring consistency across evaluations. During the assessment, segmentation results for these vessels will be truncated to average lengths measured in healthy subjects, although participants are encouraged to extend their segmentation beyond these lengths. Our provided manual segmentations similarly cover more than the defined trunk measurements.

## Data

### About the data

**1) Dataset overview:** CARE-Whole Heart has **246** cases collected from **8** centers, supporting in-distribution and out-of-distribution performance evaluation for methods. Moreover, the diversity and clinical relevance of the dataset offers a broader spectrum of anatomical and pathological variations.

**2) Scanner:** Philips 64-slice CT scanner, SIEMENS SOMATOM Force CT scanner, GE Revolution\_CT scanner, Philips Achieva 1.5T,
Siemens Avanto 1.5T, Toshiba Aquilion ONE CT scanner.

**3) Data format:** The data are all in Nifty format. Each training case has one CT or MRI scan with its corresponding label.

### Training data (106 Cases)

| Center | Num. patients | Modalities |
| --- | --- | --- |
| A | 20 | CT |
| B | 20 | CT |
| C & D | 20 | MRI |
| E | 26 | MRI |
| G | 20 | CT |

```
A ct_train
  |-- Case1001_image.nii.gz
  |-- Case1001_label.nii.gz
  |-- ...
B ct_train
  |-- Case2001_image.nii.gz
  |-- Case2001_label.nii.gz
  |-- ...
C and D mr_train
  |-- Case3001_image.nii.gz
  |-- Case3001_label.nii.gz
  |-- ...
...
```

### Validation data (50 Cases)

| Center | Num. patients | Modalities |
| --- | --- | --- |
| A | 20 | CT |
| B | 10 | CT |
| C & D | 20 | MRI |

```
ct_val
|-- CaseCTVal001_image.nii.gz
|-- CaseCTVal002_image.nii.gz
|-- CaseCTVal003_image.nii.gz
|-- ...
mr_val
|-- CaseMRVal001_image.nii.gz
|-- CaseMRVal002_image.nii.gz
|-- CaseMRVal003_image.nii.gz
|-- ...
```

### Test data (90 Cases)

| Center | Num. patients | Modalities |
| --- | --- | --- |
| A | 20 | CT |
| B | 14 | CT |
| C & D | 20 | MRI |
| F | 16 | MRI |
| H | 20 | CT |

```
ct_test
|-- CaseCTTest001_image.nii.gz
|-- CaseCTTest002_image.nii.gz
|-- CaseCTTest003_image.nii.gz
|-- ...
mr_test
|-- CaseMRTest001_image.nii.gz
|-- CaseMRTest002_image.nii.gz
|-- CaseMRTest003_image.nii.gz
|-- ...
```

**Note on Validation and Test datasets**: We have randomly shuffled the data from different centers and anonymized the center information to promote fairness.

## Rules

* **Only automatic methods are acceptable.** In this track, participants must utilize algorithms that do not require manual intervention or human-assisted processes for the segmentation task.
* **Pre-trained models are allowed.** In this track, the solutions could be developed with open pre-trained fundation models, such as SAM, CLIP and MedSAM.
* **Publicly available data is allowed.** In this track, participants may also include publicly available data. Such additional data needs to be publicly available **before** the date of the test results submission (July 10, 2026). Private annotations of such data is prohibited.

## Metrics

The performance of segmentation results will be assessed through:

* **Dice Similarity Coefficient (DSC)**: DSC is well-suited for assessing the overlap or agreement between the predicted segmentation and the ground truth, making it particularly useful when the regions of interest in medical images may partially overlap.
* **Hausdorff Distance (HD in mm)**: HD measures the maximum distance between two sets, indicating the degree of boundary mismatch.



## Leaderboards

Leaderboards will be released after test results submission.

* We will provide multiple leaderboards, representing stratified performance across different modalities. In each leaderboard, in addition to DSC and HD of 7 cardiac substructures, we will also report DSC and HD of whole heart segmentation (WHS), which are weighted average (according to their volumes) from 7 substructures.

## Citations

**Please cite these papers when you use the data for publications:**

```
@article{Zhuang2016MSMMA,
  Author = {Zhuang, Xiahai and Shen, Juan},
  Title = {Multi-scale patch and multi-modality atlases for whole heart
     segmentation of MRI},
  Journal = {Medical Image Analysis},
  Year = {2016},
  Volume = {31},
  Pages = {77-87},
}

@article{Zhuang2019MvMM,
  Author = {Zhuang, Xiahai},
  Title = {Multivariate Mixture Model for Myocardial Segmentation Combining
     Multi-Source Images},
  Journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  Year = {2019},
  Volume = {41},
  Number = {12},
  Pages = {2933-2946},
}

@article{GAO2023BayeSeg,
  Author = {Gao, Shangqi and Zhou, Hangqi and Gao, Yibo and Zhuang, Xiahai},
  Title = {BayeSeg: Bayesian modeling for medical image segmentation with
     interpretable generalizability},
  Journal = {Medical Image Analysis},
  Year = {2023},
  Volume = {89},
  Pages = {102889},
}
```