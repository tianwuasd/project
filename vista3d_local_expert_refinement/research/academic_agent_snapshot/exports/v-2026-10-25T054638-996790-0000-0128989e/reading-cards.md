# 阅读卡片

本稿由当前文献库中通过证据复核的条目组成，仍需作者审阅与综合。模型复核不等同于人工核验；未发现出版通知不等同于文献可靠。摘要、需 OCR 和出版状态未知的资料在下表中明确标注。


## Draft and Refine with Visual Experts

文档：doc_165c29e833e03cfd92ae53d9；后端：codex

- **研究问题 / supported**：Existing multimodal agents often select visual experts using textual outputs or heuristics and lack quantitative criteria for deciding when visual assistance is needed or which expert to invoke\.
  陈述 `claim_4f724344d0b29ddf874bb9aaebee0387`；证据 ev_68df7c4167d28160176535af
- **方法 / supported**：DnR reformulates a question into concrete visual queries, grounds each query with a CLIP-based model, and averages the resulting spatial relevance maps\.
  陈述 `claim_90e1e97d94e55ccacc5be05113b384df`；证据 ev_734fdbdabec54593802000bb
- **方法 / supported**：The utilization metric combines semantic change after masking relevant regions with prediction stability after masking irrelevant regions; its adaptive weighting uses relevance-map entropy and contrast\.
  陈述 `claim_cb71bac6614831182eb4d98867a8e917`；证据 ev_2ea2ef49b4c9a1fba8368134, ev_79b7ff8e9fe1e3625ab981ce, ev_a67cb53428694000ba66227b
- **方法 / supported**：Expert boxes, masks, or depth maps are rendered onto the original image for re-querying through the existing vision encoder without architectural modification\.
  陈述 `claim_b8f902e54f2a207b12ec4a8e925d0252`；证据 ev_d5f0d3fe3637da5ca349a94d
- **方法 / supported**：DnR recomputes utilization for expert-rendered candidates using the original masks, selects the largest positive improvement over the draft, and retains the draft if no expert improves utilization\.
  陈述 `claim_9152ad6486014a9ca99b717c5a55df13`；证据 ev_2788b02c6c7574d8ff2a68d5
- **方法 / partial**：The experiments use LLaMA-3-70B for query decomposition, CLIPSeg for grounding, and GroundingDINO, SAM, DepthAnything, and mDETR as visual experts; utilization uses 16 stochastic masks per image, with masking ratios 0\.25 for Top-k and 0\.75 for Bottom-k\.
  陈述 `claim_66109cbb9782c15de17d5192103877f9`；证据 ev_f4d28440ede98fb801b24733
- **数据 / supported**：The comprehensive evaluation covers 16 multimodal benchmarks across VQA, image captioning, visual reasoning, knowledge VQA, and comprehensive evaluation, using dataset subsets, GRAY rendering, and Hybrid masking\.
  陈述 `claim_a0c1e55c68f1644628628ce934f89c30`；证据 ev_847c81aa978d961c58765a5c, ev_f4d28440ede98fb801b24733
- **数据 / supported**：Hallucination evaluation uses HaloQuest, MMHal-Bench, VizWiz, and COCO Caption with highlight rendering; ChatGPT 4o-mini categorizes responses as Hallucination, Misperception, Grounded, or Correct\.
  陈述 `claim_eb9a63c9c20422a0290a78539572635f`；证据 ev_02d1c1c97655225621a7117e, ev_ccf2853d91048560958232a8
- **结果 / supported**：Table 1 reports IDEFICS VQAv2 performance increasing from 37\.8 to 47\.85, while its VizWiz performance decreases from 43\.33 to 36\.67\.
  陈述 `claim_fb56af1255aadb175013d7c34b41f663`；证据 ev_21545cf94cf2d20ce8d107b8
- **结果 / supported**：Qwen2\.5-VL captioning performance on Flickr decreases from 57\.3 to 55\.9 after DnR\.
  陈述 `claim_a73b013b2b9c8dc33ac708fbb8edc35c`；证据 ev_2ede6b7ae44d3a8511618cdb
- **结果 / supported**：Table 2 reports Qwen2\.5-VL's hallucination proportion on COCO Caption decreasing from 48\.24% to 26\.13%\.
  陈述 `claim_acbec9bcce55ed1a3d8a65c1fbb41cb7`；证据 ev_fe24f169695d7146cce654dc
- **结果 / supported**：For PaliGemma, policy-driven selection reduces average computational cost by 70\.39%; VQAv2 accuracy changes from 75\.2% under exhaustive selection to 74\.9%, and COCO CIDEr changes from 144\.7 to 143\.5\.
  陈述 `claim_e866eb57d62e39bc6168a97c663a051c`；证据 ev_fe24f169695d7146cce654dc
- **局限 / supported**：Rendering styles and parameters require dataset- and model-specific tuning; utilization levels and changes vary across datasets, architectures, and inputs, and Hybrid masking remains a heuristic sensitive to masking ratio, rendering style, and expert setup\.
  陈述 `claim_642efd9da80aa33f77f542e10fce65be`；证据 ev_4142779f17fa0fa326fb1b58, ev_a863872ff157a9bdc91c9bda
- **局限 / supported**：DnR cannot recover conceptual facts absent from the model; the paper characterizes its benefit as improving access to existing knowledge rather than augmenting knowledge\.
  陈述 `claim_f8f61f1b86fd88e7ba8f6b70ee5669d8`；证据 ev_15535ab06f0a218f2e46af76
- **作者贡献陈述 / supported**：The paper contributes a quantitative visual-utilization criterion for selecting and incorporating visual experts, a rendering interface that integrates experts without retraining, and a learned selector that approximates exhaustive expert evaluation\.
  陈述 `claim_1e2894410a0acade089cf96101632e56`；证据 ev_19eafa9caf7c2ae632cde690, ev_88ef732fb76f3bc6c1d466c3, ev_a863872ff157a9bdc91c9bda

## Dynamic Mixture-of-Experts for Visual Autoregressive Model

尚无当前文档版本的阅读卡片。

## Visual Boosting Techniques for Spatiotemporal Dense Pixel Visualizations

尚无当前文档版本的阅读卡片。

## Siamese High-Level Feature Refine Network for Visual Object Tracking

尚无当前文档版本的阅读卡片。

## VIANA: Visual Interactive Annotation of Argumentation

尚无当前文档版本的阅读卡片。

## A Literature Review Comparing Experts’ and Non-Experts’ Visual Processing of Graphs during Problem-Solving and Learning

尚无当前文档版本的阅读卡片。

## Chaos in DNA inversions \(Draft paper\)

尚无当前文档版本的阅读卡片。

## Refining posthuman aesthetic pedagogy for teacher education through multidisciplinary expert consultation

尚无当前文档版本的阅读卡片。

## Lexara-RF: Reference-Free Metrics for Evaluating Conversational Visual Analytics Agents

尚无当前文档版本的阅读卡片。

## Refining a taxonomy for guideline implementation: results of an exercise in abstract classification

尚无当前文档版本的阅读卡片。

## ExpertFlow: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling

尚无当前文档版本的阅读卡片。

## Almost an Expert

尚无当前文档版本的阅读卡片。

## Zoom-Refine: Boosting High-Resolution Multimodal Understanding via Localized Zoom and Self-Refinement

尚无当前文档版本的阅读卡片。

## Refining Child Pornography Law

尚无当前文档版本的阅读卡片。

## Bigraphical Refinement

尚无当前文档版本的阅读卡片。

## SegAgent: Exploring Pixel Understanding Capabilities in MLLMs by Imitating Human Annotator Trajectories

尚无当前文档版本的阅读卡片。

## SegAgent: Exploring Pixel Understanding Capabilities in MLLMs by Imitating Human Annotator Trajectories

文档：doc_d8186e2c40a9d4300de919e5；后端：codex

- **研究问题 / supported**：MLLMs struggle with fine-grained pixel understanding, while VQA and visual grounding do not effectively assess this capability\.
  陈述 `claim_0a17c63f1d4b3ebbb8f51b7afda7c09d`；证据 ev_7fcac78f0a4aa92986dae68e
- **方法 / supported**：HLMAT models interactive segmentation as a multi-step Markov Decision Process in which MLLMs generate text-based coordinates without specialized architectures or implicit tokens\.
  陈述 `claim_00efd3af44fd1aa3db3b0d97fd7eb22f`；证据 ev_7dbfe37bbc1fdfe1801f7367
- **方法 / supported**：SegAgent receives the current mask overlaid on the image and a task and object description, then outputs a textual click action; an interactive segmentation model uses the action history, current mask and image to update the mask\.
  陈述 `claim_c474d454691350c6aa73500806a06c38`；证据 ev_4b80da477964a5650edc9b4c, ev_8aeae7130e7a44ec562010e3
- **方法 / supported**：Training trajectories are simulated from ground-truth masks by placing clicks in false-positive or false-negative regions; generation limits trajectory length, stops at a reward threshold and discards actions below a reward-improvement threshold\.
  陈述 `claim_f126966541f27397ee62cd1d88bc47f8`；证据 ev_ca90af6f87f795a99e7a3a40, ev_e90d180ae5771c95fbda449b
- **方法 / supported**：StaR\+ retains rollout actions that increase reward, replaces reward-decreasing actions with simulator corrections, and fine-tunes on the corrected trajectories merged with the original trajectory dataset\.
  陈述 `claim_fead9af886a5410fe7f0e2ef273c07c7`；证据 ev_050dfdcb8f542a89c6f78c2e
- **方法 / supported**：At inference, the MLLM predicts the current mask's mIoU as a process reward\. Heuristic greedy search samples K candidate actions, evaluates their resulting masks using PRM, selects the highest predicted reward and returns the mask with the highest predicted reward across steps\.
  陈述 `claim_d1ba20f567e2e1aa4abd90cf207ddedc`；证据 ev_a5a6f1f2d15251dd456a83a9, ev_d5b9d5b55bba4585496cf018
- **数据 / supported**：RES comprises refCOCO, refCOCO\+ and refCOCOg\. HRES derives from HQSeg-44K using DIS5K and ThinObject5K; DIS5K contains 5,470 images across 22 groups and 225 fine-grained categories\.
  陈述 `claim_5137323fb078587d0d890fc48e616d3a`；证据 ev_7d55354dfbd98844c1e9c992, ev_94239693c8a03702c5b02cac
- **方法 / supported**：Evaluation uses cumulative IoU, with maximum steps T=7 and K=1 for RES, and T=11 and K=3 for HRES\.
  陈述 `claim_5afaf5ec4da6922969e66b50f3ba3c86`；证据 ev_2e468725ef84e27accf6992d
- **结果 / supported**：On refCOCO validation, SegAgent-LLaVA\+SAM scores 79\.20 compared with 71\.79 for Qwen box \+ SAM; on refCOCOg test\(U\), the corresponding scores are 74\.90 and 68\.94\.
  陈述 `claim_34ff41a6ca5dde8027af926f9b5ed555`；证据 ev_e82ee2055511ec08c8cccf07
- **结果 / supported**：A single round of StaR\+ yields ThinObject5K IoU of 86\.57 compared with the SFT baseline's 71\.45\. PRM alone improves performance by 2\.36 on DIS and 4\.09 on ThinObject; search with K=3 adds 7\.43 and 10\.59, respectively, over PRM alone\.
  陈述 `claim_c721ac454ceb1638414b01da8f19da21`；证据 ev_af59762e939ebb467e0f23ca
- **结果 / supported**：For annotation filtering, SegAgent-LLaVA has MAE 5\.58, MSE 175\.35, Pearson correlation 0\.91 and Spearman correlation 0\.90; SegAgent-Qwen has 6\.88, 193\.98, 0\.90 and 0\.87, respectively\.
  陈述 `claim_54a0b0c1b0476b082c0e359220d0d56b`；证据 ev_1675f7c93153b65e588502c4
- **局限 / supported**：Failures mainly involve incorrect initial target localization or inaccurate refinement coordinates, particularly points falling outside object boundaries\. The authors consider fine-grained pixel localization the principal performance bottleneck\.
  陈述 `claim_58fff044a29965c4026579a11013bfbd`；证据 ev_54a381b62ecc4597ce0345b4
- **局限 / supported**：Nearly optimal simulated trajectories lack diversity, constraining model generalization and robustness; noisy ground-truth masks can also produce erroneous training actions, especially near trajectory endpoints\.
  陈述 `claim_6a9f61c93bf4a44ca133ba7f816717b4`；证据 ev_468ca2597ea988ff97ff05b2, ev_ca90af6f87f795a99e7a3a40
- **局限 / supported**：More complex search strategies involve a computational complexity–performance trade-off; exploration of BFS and MCTS is left for future work\.
  陈述 `claim_e7a5ffaf30a3341c2a7fd2b6348b4f7a`；证据 ev_1d774a5ae20aa5af075a2c5d, ev_d5b9d5b55bba4585496cf018
- **作者贡献陈述 / supported**：The work introduces HLMAT as a protocol for fine-grained visual capability assessment, SegAgent trained on automatically generated human-like annotation trajectories, and HRES for studying complex vision-centered multi-step decisions\.
  陈述 `claim_33d5248df2cd54cd3cf0f2a50b8c5e3b`；证据 ev_9e126dfccfdf7973a23bdddd

## Les services publics economiques entre experts, praticiens et gouvernants dans le premier XX^e siecle: d'une configuration historique a l'autre

尚无当前文档版本的阅读卡片。

## Summarizing Scientific Articles: Experiments with Relevance and Rhetorical Status

尚无当前文档版本的阅读卡片。

## DRAgent: Discriminative Reasoning Agent for Referring Expression Segmentation

尚无当前文档版本的阅读卡片。

## Evolving, Not Training: Zero-Shot Reasoning Segmentation via Evolutionary Prompting

尚无当前文档版本的阅读卡片。

## Affordance2Action: Task-Conditioned Scene-level Affordance Grounding for Real-Time Manipulation

尚无当前文档版本的阅读卡片。

## FeVOS: Foresight Expression Video Object Segmentation

尚无当前文档版本的阅读卡片。

## ALTo: Adaptive-Length Tokenizer for Autoregressive Mask Generation

尚无当前文档版本的阅读卡片。

## Principled Approaches for Learning to Defer with Multiple Experts

尚无当前文档版本的阅读卡片。

## One-Stage Top-$k$ Learning-to-Defer: Score-Based Surrogates with Theoretical Guarantees

尚无当前文档版本的阅读卡片。

## Two-Stage Learning to Defer with Multiple Experts

文档：doc_a640fa1df871842d55c3510c；后端：codex

- **研究问题 / supported**：The study addresses learning to defer to multiple experts when a predictor is already available and jointly retraining it with a deferral function could be prohibitively costly, potentially taking weeks or months\.
  陈述 `claim_257786b81c9006c7aca4198d5ba09897`；证据 ev_318ac6549718bb129161cdd3
- **方法 / supported**：The approach first trains a predictor with a standard classification surrogate such as logistic loss, then learns a deferral function while keeping the predictor's scoring function fixed\.
  陈述 `claim_1055a738ada697a952e9080f8dbda0c0`；证据 ev_c9ab2dde9d92462f8d7f5cba, ev_efc83543eec8733cadcc2b87, ev_f535e28a5c5a4f9a2a84bf90
- **方法 / supported**：In the score-based setting, the second-stage surrogate combines a classification surrogate weighted by the indicator that the predictor is correct with expert-specific surrogate terms weighted by 1 minus each expert's deferral cost\.
  陈述 `claim_76b0b28083c28ca3aea1ec5393f381ac`；证据 ev_f535e28a5c5a4f9a2a84bf90
- **数据 / supported**：Experiments used CIFAR-10 and SVHN, with ResNet-4 as both predictor and deferral model and ResNet-10, ResNet-16 and ResNet-28 as experts\.
  陈述 `claim_ae8415130621affd0fda6861d0520fd0`；证据 ev_262ea7303102a13a7968638b, ev_c84ec09e7b4ea91ea21ace03
- **方法 / supported**：Training used logistic losses, Adam, batch size 128 and weight decay 1×10−4, for 15 epochs on SVHN and 50 epochs on CIFAR-10, without data augmentation\.
  陈述 `claim_203fff1173fd635415e05cfe3cd2a23e`；证据 ev_262ea7303102a13a7968638b
- **作者贡献陈述 / supported**：The paper introduces families of two-stage surrogate losses for both score-based and predictor-rejector deferral and establishes H-consistency bounds implying Bayes-consistency\.
  陈述 `claim_bf010ec3b5fd520f30c82c9293b4cf47`；证据 ev_3bfe366ac40512e19b60306a
- **结果 / supported**：The H-consistency guarantees require the first- and second-stage classification surrogates to admit the corresponding hypothesis-class consistency bounds; the resulting deferral bounds include minimizability-gap terms\.
  陈述 `claim_1bb6d0dd9cc7ab8633339237a01259e9`；证据 ev_1b2de2a6584ef6fc69f85b06, ev_210bb8fa32aa88009bd77fb6, ev_f285c656acfcc7a9121ad0f5, ev_f438ae4fd8fb05f7dc4c847c
- **结果 / supported**：For logistic losses in both stages, constant expert costs c\_j\(x,y\)=β\_j and hypothesis classes closed under scaling, the paper establishes realizable consistency in both frameworks, with zero expected deferral loss for the stated surrogate minimizers on realizable distributions\.
  陈述 `claim_a7cdd88504732bb8a913763f9f4463a8`；证据 ev_cc680f0e06020d529a0329a7, ev_fe67fc42913253429034a241
- **作者贡献陈述 / supported**：The paper also derives multiclass H-consistency bounds with one fixed scoring function, assuming the underlying hypothesis class is symmetric and complete\.
  陈述 `claim_13135079784b6a93a5546e7397c0a534`；证据 ev_0beaaea90ab257b1a96e5679, ev_4035e35041abc846c01f0000
- **结果 / supported**：Without base costs, overall accuracy for one, two and three experts was respectively 91\.85±0\.01%, 92\.77±0\.02% and 93\.30±0\.02% on SVHN, and 72\.63±0\.20%, 75\.84±0\.35% and 77\.68±0\.07% on CIFAR-10; results are means±standard deviations over three runs\.
  陈述 `claim_f58f3b63266abc631df37e0d386b7558`；证据 ev_262ea7303102a13a7968638b, ev_e7ca1a26a36352fb7a73c7cb
- **结果 / supported**：With base costs, overall accuracy for one, two and three experts was respectively 91\.66±0\.01%, 92\.05±0\.10% and 92\.19±0\.03% on SVHN, and 71\.73±0\.06%, 72\.31±0\.31% and 72\.42±0\.12% on CIFAR-10; results are means±standard deviations over three runs\.
  陈述 `claim_0abfbf28cf76971687111d0c28820cca`；证据 ev_262ea7303102a13a7968638b, ev_e7ca1a26a36352fb7a73c7cb
- **局限 / supported**：The authors report being unaware of established baselines for their two-stage deferral setting\.
  陈述 `claim_96502dc3b21c8b1b1f595666a16c2157`；证据 ev_a1380b25017ad89c123d7f44
- **局限 / supported**：The deferral cost function is typically selected through cross-validation in practice; developing a principled selection method is left for future research\.
  陈述 `claim_3d43aacab90ca89408e9bab2d0cee52a`；证据 ev_e97fad690ac4c117f87a7e60

## Principled Approaches for Learning to Defer with Multiple Experts

尚无当前文档版本的阅读卡片。

## Learning to Defer to Multiple Experts: Consistent Surrogate Losses, Confidence Calibration, and Conformal Ensembles

尚无当前文档版本的阅读卡片。

## When More Experts Hurt: Underfitting in Multi-Expert Learning to Defer

尚无当前文档版本的阅读卡片。

## Cost-Sensitive Learning to Defer to Multiple Experts with Workload Constraints

尚无当前文档版本的阅读卡片。

## Active Learning in Incomplete Label Multiple Instance Multiple Label Learning

尚无当前文档版本的阅读卡片。

## No Need for Learning to Defer? A Training Free Deferral Framework to Multiple Experts through Conformal Prediction

尚无当前文档版本的阅读卡片。

## Mastering Multiple-Expert Routing: Realizable $H$-Consistency and Strong Guarantees for Learning to Defer

尚无当前文档版本的阅读卡片。

## Learning to Defer with Limited Expert Predictions

尚无当前文档版本的阅读卡片。

## How can humans leverage machine learning? From Medical Data Wrangling to Learning to Defer to Multiple Experts

尚无当前文档版本的阅读卡片。

## Consistent Estimators for Learning to Defer to an Expert

尚无当前文档版本的阅读卡片。

## To Defer or Not to Defer? Online Learning with LLMs and Congested Experts

尚无当前文档版本的阅读卡片。

## CT-PromptSAM: Collaborative Prompting With Hybrid CNN–Transformer for Remote Sensing Semantic Segmentation

尚无当前文档版本的阅读卡片。

## Balancing Multi-Target Semi-Supervised Medical Image Segmentation with Collaborative Generalist and Specialists

文档：doc_fa9b959a24eeb86debc04b6c；后端：codex

- **研究问题 / supported**：The paper addresses scale imbalance in multi-target semi-supervised medical image segmentation: large targets may receive priority, causing smaller targets and background to be misclassified as larger structures\.
  陈述 `claim_41f56349d20ff41185a680589ec70e48`；证据 ev_8aa809e36a36c25197be4d5c
- **方法 / supported**：CGS uses a teacher–student framework with weak-to-strong consistency, a general segmentation branch, and class-specific specialist heads that distinguish background, their target class, and all remaining target classes\.
  陈述 `claim_fb68ba7901d06baff3348b6f7cc36fa6`；证据 ev_133bdc1b559c15291cc6485e, ev_a56978023a84fb4de16de8cd
- **方法 / supported**：The teacher is updated by exponential moving average of the student; supervised and unsupervised segmentation losses combine cross-entropy and Dice losses\.
  陈述 `claim_91292fa112e51678829d818d56577cfc`；证据 ev_5f4ea9e0e5d12f221d4930ab, ev_7fa2e43a7d912370bfd7316d
- **方法 / supported**：Bidirectional cross-branch consistency exchanges pseudo-labels between the generalist and specialists, while a consensus-based loss jointly optimizes regions where the branches agree\.
  陈述 `claim_cd869fc0dad090db4557c94a58551fa4`；证据 ev_857369d8d50c4465622cd09c
- **方法 / supported**：IHED filters specialist pseudo-labels using inter-head agreement: either every head predicts background, or one head predicts its target and all other heads predict their remaining classes\.
  陈述 `claim_5c3a4ad1a1c1d06c25e91c49c6af09ff`；证据 ev_c288b09bdc346e4b7b6fd002
- **数据 / supported**：ACDC contains scans from 100 patients with three segmentation targets and uses a 7:1:2 training/validation/test split\. SegTHOR has four targets and 28/4/8 training/validation/test volumes\. Synapse contains 30 CT scans annotated with eight abdominal organs and uses 18 training and 12 test cases\. Input slices are resized to 224 × 224\.
  陈述 `claim_a9634cc1982df3edfcc72422124b9c96`；证据 ev_4607c22698f2fe975053c51b, ev_64a53da69e89deb158de317e
- **数据 / supported**：Experiments use three different seeds on each of the three datasets and report means and standard deviations\.
  陈述 `claim_dc5b04a70b655032ed264475a2dfe434`；证据 ev_e38ecea0ac925653f8827691
- **结果 / supported**：On ACDC, CGS achieves DSC of 88\.83±0\.14% with 3 labeled scans \(5%\) and 89\.83±0\.11% with 7 labeled scans \(10%\), compared with FixMatch's 87\.27% and 88\.89%, respectively\.
  陈述 `claim_050023a870c1366b004d8ae12a48f9a3`；证据 ev_086bc48fe01a33c40228b24d
- **结果 / supported**：On SegTHOR, CGS achieves DSC of 81\.74±1\.29% with 3 labeled scans \(10%\) and 83\.69±1\.45% with 6 labeled scans \(20%\), compared with FixMatch's 79\.38% and 81\.94%, respectively\.
  陈述 `claim_3e7a80f654a7776e7905eb0f5a7f5e30`；证据 ev_ee7bc629e785a1f393373208
- **结果 / supported**：On Synapse with 3 labeled volumes, CGS achieves average DSC of 60\.20±0\.37% and average HD95 of 61\.40±1\.04, compared with FixMatch's 58\.33% and 80\.79\. CGS does not improve every organ: its left-kidney and spleen DSC values are 65\.39% and 61\.89%, versus FixMatch's 71\.71% and 69\.15%\.
  陈述 `claim_09213006d4d2501926d21e6e0e3ab01e`；证据 ev_d7b12f522b145bff1f4e28d2
- **局限 / supported**：At ACDC's 5% labeled setting, CGS has worse 95HD than BCP: 2\.42±0\.08 versus 1\.90\.
  陈述 `claim_23b7086859138f89e53a512d89d60bdd`；证据 ev_086bc48fe01a33c40228b24d
- **局限 / supported**：Training memory increases with the number of classes: at batch size 24 with teacher and student networks, reported GPU memory rises from 7\.13 GB for 3 classes to 10\.62 GB for 8 classes\.
  陈述 `claim_a8dcd132fb96d98ca6c9b88dd2f803b0`；证据 ev_d7f88ad0e17b0bd32c33fc78
- **局限 / supported**：The authors state that IHED and cross-branch consistency depend on the specialized segmentation heads, preventing independent ablations of these modules without that foundation\.
  陈述 `claim_93e8a0bd7f247dd00ab8d923226c9581`；证据 ev_20c96eab6f68dc9d5621ed92
- **局限 / supported**：More comprehensive simultaneous handling of occurrence and scale imbalance remains future work; adding resampling to CGS raises Synapse average DSC from 60\.20% to 64\.93%\.
  陈述 `claim_de53bf8872aae8f67100d41ecc21360e`；证据 ev_161ef218bac27709f7bb2fd5, ev_9871388095b2b11ec516f275
- **作者贡献陈述 / supported**：The authors present CGS, its cross-branch consistency losses, and IHED as contributions for more balanced training and improved pseudo-labels\. Specialists are discarded at inference, so the collaboration occurs during training and inference uses the general branch without additional specialist memory or time costs\.
  陈述 `claim_692842d6418bab7340167ed3185958ab`；证据 ev_64a53da69e89deb158de317e, ev_867063645d3106089853bdf0

## SemiSAM-O1: Pushing the Boundary of Annotation-Efficient Medical Image Segmentation with Generalist Knowledge Fusion

尚无当前文档版本的阅读卡片。

## GSCo: Towards Generalizable AI in Medicine via Generalist-Specialist Collaboration

尚无当前文档版本的阅读卡片。

## EoS-FM: Can an Ensemble of Specialist Models Act as a Generalist Feature Extractor?

尚无当前文档版本的阅读卡片。

## An Open Natural Language Processing Development Framework for EHR-based Clinical Research: A case demonstration using the National COVID Cohort Collaborative \(N3C\)

尚无当前文档版本的阅读卡片。

## TAGS: A Test-Time Generalist–Specialist Framework with Retrieval-Augmented Reasoning and Verification

尚无当前文档版本的阅读卡片。

## On-Device Collaborative Language Modeling via a Mixture of Generalists and Specialists

尚无当前文档版本的阅读卡片。

## U-CFR: Uncertainty-Guided Cascade Forward Refinement for Interactive Segmentation

文档：doc_3628754cfa67f573f6b520b8；后端：codex

- **研究问题 / supported**：The paper addresses interactive segmentation errors around fine details, small objects and complex boundaries, and refinement loops that distribute effort uniformly rather than targeting critical error regions\.
  陈述 `claim_67f1046145e522815064840053aa10b1`；证据 ev_6f63a458bee2ece085b7a9cd, ev_e08a8633b6436b2a1b155ec3
- **方法 / supported**：The architecture uses a shared MAE-pretrained plain ViT with an FPN; positive and negative click disk maps are concatenated with the previous segmentation mask and embedded alongside image patches\.
  陈述 `claim_b03cbeeef4d2025ae2be729190a3bdd8`；证据 ev_e935e4b874cabc24d8013162
- **方法 / supported**：An auxiliary edge head receives supervision from edge maps generated from segmentation masks using differential operators such as Laplacian or Sobel filters; training combines normalized focal segmentation loss with Dice edge loss\.
  陈述 `claim_21ddd5330ba7b8e6fc00a082b487c6c5`；证据 ev_4867bd5181dd0d1dc24befdd, ev_c78daeac7664b7bcfb16509f
- **方法 / supported**：U-CFR computes predictive uncertainty as 1−\|2Pseg−1\| and multiplies it by the normalized Sobel gradient magnitude of the segmentation probability map to obtain boundary-aware uncertainty\.
  陈述 `claim_661d79345d66bccaa920699241606c00`；证据 ev_58a76a775ab9a983490e7ca3, ev_93f1532dbc0206767f92b62c
- **方法 / supported**：After each user interaction, U-CFR selects the maximum boundary-aware uncertainty location as a pseudo-click candidate\. It skips candidates with probabilities in the inclusive interval \[0\.49, 0\.51\], assigns positive labels above 0\.51 and negative labels below 0\.49, then feeds accepted clicks back into the model\.
  陈述 `claim_26a792c324aa9c8bbaf21f63fb507f13`；证据 ev_3f3a0200758a5b7f0d61552b, ev_58a76a775ab9a983490e7ca3
- **数据 / supported**：Training uses SBD, reported as containing 8,498 training images and 2,820 test images\. Evaluation includes SBD, COCO\_MVal, Pascal VOC, DAVIS, Berkeley, GrabCut, BraTS, ssTEM and OAIZIB\.
  陈述 `claim_70927a5ebeb2a5dd4c3c875c53aa386c`；证据 ev_13de5fd89f1e6dffe877ae36
- **数据 / supported**：Out-of-domain evaluation uses BraTS, ssTEM, OAIZIB and a curated irradiated-metallic SEM precipitate dataset; models trained solely on SBD are applied without fine-tuning\.
  陈述 `claim_8f51d824603848d731a8bf45b0c4c354`；证据 ev_8062531f238f90af764cf9bc
- **结果 / supported**：Adding the edge head reduces NoC@90 relative to SimpleClick from 2\.81 to 2\.70 on Pascal VOC, 4\.07 to 3\.90 on COCO\_MVal and 5\.24 to 5\.01 on SBD\.
  陈述 `claim_1999dce50acdcddb89e7403c7dbda704`；证据 ev_32f7e726071809ea757e1dd8
- **结果 / supported**：On COCO\_MVal, the edge-head model increases first-click mIoU from 61\.03% to 66\.05% and first-click NSDS from 0\.29 to 0\.31\.
  陈述 `claim_cba1fa1631bbf5830afbd2c361c54e69`；证据 ev_420ae82d8eb7d3c889da1aa8
- **结果 / supported**：Compared with one-step CFR on the same base model, one-step U-CFR reduces NoC@90 from 2\.72 to 2\.64 on Pascal VOC, 3\.88 to 3\.80 on COCO\_MVal and 4\.98 to 4\.94 on SBD\.
  陈述 `claim_e2ffba7b7342f08528ab103e69cb1901`；证据 ev_e65e70bdd19e6b8521b3a0ef
- **结果 / supported**：Compared with SimpleClick, the full model reduces Berkeley NoC@90 from 2\.46 to 2\.19, reported as an 11% improvement, and NoC@95 from 6\.71 to 6\.07, reported as a 9\.5% improvement\.
  陈述 `claim_c31f22889dffd1572bcd9cd4041d022c`；证据 ev_02779b600bb8c5126c542269
- **结果 / supported**：On OAIZIB, the reported model reduces NoC@85 from SimpleClick's 18\.44 to 16\.68, increases first-click mIoU from 2\.52% to 6\.32%, and increases NSDS at 10 clicks from 0\.58 to 0\.72\.
  陈述 `claim_e78e285d993b69eb316197e4626812a5`；证据 ev_8062531f238f90af764cf9bc
- **结果 / supported**：On SEM-Precipitate, the reported model requires 7\.92 clicks for 85% IoU versus SimpleClick's 10\.37 and achieves NSDS of 0\.87 versus 0\.81 at 10 clicks and 0\.93 versus 0\.89 at 20 clicks\.
  陈述 `claim_26c36bfd854b8183a0c5ab43e557eca1`；证据 ev_be50fdb0447d2c70c672134e
- **局限 / supported**：U-CFR does not improve every refinement metric: Table 2 reports lower mIoU at 10 clicks than CFR on Pascal VOC \(97\.80% versus 97\.94%\), COCO\_MVal \(94\.92% versus 95\.11%\) and SBD \(92\.72% versus 92\.85%\)\.
  陈述 `claim_3f7c12e422ebf3058cbb935ffa94409c`；证据 ev_e65e70bdd19e6b8521b3a0ef
- **局限 / supported**：The edge head is intended as a training-time regularizer rather than a geometrically precise standalone boundary detector; the paper notes that topology-aware losses could better capture thin-boundary connectivity\.
  陈述 `claim_652c1ec74d02a6265e253144a45111d6`；证据 ev_4867bd5181dd0d1dc24befdd, ev_c78daeac7664b7bcfb16509f
- **作者贡献陈述 / supported**：The paper's stated contributions are an auxiliary edge-detection head for boundary-aware representations and an inference-time uncertainty-guided pseudo-click refinement framework that operates without additional user input\.
  陈述 `claim_19be5410599d0cbda4655d59ed5965d7`；证据 ev_146f616ede5204f8dc5155ee, ev_f8182f44a1746d2412d97d43

## 3D Kidneys and Kidney Tumor Semantic Segmentation using Boundary-Aware Networks

尚无当前文档版本的阅读卡片。

## NeuroRefiner: Morphology-Aware Multi-agent Refinement for 3D Fluorescence Microscopy Neuron Segmentation

文档：doc_74d94da70bc6c9179401aa79；后端：codex

- **研究问题 / supported**：Sparse, filamentous neurons and substantial noise make 3D fluorescence microscopy segmentation difficult; limited receptive fields in sliding-window inference hinder long-range morphology modeling and can produce fragmented masks\.
  陈述 `claim_b92ea458b5277700f337e4275ac55236`；证据 ev_6511a606fb17b1ba89781dc9
- **方法 / supported**：NeuroRefiner iteratively combines global inspection, region-specific instruction generation, TopoRefineNet editing, and validation of local refinements; it stops when morphology is judged complete or the maximum iteration count is reached\.
  陈述 `claim_db409fc8cc41b9bc2c5cba000457e703`；证据 ev_66bac5a1c5515d0d4aaf1ad4, ev_ddeecb67e2ea5b84f49d122d
- **方法 / supported**：The Global Inspector uses global and patch-level mask projections together with the zeroth Betti number, computed as the connected-component count in each corresponding 3D region, to identify blocks with false-positive or false-negative errors\.
  陈述 `claim_56cda9f870cb7406020c85ad6a7adf87`；证据 ev_6d6fa1c8b6871e0cdab84db8
- **方法 / supported**：The Refinement Advisor compares image and mask projections in the xy and yz planes and generates instructions specifying both a spatial location and an editing operation\.
  陈述 `claim_b5be9e9cc24c33afa5ea98176da06cfd`；证据 ev_ce17b2813fda76afd794fc75
- **方法 / supported**：The main method description requires the Change Validator to check instruction adherence and topological improvement, retrying failed edits with new instructions and escalating persistently unresolved blocks to human experts\.
  陈述 `claim_e376a5eca89b16e1a5145f11de1577ae`；证据 ev_cfc1138ca153b06aea3b053e
- **方法 / supported**：TopoRefineNet encodes the concatenated image and noisy mask, embeds instructions with a frozen pretrained text encoder, and fuses text with the deepest visual features using cross-attention before decoding a refined mask\.
  陈述 `claim_0fab1943fb43cc4cb9e5f2b06028177d`；证据 ev_b36fafeadedc252d16ea85c1, ev_e803a5d85ffa26e9bd953cbd
- **方法 / supported**：TopoRefineNet uses two training stages: synthetic morphological defects with perturbation-derived instructions, followed by real segmentation errors with instructions synthesized using Qwen3-VL\.
  陈述 `claim_4d67d683a67fd9fb4049090eecb43728`；证据 ev_7f6b8e2598c8f68863de063f, ev_bc8951167e8eee81dcc543e1
- **数据 / supported**：Evaluation uses BigNeuron, CWMBS, and ZBFWB with previously established training–testing splits\. BigNeuron contains expert-annotated neurons from multiple species and imaging platforms; CWMBS includes 83 strong-noise and 162 weak-signal volumes of 256×256×256 voxels at 0\.2×0\.2×1 µm resolution\.
  陈述 `claim_1afe89a6ff4faa3edc1e8f92fe74bcb9`；证据 ev_00ebe63c442a9ec637016570, ev_7f6b8e2598c8f68863de063f
- **数据 / supported**：ZBFWB contains confocal images from 6-day-old zebrafish larvae, with dimensions of 1000×2000×250 voxels and resolution of 0\.5×0\.5×1 µm\.
  陈述 `claim_f45696c0f2e067b628b11c1dfb915a7d`；证据 ev_7f6b8e2598c8f68863de063f, ev_b69dbd8e2cbeb706581a07c1
- **数据 / supported**：The second training stage selects 5,310 error-containing volumes of 128×128×64 voxels from predictions by 3D U-Net, V-Net, UNETR, nnFormer, and SwinUNETR, using connected-component and F1 scores\.
  陈述 `claim_f38787e56950f5a77daafede182cdf22`；证据 ev_70ddd39113f52ec5a71a5545
- **方法 / supported**：Experiments use Qwen3-VL-8B for all agents, Qwen3 Embedding 0\.6B for text embeddings, 128×128-pixel inspection regions, and a maximum of 5 iterations\.
  陈述 `claim_945e55ec3b1c033402cdb55a2fc56b9f`；证据 ev_b69dbd8e2cbeb706581a07c1, ev_c9eff9e9a48d19bb35e2bb49
- **结果 / supported**：On ZBFWB, NeuroRefiner initialized from 3D U-Net achieves F1 of 86\.13%, compared with 72\.94% for 3D U-Net and 83\.11% for GBP-Net: gains of 13\.19 and 3\.02 percentage points, respectively\.
  陈述 `claim_480e4cb97a85514046b4767d963ff6b9`；证据 ev_00ebe63c442a9ec637016570, ev_58b5bb17544d7c446ca5958c, ev_e57e89c27283311639d49804
- **结果 / supported**：On BigNeuron, NeuroRefiner achieves F1 of 90\.05%, SSD of 6\.94, and MES of 72\.02, compared with 86\.33%, 11\.27, and 66\.82 for its 3D U-Net initialization\.
  陈述 `claim_6380b8eb0bedfd77df0735b7ede72ebf`；证据 ev_00ebe63c442a9ec637016570, ev_e57e89c27283311639d49804
- **结果 / supported**：On CWMBS, refining 3D U-Net masks increases F1 from 74\.94% to 90\.07% on weak-signal volumes and from 79\.23% to 86\.37% on strong-noise volumes; refining nnUNet masks yields F1 of 91\.85% and 84\.84%, respectively\.
  陈述 `claim_09ea9b15a536f441cd0677f078f26ec0`；证据 ev_70ddd39113f52ec5a71a5545, ev_a4764e1eb47bacf9fcb251fb
- **结果 / supported**：In agent ablations, the full system achieves F1 of 86\.13% on ZBFWB and 88\.83% on CWMBS, versus 76\.39% and 79\.96% when TopoRefineNet refines all blocks without agents\.
  陈述 `claim_e2766091015dccdaea78269b49d219e3`；证据 ev_bcc8eb25366fe99c72a7b15f, ev_ffb0953ebab66abd33f5294e
- **结果 / supported**：Training only on real defects from scratch increases the Change Validator rejection rate by a reported 23\.57% and degrades F1 by a reported 1\.17% on ZBFWB and 1\.89% on CWMBS relative to two-stage training\.
  陈述 `claim_511bde4bc161672d49ae91d2161ae0d7`；证据 ev_d082e7277f3537c866ce4a9e, ev_e43bc1d0edd163a90ca42284
- **局限 / supported**：NeuroRefiner's 2D maximum-intensity projections lose depth information in dense regions, hindering precise 3D error localization; neuron overlap can also interfere with inspection of some complex volumes\.
  陈述 `claim_688af1efd840b66e391cf7d5fdf8999d`；证据 ev_b48eabbb609a788b37a08624, ev_d082e7277f3537c866ce4a9e
- **作者贡献陈述 / supported**：The authors present NeuroRefiner as the first LLM-based multi-agent system for topology-aware iterative neuron segmentation optimization, and contribute TopoRefineNet with a two-stage training strategy for instruction-guided structural editing\.
  陈述 `claim_076d25bc58a8c73920e81e4f9e9050bb`；证据 ev_e983fdedc095873b3c0a2f92
- **作者贡献陈述 / supported**：The framework performs progressive refinement without agent training and exposes diagnostic decisions and editing instructions for auditing\.
  陈述 `claim_5164c58738d816b5708fc72cb6fdbfe3`；证据 ev_d082e7277f3537c866ce4a9e, ev_ddeecb67e2ea5b84f49d122d

## BRIGHT: A Collaborative Generalist-Specialist Foundation Model for Breast Pathology

尚无当前文档版本的阅读卡片。

## SC-CoSF: Self-Correcting Collaborative and Co-Training for Image Fusion and Semantic Segmentation

尚无当前文档版本的阅读卡片。

## Improving Policy Optimization with Generalist-Specialist Learning

文档：doc_c038e863ea0192aea65a5d2f；后端：codex

- **研究问题 / supported**：Training reinforcement-learning policies over many environment variations can be inefficient and plateau at suboptimal performance\.
  陈述 `claim_cc2098ccb80b9a321963644d2c325dd2`；证据 ev_ad704acc19c856c400e94e2e, ev_c9897dc006f4374fca4c744a
- **方法 / supported**：GSL trains a generalist until a performance plateau, initializes specialists from its checkpoint on disjoint subsets of the lowest-performing variations, and resumes generalist training using demonstrations from specialists and from the generalist on remaining variations\.
  陈述 `claim_72783b68d58a60d6ee6e5b26a199a71d`；证据 ev_d3ddbbc2f6b0a8ccea2e3026
- **方法 / supported**：The experiments combine PPO or PPG with DAPG on Procgen, PPO with DAPG on Meta-World, and SAC with GAIL on ManiSkill, including online environment interactions during demonstration-guided learning\.
  陈述 `claim_bed2ddfe92dd0948e2bc823d9dfb4d08`；证据 ev_309ae504f55108ddc6570b87, ev_a5aa56e14db79b0e5851b8f8
- **数据 / supported**：Procgen evaluation uses seven selected challenging environments, with 1024 training levels and 1000 unseen test levels per environment; reported results average five runs with standard deviations\.
  陈述 `claim_906e5227f1a2eb4b8409bcfdeb800f52`；证据 ev_309ae504f55108ddc6570b87, ev_e3764f504c1a9d35a5ff3f5a, ev_e41959f9ef6d4ce04d37a4d1
- **数据 / supported**：Meta-World experiments use V2 MT-10 and MT-50, containing 10 and 50 skills respectively, and evaluate performance only on training environments\.
  陈述 `claim_b6a45c85e7962a46b39ca23c69318884`；证据 ev_e3764f504c1a9d35a5ff3f5a
- **结果 / supported**：On BigFish, PPO-based GSL achieves training return 31\.1±0\.8 versus PPO's 24\.6±0\.7, and test return 30\.0±0\.5 versus 24\.3±1\.1; these are raw episode reward means and standard deviations over five runs\.
  陈述 `claim_c85a6f2171b40ae9f70de12c2990da8b`；证据 ev_309ae504f55108ddc6570b87, ev_a05dcc49215a19d83c2d6da6
- **结果 / supported**：Meta-World training success rates increase from 58\.4±10\.1% to 77\.5±2\.9% on MT-10 and from 31\.1±4\.5% to 43\.5±2\.2% on MT-50, averaged over five runs\.
  陈述 `claim_d34fe37d6a4d319fbd2b90d4a21f7269`；证据 ev_309ae504f55108ddc6570b87, ev_5e225a674cc76ac751b6468f
- **结果 / supported**：In timing ablations on BigFish and PushChair, launching specialists before the generalist plateau worsens sample complexity; training specialists from scratch is reported as inefficient and ineffective\.
  陈述 `claim_b6c714a93c16599c98100030c787759e`；证据 ev_05f3af2f5c490c6eae385115
- **结果 / supported**：On BigFish, GSL with behavior cloning obtains training/test returns of 25\.3±0\.4/22\.1±0\.9, compared with 31\.1±0\.8/30\.0±0\.5 for GSL with DAPG and 24\.6±0\.7/24\.3±1\.1 for PPO alone\.
  陈述 `claim_01bbc314251aa32528f3f236bbb23a54`；证据 ev_67503e1ef9a162cfc5b0f79a
- **局限 / supported**：ManiSkill evaluation is restricted to a reduced PushChair setup with eight chairs because physical simulation and point-cloud rendering are expensive, with an observed speed of 30 environment steps per second\.
  陈述 `claim_8fedbdaf0b151a36a4e069cc5085c209`；证据 ev_48b5ae36db41368c105eb0c9
- **局限 / supported**：The plateau criterion requires hyperparameter tuning: starting too early loses initial generalist efficiency, while starting too late wastes samples and could make specialist-guided fine-tuning harder\.
  陈述 `claim_6242dff2366988517238293d055311e6`；证据 ev_31ff59cac0606136c36fc6aa, ev_67503e1ef9a162cfc5b0f79a
- **局限 / supported**：The DnC RL comparison was skipped because the authors found its quadratic computational complexity in the number of specialists computationally intractable in their setup\.
  陈述 `claim_06665a95209a7663f2af41a60c50710a`；证据 ev_ab68da109d1651d59bf5de1a
- **作者贡献陈述 / supported**：The paper contributes a generalist-specialist meta-algorithm integrating reinforcement learning with learning from demonstrations, and identifies specialist-training timing as a key design choice\.
  陈述 `claim_1992b4942bfd08e46058a2d51d7f78c4`；证据 ev_baf7ea490f2740a3cfac4c69, ev_bc9e0f7f1c6a3d8aa24adc97

## Generalist Open-World Temporal Perception

尚无当前文档版本的阅读卡片。

## GSL-PCD: Improving Generalist-Specialist Learning with Point Cloud Feature-based Task Partitioning

尚无当前文档版本的阅读卡片。

## VISTA3D - A Unified Segmentation Foundation Model For 3D Medical Imaging

文档：doc_f2ae2239a302703043a71464；后端：codex

- **研究问题 / supported**：Manual 3D medical image segmentation is time-consuming, while task-specific expert models require data curation and engineering expertise; rare pathologies and animal data also motivate zero-shot segmentation and interactive correction\.
  陈述 `claim_ead15b4acb63fcd33b28da3d06046649`；证据 ev_7abcc58f5479d9854745e852, ev_f673c39efae44bfb3fccbdb4
- **作者贡献陈述 / supported**：VISTA3D unifies automatic segmentation for 127 supported classes, interactive editing, and 3D point-prompted zero-shot segmentation through two branches sharing an image encoder\.
  陈述 `claim_a6eb998423f9327d2be21408d5b75f1b`；证据 ev_d0a444e3c1665ed931de8d5b
- **方法 / supported**：The model uses a MONAI SegResNet CNN backbone, 128-voxel cubic training patches, and sliding-window inference\. Its automatic head accepts a class index and produces a binary segmentation; its interactive head accepts positive and negative 3D clicks\.
  陈述 `claim_9ef8961dbcce582815cda453666f1f91`；证据 ev_42b1587f0d110edccab90427, ev_a6d4e0f9f7c110cbeadf55e0, ev_be03b4a69996f461196d63c1
- **方法 / supported**：Interactive refinement merges the independent branch outputs by adding or removing only connected-component regions containing the relevant clicks, aiming to avoid unintended changes to correct segmentation regions\.
  陈述 `claim_7692a158c6549c75de98867c5f455c18`；证据 ev_e48e371374b873b9a25d8fb2, ev_e7432595d4bb94f08dbef5ee
- **数据 / supported**：The curated dataset contains 11,454 CT volumes from in-house and public sources\. Each source is randomly split into 64% training, 16% validation, and 20% test data; TotalSegmentator supplies 117-class pseudo-labels and SAM supplies supervoxels for every scan, with unreliable pseudo-labels removed by post-processing\.
  陈述 `claim_7fd36cd17422e9b0d6f609feb43734f9`；证据 ev_cece72e3c8156ebf6163681a, ev_e7432595d4bb94f08dbef5ee
- **方法 / supported**：The proposed supervoxel method sums upsampled SAM ViT-H features extracted in axial, coronal, and sagittal views, then applies SLIC with nsegments = 100 and sigma = 3\. These supervoxels train the interactive branch to transfer SAM knowledge to 3D segmentation\.
  陈述 `claim_23a8a17778dbee73a1697fefecce1e6e`；证据 ev_2ebe6904acde42e3f4621e02, ev_f5a612daacfd6e48d7c1edd9
- **方法 / supported**：Training proceeds through interactive-branch training, interactive finetuning with dataset oversampling, automatic-branch training with the encoder frozen, and automatic finetuning using MAISI-generated anomalies to augment under-represented classes\.
  陈述 `claim_18b4c5d12043f32b7d645c4e43f74904`；证据 ev_0adce17f3722cb52357e2803, ev_2ebe6904acde42e3f4621e02, ev_818fc2a99b358e40300ad6f8
- **结果 / supported**：Across the 14 datasets in Table 1, average Dice scores are 0\.706 for Auto3DSeg, 0\.718 for nnUNet, 0\.711 for VISTA3D automatic segmentation, 0\.760 for VISTA3D point segmentation, and 0\.792 for VISTA3D automatic segmentation with point correction\.
  陈述 `claim_92014ad5643f82196be0564d122d0ed7`；证据 ev_13347963e3c464ecf99c5af3, ev_b8c5a394b480d13996212dd1
- **局限 / supported**：In the supported-class evaluation, a 'single point' means one click per sliding-window patch, and a click affects only the 128-voxel cubic patch containing it; correction clicks are simulated from false-positive or false-negative regions\.
  陈述 `claim_83179729237520adfb56a04a4a46e495`；证据 ev_6b4df8e285d05a174518ba7e
- **局限 / supported**：TotalSegV2 comparisons are biased toward nnUNet and TotalSegmentator because the ground truth was generated by pretrained TotalSegmentator, and its training data may include the study's test split\.
  陈述 `claim_d6732c09fb30c004d80093ad9eb4a4b8`；证据 ev_b8c5a394b480d13996212dd1
- **数据 / supported**：Zero-shot evaluation uses four external datasets: 140 mouse micro-CT scans at 0\.2 mm resolution with four annotated organs, 210 C4KC-KITS kidney-tumor scans, 53 adrenocortical-carcinoma scans, and 105 hepatocellular-carcinoma scans\.
  陈述 `claim_0a21c05b7d740c15ab93c16d9531abf2`；证据 ev_33f34993e04e383d9d431ac7, ev_91db90c1f3eddc9915f9d782
- **结果 / supported**：The authors report superior zero-shot accuracy with reduced annotation effort and report that removing supervoxel training worsens zero-shot performance; the supplied figure text does not provide numerical Dice curves\.
  陈述 `claim_1b78422c5ca95d80fee8a6f5933ee2f2`；证据 ev_13347963e3c464ecf99c5af3, ev_91db90c1f3eddc9915f9d782
- **结果 / supported**：On a 16 GB V100 GPU, a 512 × 512 × 512 volume resampled to 1\.5 mm isotropic spacing takes 2m09s for VISTA3D's default 118-class segmentation versus 12m01s for TotalSegmentator's default 117-class segmentation, without special model optimization\.
  陈述 `claim_596c6aa0fe226478bf6f09de5c2b7d63`；证据 ev_33f34993e04e383d9d431ac7
- **结果 / supported**：With one training case, VISTA3D achieves held-out average Dice scores of 0\.926 on mouse micro-CT and 0\.795 on WORD, compared with 0\.820/0\.214 for Auto3DSeg, 0\.759/0\.185 for nnU-Net, and 0\.791/0\.779 for TotalSegmentator\. At 100 WORD training cases, VISTA3D and TotalSegmentator both score 0\.875\.
  陈述 `claim_e764f012164c6a5e14c998dbe490e5b5`；证据 ev_13cbe13e632bf0dcf21a00b3, ev_91db90c1f3eddc9915f9d782
- **局限 / supported**：Extending support to MRI and PET, improving zero-shot interaction, and validating and integrating the workflow with clinical partners are identified as future work\.
  陈述 `claim_2da704822a1f516da213cbe1d121f15c`；证据 ev_03692352e187cb4f620f0baa, ev_5711344a9ec9d691eb99f65f

## REDNet - Reliable Evidential Discounting Network for Multi-Modality Medical Image Segmentation

文档：doc_a7db2305ee83861d8943cfa0；后端：codex

- **研究问题 / supported**：Medical image segmentation can degrade when one or more modalities contain imperfections; existing models may fail to detect these imperfections or dynamically adjust to data quality, particularly when test and training data differ substantially\.
  陈述 `claim_f2b1020c3ab93c6a3514c4f13d6267d8`；证据 ev_9f11b1f325573c57d24f81b6
- **方法 / supported**：REDNet processes 3D volumes with separate modality-specific subjective logic networks, evaluates intra-modality consistency and cross-modality differences, discounts evidence at each pixel, and fuses it using Dempster–Shafer evidence theory\.
  陈述 `claim_10875d77ca6f6cc880b31fa4a68a6bba`；证据 ev_a9a9390531f391c2541f9a3e, ev_b8f0be9b27964ef596323260
- **方法 / supported**：ICEM compares evidence singular values with stored training-data singular values using the minimum class-normalized L1 distance, and combines this consistency metric with convolutional features\.
  陈述 `claim_8adcaf8524b546c5b0caf96ed31e0362`；证据 ev_293a5211674ddc0f02cc5c77, ev_56f5ceb5bf39928f04915e36
- **方法 / supported**：CDAM constructs differential evidence between modalities and processes concatenated evidence through convolutional layers; larger discrepancies reduce the cross-modality weight and lead to greater discounting\.
  陈述 `claim_7f719c5ff2fd39a5003bb2999319ebde`；证据 ev_0adbdda80c725fbc6042b73d, ev_8a3f0ac9cf8864f209200380, ev_fece48f1a33611bf1c7d813a
- **方法 / supported**：The discounting module combines intra-modality and cross-modality information, with both weighting hyperparameters set to 1, before merging discounted opinions using Reduced Dempster’s Combination Rule\.
  陈述 `claim_df94e5029721914fbafb45106c51af3e`；证据 ev_3ab530c6dfcfbb8ba153eb61, ev_5217e4910db03f7f2120effe
- **数据 / supported**：BraTS2021 contains 1251 scans with T1, T1Gd, T2 and FLAIR modalities; the study used 834 training, 208 validation and 209 test scans\.
  陈述 `claim_6c8d3b750964153c0bf384eb21bf5848`；证据 ev_9bf378bc6cff5581822a3624, ev_b961e1894536a4b5e0f604c8
- **数据 / supported**：The Changhai Hospital dataset includes 139 patients with serous or mucinous cystic neoplasms, arterial, portal venous and delayed phases, and radiologist annotations of pancreas and tumor regions; the three phases were registered to the venous phase using Elastix\.
  陈述 `claim_1335d432ffbddcb0a5bd306f98308fac`；证据 ev_9bf378bc6cff5581822a3624
- **方法 / supported**：Training used a learning rate of 1e-4 for 200 epochs, initial single-modality training followed by multi-modality training, and 128 × 128 × 128 crops\. Evaluation used sliding windows with 0\.6 overlap, Dice and HD95\.
  陈述 `claim_b0fcdb659e358bc6a087bf9acabcd8e3`；证据 ev_5e7b7cc5dbdaee432b2d9b07
- **结果 / supported**：Models used the same dataset split, were selected using validation performance, and were evaluated without post-processing\. REDNet reportedly outperformed other compared models on the in-house dataset, while slightly trailing the brain-tumor-focused CKD method in some BraTS measures\.
  陈述 `claim_d29db3a3fa6d4cd6d0e3e76b6174eb44`；证据 ev_3178400c310350b736541f7f
- **结果 / supported**：In the reported Gaussian-noise experiments on both datasets, REDNet maintained the highest Dice and lowest HD95 among compared methods under noisy conditions; tested noise magnitudes were σ = \{0, 0\.1, 0\.3, 0\.5, 1, 1\.5, 2, 2\.5, 3, 3\.5\}\.
  陈述 `claim_fe9a0541cc7cab838c4e4220eab55a7c`；证据 ev_a53b45d0c04a3adad69a78a4
- **结果 / supported**：On BraTS with Rayleigh noise at scale 2 and synthetic intensity non-uniformity at magnitude 50, REDNet reportedly outperformed eight compared models across the standard BraTS evaluation metrics\.
  陈述 `claim_f4dfb790b38bc683855b722ac932c4d0`；证据 ev_90f02ad2c09d9bf123d4117b
- **结果 / supported**：Under N-T1Gd corruption, the ICEM ablation model \(Model3\) achieved ET, TC and WT scores of 0\.6955, 0\.8107 and 0\.9197, compared with reported baseline scores of 0\.6760, 0\.8027 and 0\.9099\.
  陈述 `claim_3431a4bc49d6f2425fa9063cd3e8cc86`；证据 ev_2c01c38963727a0f567d80ad, ev_627f9d3030bc92ec12a85ea4
- **结果 / supported**：The CDAM ablation model \(Model4\) achieved ET, TC and WT scores of 0\.8521, 0\.8961 and 0\.9337, exceeding Models1–3 in the reported comparison, but slightly underperformed Model3 under N-T1Gd corruption\.
  陈述 `claim_bdd1dcab0cc99b6154f3b12899c261cd`；证据 ev_80522cc9e9280bc39d3ab103
- **局限 / supported**：For test images from uCT 960\+ and iCT 256 scanners that differed from the training distribution, segmentation quality decreased, although the reported results remained anatomically consistent\.
  陈述 `claim_c5b623f96d0e2cc6933974b1e9ebc86d`；证据 ev_7b7d8f08e9051a221499e021
- **局限 / supported**：Always-on discounting reduces belief masses even for perfectly reliable inputs, slightly diminishing usable information from high-quality sources despite discount factors close to 1\.
  陈述 `claim_866d4d5a9f197b00452600fd680069b3`；证据 ev_fd611988bcb7f714bc563b1d
- **作者贡献陈述 / supported**：The study contributes a framework that dynamically calibrates modality influence at pixel level using ICEM, CDAM and discounting fusion, evaluated against conventional and uncertainty-based methods on BraTS2021 and an in-house pancreas dataset\.
  陈述 `claim_b42d5b4618697bb3df31b59f0d566ca2`；证据 ev_a4ad9c0b3d6627f5baf01d89

## Learned iterative segmentation of highly variable anatomy from limited data - Applications to whole heart segmentation for congenital heart disease

文档：doc_a3ae9432211ef4069f9ba3f0；后端：codex

- **研究问题 / supported**：Whole heart segmentation in congenital heart disease is challenged by extreme anatomical variability, limited labeled data, low cardiac MR signal-to-noise ratio and resolution, weak boundaries between structures, and stent artifacts\.
  陈述 `claim_02b7fb2d0fa08ca8f3964e5a12a986ee`；证据 ev_0912e215a3f00194456bb922, ev_f3749a082723dee8ecbae06a
- **方法 / supported**：The task labels LV, RV, LA including pulmonary veins, RA, AO, PA, SVC and IVC from a single-phase 3D cardiac MR scan and one click per structure; myocardium segmentation and time-series imaging are outside the stated focus\.
  陈述 `claim_04319f58ce6cc59dec3262c930f0e3dc`；证据 ev_20d0f408ef59408b3236aadf
- **方法 / supported**：The recurrent model uses a modified 3D U-Net to jointly predict the next segmentation and a stopping indicator, feeding the predicted segmentation into the next iteration\. Users can override automatic stopping by selecting an earlier segmentation or requesting more iterations\.
  陈述 `claim_e969435660f2340dea9efd837a39695b`；证据 ev_a3c2bd68ce63175435d03ae0
- **方法 / supported**：A teacher-forcing sequence likelihood loss decomposes into segmentation and stopping losses over decoupled time steps, eliminating backpropagation through time\. Consecutive partial-segmentation training pairs are generated on the fly from complete ground truth\.
  陈述 `claim_4d708cd0b157cbbd629cc61e2e09a9db`；证据 ev_6aaf261a29fc95efbad3fa06, ev_fca92bd44623370062bf401b
- **方法 / supported**：Training targets prescribe constant-rate growth along vessel centerlines and outward chamber growth using concentric spheres intersected with ground truth\. Input segmentations are corrupted with nonrigid deformations and foreground blobs while output targets remain unchanged\.
  陈述 `claim_f5f9197cc019dbd92bb85ec8976d9f7f`；证据 ev_1406053f15f0edbe2cf6ce14, ev_59e1ce0dc6b2503979079696, ev_dce5a7458bbbb9dca75bcc00
- **数据 / supported**：The dataset contains 60 cardiac MR images with manual labels for eight structures: 20 HVSMR scans and 40 retrospectively retrieved clinical scans\. All were acquired at Boston Children’s Hospital on a 1\.5T Philips Achieva scanner\.
  陈述 `claim_ec6ff1899a17a2dc9941448282dcd2de`；证据 ev_95e59bfbd2f76597e5ae295f
- **数据 / supported**：Subjects ranged from younger than 1 to 55 years\. Average image resolution was 0\.9 × 0\.9 × 0\.85 mm; images were cropped around the heart and resized to 126 × 180 × 144\. Severity categories described anatomical malformations, not clinical prognosis\.
  陈述 `claim_446c20dbaa46d91dfa2ba668f0320a4f`；证据 ev_383ff5709e13506c413f2e52, ev_6f4feb67c90fe6f453ac35cf
- **方法 / supported**：Experiments trained on 20 images and tested on 40, or trained on 48 images and tested on 12\. Training used four-fold cross-validation, and the four resulting networks were ensembled for test predictions\.
  陈述 `claim_2f3d729650dace72a10f8f57e0cc1ffd`；证据 ev_1028a99a3a9972082e1c5a5e, ev_3ab9b90fa9aeb2680cbb57e7, ev_8f87ac8e68ddeadd4fc012ec
- **方法 / supported**：User interaction was simulated\. Iter–U selected the best segmentation among the first 40 iterations, while Iter–U–All selected the iteration with the best mean accuracy across all eight structures\.
  陈述 `claim_d94537bbb106ce6e0eb275b3e11cedd0`；证据 ev_1028a99a3a9972082e1c5a5e, ev_161d9a0150fe3325f056a73c, ev_3ea094b3134e56dcc15ec1aa
- **结果 / supported**：For severe subjects with 20–HVSMR training, mean Dice scores across eight structures were 79\.0 ± 17\.2 for Iter–A, 85\.3 ± 12\.8 for Iter–U, 68\.2 ± 27\.8 for U–Net\+S and 64\.6 ± 31\.9 for U–Net–All\. Iter–A comparisons yielded p &lt; 10^-11 versus U–Net\+S and p &lt; 10^-12 versus U–Net–All; both Iter–U comparisons yielded p &lt; 10^-22\.
  陈述 `claim_922764a5dc06433edb9ffb5a8602d91f`；证据 ev_3ab9b90fa9aeb2680cbb57e7, ev_b76df7f900839d3d1a08f2cd
- **结果 / supported**：For severe subjects with 48–HVSMR\+ training, mean Dice scores were 85\.1 ± 15\.6 for Iter–A, 88\.2 ± 14\.0 for Iter–U, 82\.0 ± 19\.7 for U–Net\+S and 84\.9 ± 18\.0 for U–Net–All\. Iter–U significantly exceeded U–Net\+S at p &lt; 10^-2\.
  陈述 `claim_f62236f36219afda00a8f13f0b5b0ee1`；证据 ev_3ab9b90fa9aeb2680cbb57e7, ev_b76df7f900839d3d1a08f2cd
- **结果 / partial**：All seven methods performed well in mild/moderate subjects\. Multiclass iterative models significantly improved over all three baselines with 20–HVSMR training, but no statistically significant overall improvements over the baselines were found with 48–HVSMR\+ training\.
  陈述 `claim_d3663622dbdaf8c0543608f89bb4f986`；证据 ev_359a3c37019bc185bcf06a56, ev_7600a6d300e0051166f117e2, ev_d15b532e59ea7338b1b381ef
- **结果 / supported**：Binary iterative inference required 0\.65 ± 0\.15 seconds per iteration on an NVIDIA TITAN X GPU, with mean structure segmentation times ranging from 2 to 18 seconds and approximately 1 minute for the whole heart\.
  陈述 `claim_a1737ecabcd3a7552201379385a0ab78`；证据 ev_86dfd590543ed0c4d693657b
- **方法 / supported**：Dice evaluation excluded optional vessel regions from both predicted and ground truth segmentations, comparing only required regions\.
  陈述 `claim_6a6adfe55bfbf980d0b97fc3795e14dd`；证据 ev_7600a6d300e0051166f117e2, ev_86dfd590543ed0c4d693657b
- **局限 / supported**：Failures included chamber growth through septal defects or valves and failure to grow through narrow or artifact-affected main pulmonary artery trunks\. PA segmentation remained difficult generally, and some other structures remained difficult for particular pathologies\.
  陈述 `claim_8e501e1892c18f150a876de5f33c38f3`；证据 ev_2565aca0bbee79e931d02dfa, ev_97ec38b059f84954aca4ff13
- **局限 / supported**：Multiclass iterative models generally performed worse than binary models and gained less from simulated user stopping\. A possible explanation is that a single multiclass iteration may not simultaneously provide the best segmentation of every structure; multiclass procedures were inherited from tuning for binary segmentation\.
  陈述 `claim_7e803bab7b1e577da5274f5df38de1d8`；证据 ev_2565aca0bbee79e931d02dfa, ev_ef4142c372c4b249f496a8c8
- **局限 / supported**：Actual-user accuracy and the hypothesis of faster, more intuitive interaction remain to be assessed in a user study; the reported experiments used simulated users\.
  陈述 `claim_ab97ee1e6157858c8a27075f5d2e960e`；证据 ev_af60aa1becfed13468542610, ev_ef4142c372c4b249f496a8c8
- **作者贡献陈述 / supported**：The authors claim, to the best of their knowledge, the first whole heart segmentation that individually labels cardiac chambers and great vessels in cardiac MR for congenital heart disease patients\.
  陈述 `claim_34ee92d0f45940016a468c7c433dad35`；证据 ev_2bb9b1cb338e7f93a68e282b
- **作者贡献陈述 / supported**：The methodological contribution is an iterative segmentation framework with prescribed growth dynamics and a sequence loss using teacher forcing, enabling automatic stopping and optional selection among evolving segmentation candidates\.
  陈述 `claim_e8e07adc8291d6e168ec3f6b72c46053`；证据 ev_1e42d2f7f85657338ef4c018, ev_2bb9b1cb338e7f93a68e282b, ev_fca92bd44623370062bf401b

## DeferredSeg: A multi-expert deferral framework for medical image segmentation

文档：doc_3e100deb0df450ee9d65bf41；后端：codex

- **研究问题 / supported**：Most existing learning-to-defer methods make instance-level routing decisions, whereas segmentation requires spatially varying decisions near ambiguous boundaries, low-contrast tissues and small lesions\.
  陈述 `claim_abda0cbf13406d1b184c33ec6e29eb9c`；证据 ev_fb0d61a7a7c7a881ea2c1e5b
- **方法 / supported**：DeferredSeg adds J\+1 pixel-wise routing channels for a base segmentor and J experts while retaining the original segmentation output channels\.
  陈述 `claim_936e402162191b5aea0a63327723c70b`；证据 ev_4a7114cc7ab798a623c1a56d
- **方法 / supported**：The Deferral Collaboration loss rewards correct expert predictions, suppresses incorrect expert routing, rewards the model branch when its thresholded prediction is correct, and supervises segmentation probabilities with binary cross-entropy\.
  陈述 `claim_356a9773e059f667b21159e3585fabb5`；证据 ev_4b64f38b0995b34258c342a4
- **方法 / supported**：Training combines collaboration loss, spatial-coherence loss and a load-balancing penalty; the latter penalizes batch-average soft expert routing probabilities outside configurable lower and upper bounds\.
  陈述 `claim_e0edd82803a5441a6872b3a9c9222151`；证据 ev_3f67c57af8cc016a40bbdcbf
- **方法 / supported**：At inference, each pixel is assigned to the branch with the largest routing logit\. Model-routed pixels use the base prediction thresholded at 0\.5; expert-routed pixels use the selected expert's prediction\.
  陈述 `claim_b4f72f96263f5c0a92edc0058fa438fb`；证据 ev_3a6f1cb8eb8615abccda4c09, ev_540c2cddf9c91ae880f52164
- **方法 / supported**：The auxiliary CNN aggregated deferral predictor supports spatial-coherence training; at inference it is used only for visualization and does not determine hard routing or final fusion\.
  陈述 `claim_98ea2fbfff9ac0fbcf6c08b7fc9cba8d`；证据 ev_1f2a3641998b100cbc978983
- **数据 / supported**：Evaluation uses synthetic expert masks on PROMISE12, LiTS and AMOS22, and pre-existing clinician annotations as offline expert branches on Chaksu, QUBIQ and CURVAS; metrics are DSC, Jaccard and Sensitivity\.
  陈述 `claim_3447d492b730483ec07bf2368c2bd939`；证据 ev_ac7f6914676a81a3876508c5
- **数据 / supported**：Synthetic masks and clinician annotations are proxies for prospective expert corrections\. They provide predictions for selected pixels and are not routing inputs or determinants of where deferral occurs\.
  陈述 `claim_bb215d32fd889132e0c74988637c239b`；证据 ev_3277af82a0c4a13298ab2eb7, ev_fb67c77f09275e9d9184bd14
- **数据 / supported**：Comparative synthetic experts have foreground/background/boundary accuracy profiles of 92/98/98%, 85/99/94% and 75/95/90%, respectively\.
  陈述 `claim_bb874a95c59f9b539ca1ad817b645811`；证据 ev_9192097bb19f87b69945485c
- **数据 / supported**：QUBIQ reference masks use STAPLE fusion of annotations disjoint from the expert branches; Chaksu and CURVAS consensus references derive from the same annotation pool used for expert branches\.
  陈述 `claim_d1ff75d54ba5ad2f6aad14c30228e9eb`；证据 ev_5d9a1bb3436c81228fca0852
- **方法 / supported**：PatchCorr uses the same expert source as one-expert DeferredSeg, with its correction budget matched to DeferredSeg's one-expert deferral ratio on each dataset\.
  陈述 `claim_db3714f38a7ec2431c4c952a5f993b07`；证据 ev_028c4b034fb4e7fa9a28c4c9
- **结果 / supported**：On synthetic-expert PROMISE12, System DSC is 97\.06±0\.05% with one expert and 96\.00±0\.14% with three experts, compared with 86\.17±1\.53% for fine-tuned MedSAM and 92\.24±0\.03% for PatchCorr\.
  陈述 `claim_f77eb863c77077f892ea6c49fe55301b`；证据 ev_d92c80d33d9218452fd4c377
- **结果 / supported**：PROMISE12 deferral ratios are 4\.32% with one synthetic expert and 10\.28% with three; the corresponding model-retained proportions are 95\.68% and 89\.72%\.
  陈述 `claim_6a699af9b2cdf15516447ead48dac6cd`；证据 ev_cce7e8e1462c854f5127d4e3
- **结果 / supported**：In offline QUBIQ brain-growth evaluation, System DSC increases from 83\.15% for fine-tuned MedSAM to 86\.68% with one expert and 89\.33% with three experts\.
  陈述 `claim_b4b1a1b4e2cee9965191e14608e8cfd0`；证据 ev_696d3a4947114a7503668d33
- **结果 / supported**：On offline QUBIQ prostate evaluation, one- and three-expert System DSC values are 96\.73% and 95\.71%, versus 94\.77% for fine-tuned MedSAM; Sensitivity decreases from 96\.20% to 93\.69% and 92\.76%, respectively\.
  陈述 `claim_7058339408f9b2d7fb04dc943f52ea32`；证据 ev_301c138c2f726903f2749ae2, ev_696d3a4947114a7503668d33
- **结果 / supported**：One-sided paired Wilcoxon tests find significant case-level DSC improvements over fine-tuned MedSAM in 15/16 comparisons for each expert configuration, over PatchCorr in 16/16 for each configuration, and over CENet-FT in all six CENet tasks\.
  陈述 `claim_71a36c34568d9395f4c67b7bfddf195e`；证据 ev_78d154815ad18aad2300dc4c, ev_d2257091cfd6a118c537dd88
- **结果 / supported**：The non-significant comparisons against fine-tuned MedSAM are AMOS22 liver with one expert and LiTS liver with three experts\.
  陈述 `claim_3633c767b34013144e4487d5c52249a8`；证据 ev_68448704556a74a4a4984e02, ev_8ae5c8bcc331258888807ab1
- **结果 / supported**：In the PROMISE12 three-expert ablation, DC alone yields 95\.39±1\.64% System DSC, DC\+SC yields 95\.79±1\.71%, DC\+LB yields 96\.59±1\.46%, and all three losses yield 96\.58±1\.04%\.
  陈述 `claim_508d8307a66efbc9e0483c0b990e1c97`；证据 ev_d2257091cfd6a118c537dd88
- **局限 / supported**：Adding experts does not guarantee better performance: benefits depend on regional complementarity and the target task\.
  陈述 `claim_459e43b608da7a9709d1030b11cecfb0`；证据 ev_6da0f8d1ffe0e2d71fedb829, ev_fa6a9fb4c04af74625d8696c
- **局限 / supported**：Training still requires previously annotated expert data, and the real-expert evaluation is offline; broader prospective studies are needed to assess clinical generality and interaction efficiency\.
  陈述 `claim_94abb84ae78c65275b001f0ff6570058`；证据 ev_010c975d9b832298a35913c2
- **局限 / supported**：A proof-of-concept Streamlit interface demonstrates expert-region assignment, local correction and fusion, but usability, annotation time, interaction cost and clinical efficiency remain unevaluated in prospective clinician studies\.
  陈述 `claim_448a887c0708b032a6ecf5cfc6c849f8`；证据 ev_4f8cad3598c5597885133f37
- **作者贡献陈述 / supported**：The contribution is a pixel-wise medical segmentation deferral framework with collaboration and spatial-coherence losses, multi-expert load balancing, and instantiations on MedSAM and CENet\.
  陈述 `claim_a054c9ce6d2e4b0493c382ea4417fe14`；证据 ev_081f5659c894c1d4ab71e1e1, ev_de15bd845d160fa8fbcbc2c4

## DeferredSeg:A Multi-Expert Deferral Framework for Medical Image Segmentation

尚无当前文档版本的阅读卡片。

## From Few-Shot Segmentation to Clinician-in-the-Loop Medical Image Analysis

尚无当前文档版本的阅读卡片。

## SDF4CHD - Generative modeling of cardiac anatomies with congenital heart defects

文档：doc_9305e394275645706ffe08d3；后端：codex

- **研究问题 / supported**：Rare CHD types make large training cohorts difficult to acquire, while prior cardiac generative approaches primarily model normal topology and do not readily capture CHD topological variations\.
  陈述 `claim_599a9d572d229d7c08601afda8a29afc`；证据 ev_9bc45964f001705008da9241, ev_a9da8206185605d565e3be4f
- **方法 / supported**：SDF4CHD disentangles CHD type and patient-specific shape using diagnosis-conditioned implicit anatomical templates, a deformation module, and a correction module for additional patient-specific topological variations\.
  陈述 `claim_bcf2215df4f1a2a4185a66c96ae215ac`；证据 ev_46b98c3024aa55be0d75f321, ev_85c702a343b7fdc1c50699ab
- **方法 / supported**：The shape module uses neural ordinary differential equations to learn invertible deformations, with a spatial latent code of 4×4×4×27\.
  陈述 `claim_4e3f6cf95980d509186e49c11eea9d53`；证据 ev_35aabb921bfa6504acf0dde3
- **方法 / supported**：In practice, training used occupancy values rather than signed distances; anatomical boundaries were represented by the 0\.5-level set\.
  陈述 `claim_832784e3f96208365171db5125509aee`；证据 ev_11cbe3fdd8494f689e9944b5
- **数据 / supported**：The study used imageCHD CT data and diagnosis labels with segmentations of seven cardiac structures, manually processed under radiology expert supervision; the reported splits contained 67 training, 8 validation, and 34 testing patients\.
  陈述 `claim_563d8bd09c9e6445c476940654ef204d`；证据 ev_64378ab50821c47a5ba1270b, ev_f56b267e1c8381b7a6defdd0
- **数据 / supported**：The six modeled CHD types were ASD, VSD, DORV, pulmonary atresia, tetralogy of Fallot, and d-transposition of the great arteries\.
  陈述 `claim_57209670bef7b40cab34ade5339f2f89`；证据 ev_f56b267e1c8381b7a6defdd0
- **结果 / supported**：In blinded manual assessment of 50 randomly generated cases per method, input diagnoses disagreed with diagnoses inferred from generated anatomy in 3 SDF4CHD cases and 48 CDeepSDF cases; most SDF4CHD discrepancies involved distinguishing DORV from ToF\.
  陈述 `claim_9ca7dace0fb5a47057669552fae69267`；证据 ev_84ee4319a5364ee140ac9943, ev_bcbd0d39482b1e1d7457e281
- **结果 / supported**：Across 30 generated anatomies per CHD type, SDF4CHD had higher mean shape plausibility scores than separate networks trained per type; plausibility was measured as maximum Dice similarity to training anatomies\.
  陈述 `claim_b0472f4bfacf8920c8a163753862e9e1`；证据 ev_eb139e29dee79caf1c7152ba
- **结果 / supported**：For unseen anatomy reconstruction, whole-heart Dice was 86\.3 ± 2\.7 without correction and 88\.8 ± 2\.0 with correction; CDeepSDF and DeepSDF each achieved 88\.8 ± 2\.7\.
  陈述 `claim_f5239625c7243996725b2dc8046e2ae0`；证据 ev_63414c4e3d43bc39a11be278, ev_84389058b399b8ab8b752658
- **结果 / supported**：Adding 420 synthetic images to 67 real training images improved test whole-heart Dice from 84\.7 ± 6\.0 to 86\.3 ± 5\.3 for 3D UNet and from 82\.1 ± 7\.1 to 84\.0 ± 6\.3 for 2D UNet; both improvements were reported as significant\.
  陈述 `claim_6e8d4a931e32b45c29d00b38466f8285`；证据 ev_468c6f16ed86a8dbdeef6fda, ev_b29f658f0c7b807897ae56aa
- **结果 / supported**：With synthetic augmentation, 3D UNet median VSD diameter percentage error decreased from 54% to 38%, and median area percentage error decreased from 72% to 59%\.
  陈述 `claim_d9e8e6dff94c12167295195bd89cc7ea`；证据 ev_85fd761a46d0e5563ebb19dd
- **局限 / supported**：The method failed to reconstruct an apical VSD in a test case with two VSDs because its learned templates contained only subaortic VSDs; the correction network also missed the apical defect\.
  陈述 `claim_04cd50911e3dd064533e288ad1a36bfc`；证据 ev_e68512c7b73bda2d4b82ee1c
- **局限 / supported**：The shared Gaussian shape prior across CHD types is a simplifying assumption; patient-specific shape distributions may differ by type and are difficult to capture with few examples\.
  陈述 `claim_84390b61ba7d6dcc87316cd1b1dc5aff`；证据 ev_e88ad8c1b19a00be4c379f44
- **局限 / supported**：Higher-resolution latent shape codes improved reconstruction but produced non-physiological distortions during synthetic generation\.
  陈述 `claim_51c10988f3cf229ac8bbefe6663ba792`；证据 ev_3707576fddc6a7ae1b5e019f
- **局限 / supported**：The predicted diffeomorphic mapping does not guarantee strictly positive mesh-element Jacobians, despite reducing surface-intersection artifacts\.
  陈述 `claim_99732cb1a06cfe06a84b4b37a4e8745a`；证据 ev_e9ba442b3d854d3ec3ff61aa
- **局限 / supported**：SDF4CHD and the baselines struggled to represent smaller vessel branches; generative modeling of vessel-connection variations was not extensively explored\.
  陈述 `claim_60a8b49401240082f7a32ffce198362f`；证据 ev_44a6228d05888ee4fba56c64, ev_8505ca9805b8e541c6c34c6b
- **作者贡献陈述 / supported**：The contribution is a type-controlled cardiac generative representation combining smooth implicit fields and diffeomorphic deformations, with demonstrated applications to synthetic image-segmentation pairs and computational-simulation meshes\.
  陈述 `claim_b1b6c06442c2549eeaf82c6ae6d25967`；证据 ev_21ef63af3042c61e19c20a9b, ev_9e9ccc31a264d6c4f682f15a
