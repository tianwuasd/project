# SAM3 原模型有限切片检查

入口为 `run.py models sam3 inspect|predict`。使用 ModelScope `facebook/sam3` 固定 revision `96f3e1b404ba14f2cfac60ee6ae87c269a7b7923`，Transformers 5.24.0 的 Sam3Model/Sam3Processor，直接读取 model.safetensors，不需要转换权重。SAM License 随官方材料保存。

配置模板为 `configs/models/sam3.yaml`。将本地模型、独立依赖、CHD1282 manifest/cohort、实际数据根、几何与强度证据填入私有配置。解释器复用已有真实 PyTorch/CUDA 环境，新增依赖从显式 target 目录导入；不修改共享环境。材料工具 `src/chd_segmentation/models/sam3/materials.py` 使用官方 SDK，固定一套十文件和十一依赖，监督时限和产物范围。

```powershell
python run.py models sam3 inspect --config <PRIVATE_CONFIG> --case <CASE_ID> --slice 53
python run.py models sam3 predict --config <PRIVATE_CONFIG> --case <CASE_ID> --slice 53 --output <WORK_ROOT>/prediction --deadline <UNIX_SECONDS>
```

inspect 只检查已登记病例头、材料和索引，不读取影像体素。predict 每次只接受一个登记病例的一张切片，batch 为 1，使用一次预先配置的文本提示；训练更新次数为 0。仅支持明确的 CUDA BF16 配方；资源不足、依赖缺失、权重键或形状不符直接终止。

输入几何复用 CHD1282 的共享数组索引及逐例作者 Excel 间距，保留原始 affine 与研究 affine；RAS 研究约定不代表恢复扫描原点。轴向 plane=image[:,:,k].T，模型行 j、列 i；官方 processor 恢复原始切片尺寸后，掩膜转为 instance,i,j 保存。输出压缩数组仅含选定切片及实例，不生成未处理区域的背景体积。summary.json 的 coverage 显式记录切片索引与原形状。

固定 HU 窗按配置映射至 uint8 并复制为 RGB；已有窗化输入或撤销标准化需要明确语义和参数。提示在读取数据前配置，不读取标签生成文本。没有目标实例时如实记录零实例；本入口不验证临床效果。

公共 ResourceSession 负责准入、CPU/内存额度和任务登记，单独子进程加载官方模型。RAM 硬上限为 8 GiB，仍受整机 80% 和项目限制；模板保持原工程 4 GiB 显存预留。个别局部检查如已获明确许可，可在私有配置指定其他预留，最低 1 GiB。实际模型显存额度取 min(配置上限, 当时空闲−1 GiB)，配置上限最多 7 GiB；运行中继续检查真实空闲量。原模型和生产默认配置保持原样。

模型加载与推断最长600秒，准入等待最多180秒，且均服从命令传入的15分钟总体截止，预留120秒诊断清理。官方同步加载由HF_DEACTIVATE_ASYNC_LOAD=1启用，使用Windows原生pread及BF16，图像模型保持1008输入。原始工作集默认3.5GiB，使用公共reserve_margin分配额外缓冲。父进程通过公共ResourceSession登记worker，worker通过共享handle的SessionGuard.check_active检查取消与资源压力；退出时先取消，再有界终止并join，仍存活时拒绝后续文件清理。副产物限于显式工作根，由调用者在进程退出、记录实际结果和核对归属后清理；长期模型及依赖只读复用。不要把资料获取或接口导入通过当作真实前向通过。Windows运行结果不代表Linux/NCCL已验证。

当前已通过完整图像权重加载、单片一次真实CUDA BF16前向、有限值检查、零实例空间尺寸规范化及结果保存。实际零实例掩膜由(0,288,288)还原为(0,512,512)，未产生阳性像素；覆盖范围只含指定切片。非空实例的体素空间对应、全体积效果和临床表现尚未验证。源码保留在独立分支供审核。
