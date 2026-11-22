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

模板采用已验收单片配方：wait_seconds=120、worker_timeout_seconds=360；运行时将总体deadline设为实际启动时间加600秒。--deadline是Unix秒时间戳，已经结束的批次不能沿用。运行前为每批明确目录、产物范围和截止，运行后由调用者保存正式回执并清理副产物。

当前已通过完整图像权重加载、单片一次真实CUDA BF16前向、有限值检查、零实例空间尺寸规范化及结果保存。实际零实例掩膜由(0,288,288)还原为(0,512,512)，未产生阳性像素；覆盖范围只含指定切片。非空实例的体素空间对应、全体积效果和临床表现尚未验证。源码保留在独立分支供审核。

## 手动迁移所需材料

源码沿项目既有Git固定版本更新流程取得审核发布后的提交，当前独立分支仅在本地，服务器更新须等待发布指示。后续合并以届时最新main为基础，保留助手1已发布的公共资源修复；本地11文件只说明本次验收基线，不能覆盖服务器更新版本。服务器继续使用已有有效Git工作树及run.py/update.py，无需重新手动搬运整个工程。本轮没有推送、生成迁移包或登录服务器。

模型材料仅需固定版本目录中的10文件：model.safetensors、config.json、processor_config.json、merges.txt、special_tokens_map.json、tokenizer.json、tokenizer_config.json、vocab.json、LICENSE、README.md，总3,445,029,864字节。目标机已有同版本完整材料时直接引用；缺少时按下文选择人工传输或官方固定版本下载。

环境使用目标机兼容的Python、PyTorch/CUDA、torchvision及工程pyproject.toml依赖；SAM3新增依赖的固定清单见materials.py的PACKAGES。已验收版本为Python3.11、torch2.12.1+cu126、torchvision0.27.1+cu126、Transformers5.24.0。Windows隔离目录含Windows二进制扩展，不直接复制到Linux；目标机已有依赖直接复用，缺项安装需单独获准，Linux运行尚未验收。

以下为用户在服务器目标Python环境中手动执行的检查及缺项获取入口，使用服务器现有正常网络，无代理参数。本轮未执行这些命令。先检查已有版本，满足时直接复用；安装命令只指定SAM3附加包，不包含PyTorch/CUDA安装：

```bash
python -B -c "import sys, torch, torchvision, importlib.metadata as m; print(sys.version); print(torch.__version__, torchvision.__version__); print({n:m.version(n) for n in ['transformers','modelscope','tokenizers','safetensors']})"
python -m pip install --upgrade-strategy only-if-needed modelscope==1.40.3 modelscope-hub==0.7.1 transformers==5.24.0 regex==2026.11.19 tokenizers==0.23.3 typer==0.29.2 cryptography==50.0.2 cffi==2.1.1 pycparser==3.12 shellingham==1.5.4 annotated-doc==0.0.7
```

权重可复用目标机同版本目录，也可手动传输本机已批准目录；这通常是唯一需要人工传输的大文件材料。用户选择服务器自行获取时，先将SAM3_WEIGHTS设为目标长期目录，再使用官方固定版本命令；已存在完整材料时不执行下载：

```bash
modelscope download --model facebook/sam3 model.safetensors config.json processor_config.json merges.txt special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json LICENSE README.md --revision 96f3e1b404ba14f2cfac60ee6ae87c269a7b7923 --local_dir "$SAM3_WEIGHTS" --max-workers 2
```

私有配置优先复用服务器已有manifest/cohort及CT路径绑定，只补目标机缺少的映射和配置字段，保留作者间距、RAS研究约定和强度证据。必要私有映射按私有通道手动传递，源码提交不含病例、权重或私有配置。已有原始CT、运行环境及数据继续引用原位置；本地测试输出、缓存、日志、医学LoRA和Windows虚拟环境不属于迁移材料。
