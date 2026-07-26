# Vast.ai RTX PRO 6000 接入与实验工作流

最后更新：2026-07-13T08:58:42-07:00

## 1. 执行结论

推荐采用：

```text
本地工作站：代码、Git、文档、单元测试、trace 分析和实验编排
                 |
                 | 固定 commit / 固定镜像 / 只上传必要输入
                 v
Vast.ai RTX PRO 6000：SGLang GPU 集成、7B/8B、长上下文、HiCache 和性能实验
                 |
                 | 下载 compact CSV/JSON/log，随后销毁实例
                 v
本地工作站：结果分析、图表、提交和长期保存
```

不建议完全迁移到 Vast.ai，也不建议继续只依赖本地 RTX 2080 SUPER。

- 本地 8GB SM75 仍适合快速开发、correctness smoke test 和小模型回归。
- RTX PRO 6000 S 的 96GB VRAM 能解锁当前研究真正需要的 7B/8B BF16、长上下文、大 KV working set 和 CPU/GPU tier 实验。
- Vast.ai 是异构 marketplace，host、CPU/RAM、磁盘、网络和 PCIe 条件会变化；它适合作为按需实验执行面，不适合作为唯一事实来源或长期开发环境。

## 2. Vast.ai 是怎么 host 的

Vast.ai 的标准实例是运行在第三方 provider 机器上的 Linux Docker container：

- GPU 在实例运行期间独占，不与其他租户同时共享。
- CPU 和系统 RAM 通常按租用 GPU 占整机 GPU 的比例分配。
- container disk 在创建时固定，之后不能扩容。
- `/dev/shm` 由平台按 GPU 份额自动分配。
- 实例通常没有独立公网 IP；内部端口映射到共享公网 IP 的随机外部端口。
- 标准 Docker instance 不支持 Docker-in-Docker。
- provider 在技术上可能访问其机器上的文件；敏感工作应选择 Secure Cloud，并避免放入长期凭据。

Vast.ai 提供三种启动模式：

| 模式 | 行为 | 本项目用途 |
| --- | --- | --- |
| `Entrypoint` | 按镜像自身 `ENTRYPOINT`/参数运行，不注入 SSH/Jupyter | 固定镜像、无人值守 benchmark |
| `SSH` | 平台覆盖镜像 entrypoint，注入 SSH；可通过 on-start 启动服务 | 首轮开发、调试和 smoke test |
| `Jupyter` | 平台覆盖 entrypoint，注入 Jupyter 与 SSH | 本项目不是首选 |

模板本质上是 `docker run` 的平台包装。镜像可来自 Docker Hub、GHCR 或其他 registry；私有镜像需要单独配置 registry 凭据。

## 3. 现有 Docker 能否直接运行

### 3.1 简短答案

核心软件可以复用，但现有本地启动方式不能原封不动搬过去。

1. **镜像层面：可以。**
   当前本机 `lmsysorg/sglang:dev` 已验证包含：
   - CUDA `12.9.1`；
   - PyTorch `2.9.1+cu129`；
   - `sgl-kernel 0.3.21`；
   - FlashInfer `0.6.4`；
   - Triton `3.5.1`；
   - 编译架构包含 `sm_120` 和 `compute_120`；
   - `/usr/sbin/sshd`。

2. **源码层面：可以。**
   `sglang-running` 当前源码明确包含：
   - CUDA `>=12.8` 的 Blackwell/SM120 检测；
   - RTX PRO 6000 的 Triton shared-memory 特殊配置；
   - SM120 的 FP8/MXFP4、attention 和 kernel 路径。

3. **宿主启动脚本层面：不能直接复用。**
   `scripts/run_qwen3_0_6b_docker.sh` 自己调用：
   - `docker run`；
   - `--runtime=nvidia`；
   - `--gpus all`；
   - `--ipc=host`；
   - host bind mount。

   Vast.ai 的实例本身已经是 Docker container，且不支持 Docker-in-Docker。因此该脚本不能在实例内再次执行；需要把模型命令转换为 Vast template 的 image、launch mode、port、environment 和 on-start 配置。

### 3.2 不需要移植的部分

- 不需要把 SGLang 改写成另一种部署格式。
- 不需要为 Vast.ai 单独维护一套 Python 实现。
- 不需要保留本地 SM75 的低显存限制。
- GPU runtime、设备分配和 shared memory 由 Vast.ai 外层平台负责。

### 3.3 需要打包或调整的部分

#### 首轮 smoke test

可直接选择官方 `lmsysorg/sglang:dev`，使用 SSH 模式验证 RTX PRO 6000。

`dev` 是 mutable tag，只适合兼容性验证。运行时必须记录：

- 实际 Docker digest；
- SGLang commit/version；
- CUDA、PyTorch、FlashInfer、Triton 和 driver；
- GPU 精确名称与 compute capability。

#### 正式 prototype

正式实验应建立与 Git commit 一一对应的自定义镜像：

```text
ghcr.io/<owner>/sglang-kvcache:<git-sha>-cu129
```

镜像应在本地、CI 或其他可运行 Docker build 的环境构建后推送到 registry，再由 Vast.ai 拉取。不要尝试在标准 Vast Docker instance 内执行 Docker build。

推荐把现有 build 默认值从 CUDA `12.8.1` 提升到 `12.9.1` 或当前 SGLang 默认的 CUDA 13 路线。CUDA 12.8 是 SM120 的最低支持线，不是最稳妥的长期基线。

#### 当前 DeepEP 风险

现有 `docker/Dockerfile` 的 DeepEP 编译 arch list 对 CUDA 12.9/13 包含：

```text
9.0;10.0;10.3
```

但没有 `12.0`。因此：

- 单 GPU dense Qwen 7B/8B、KVCOMM、KVFlow 和 HiCache 实验不依赖 DeepEP，可先运行。
- 在 RTX PRO 6000 上启用 DeepEP/MoE 路径前，必须补充并验证 SM120 编译与运行支持。

### 3.4 Vast 模板与本地 `docker run` 的对应关系

| 本地配置 | Vast.ai 对应方式 |
| --- | --- |
| `--runtime=nvidia --gpus all` | 平台自动分配独占 GPU，不手工传入 |
| `--shm-size` / `--ipc=host` | shared memory 由平台自动按 GPU 份额配置；实例内用 `df -h /dev/shm` 验证 |
| `-p 30000:30000` | 使用 SSH tunnel，或请求内部 30000 后接受随机外部端口 |
| `-v ~/.cache/huggingface:...` | 改用实例 container disk/volume 中的 HF cache |
| 本地镜像 | 使用 registry image path/tag |
| `docker run ... launch_server` | SSH 模式下放入 on-start/SSH 命令；Entrypoint 模式下作为镜像参数 |

### 3.5 首轮 smoke template

建议先在 Vast Web UI 创建一个 private template：

| 字段 | 首轮值 |
| --- | --- |
| Image | `lmsysorg/sglang:dev` |
| Launch mode | `SSH` |
| Disk | `200GB` |
| Public application port | 不开放，使用 SSH tunnel |
| On-start | 只创建 workspace 并保存环境变量 |

On-start 可以保持很短：

```bash
set -eu
mkdir -p /workspace/hf-cache /workspace/results /workspace/logs
env >> /etc/environment
```

SSH 后先运行：

```bash
export HF_HOME=/workspace/hf-cache
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 30000 \
  --reasoning-parser qwen3
```

使用 `127.0.0.1` 配合 SSH tunnel，可以避免把未认证的 SGLang API 直接暴露到公网。

## 4. 账号如何连接和使用

### 4.1 当前本机状态

本次检查确认：

- 系统当前没有安装 `vastai` CLI；
- `~/.config/vastai/vast_api_key` 当前不存在；
- 因此尚未实际验证用户 Vast.ai 账号、余额或实例权限。

### 4.2 推荐账号配置

1. 登录 `https://cloud.vast.ai`。
2. 在 Keys 页面创建一个单独的 scoped API key：
   - 名称建议 `code-agent-kvcache-local`；
   - 仅开放实例查询、创建、停止、销毁等所需权限；
   - 不开放 key 管理和非必要 billing 权限。
3. 创建一把 Vast 专用 Ed25519 SSH key，不复用 GitHub 私钥：

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_vast
   ```

4. 只把 `.pub` 公钥加入 Vast.ai；私钥永远保留在本机。
5. 在隔离 Python 环境安装 CLI：

   ```bash
   python3 -m venv ~/.venvs/vastai
   ~/.venvs/vastai/bin/pip install vastai
   ```

6. 用户在本机自行配置 API key，不要在聊天、日志或仓库中粘贴：

   ```bash
   ~/.venvs/vastai/bin/vastai set api-key <API_KEY>
   ~/.venvs/vastai/bin/vastai show user
   ~/.venvs/vastai/bin/vastai create ssh-key ~/.ssh/id_ed25519_vast.pub
   ```

`vastai set api-key` 会将 key 写入 `~/.config/vastai/vast_api_key`。CI 场景应改用 secret store 注入 `VAST_API_KEY`。

### 4.3 实例生命周期命令

典型流程：

```bash
vastai search offers '<filters>'
vastai create instance <OFFER_ID> --image <IMAGE> --disk <GB> --ssh --direct
vastai show instance <INSTANCE_ID>
vastai ssh-url <INSTANCE_ID>
vastai copy local:./input/ <INSTANCE_ID>:/workspace/input/
vastai copy <INSTANCE_ID>:/workspace/results/ local:./results/
vastai destroy instance <INSTANCE_ID>
```

SSH 连接通常是：

```bash
ssh -i ~/.ssh/id_ed25519_vast -p <SSH_PORT> root@<SSH_HOST>
```

SGLang API 推荐通过 SSH tunnel 使用，不直接暴露到公网：

```bash
ssh -i ~/.ssh/id_ed25519_vast \
  -p <SSH_PORT> root@<SSH_HOST> \
  -L 30000:localhost:30000
```

本地随后访问：

```text
http://127.0.0.1:30000
```

### 4.4 凭据边界

- Vast master/scoped API key 只保留在本地控制面。
- 不向 rented instance 注入 GitHub 写权限。
- prototype 代码先由本地使用显式 `ccdd2023` 身份推送，远端只按 public commit SHA 拉取。
- gated Hugging Face 模型使用 read-only token，并通过 Vast account secret/environment 注入，不写入模板、镜像或日志。
- Vast instance 自带 per-instance `CONTAINER_API_KEY`；不需要把本地主 API key复制进去。

## 5. 选择什么样的 RTX PRO 6000 offer

用户所说的型号应优先确认是：

```text
RTX PRO 6000 S / Blackwell Server Edition
96GB GDDR7
SM120
```

不要与 `RTX 6000 Ada` 的 48GB 型号混淆。实例启动后必须以 `nvidia-smi` 和 `torch.cuda.get_device_capability()` 为准。

首轮建议筛选：

| 维度 | 建议 |
| --- | --- |
| 实例类型 | On-demand |
| Host tier | Secure Cloud 优先，至少 Verified |
| GPU | 1x RTX PRO 6000 S，约 96GB |
| Reliability | 建议 `>= 0.98` |
| Direct port | 至少 1 个，便于 direct SSH |
| System RAM | 最低 128GB；HiCache 建议 256GB 或更高 |
| Disk | 建议 200GB 起；正式多模型实验 300GB 更稳妥 |
| Disk bandwidth | 优先 NVMe 和高 `disk_bw` |
| PCIe | 优先 Gen5 x16 / 高 `pcie_bw` |
| Internet | 模型冷下载时优先高 download bandwidth |
| Max duration | 覆盖计划实验窗口 |

Vast.ai 上 CPU/RAM 是按 GPU 份额分配的。若一台 8-GPU 机器只租 1 GPU，可能只得到约八分之一 CPU/RAM。对 HiCache 来说，不能只看 GPU 型号，必须看 offer card 中实际分配的 RAM。

Vast 当前 RTX PRO 6000 S 页面显示起价约 `$1.33/h`，实际总价随 marketplace offer 变化，并另含 storage/bandwidth：

- 4 小时约 `$5.32` GPU 费用；
- 20 小时约 `$26.60`；
- 100 小时约 `$133`；

以上只用于预算级估算，不作为固定报价。

## 6. 对当前实验的具体 benefit

### 6.1 本地机器的限制

本地 RTX 2080 SUPER：

- 8GB VRAM；
- SM75；
- Qwen3-8B-AWQ 已经需要约 5.73GB weights；
- 1024 token KV 约 0.14GB；
- 只能在很紧的 context、低 concurrency、量化和 fallback kernel 下运行。

它无法支持具有说服力的：

- 7B/8B BF16 KVCOMM；
- 大 working-set KV pressure；
- 长上下文和多 artifact resident cache；
- 大 CPU backup pool 与 H2D break-even；
- 真实三阶段 workflow 并发；
- dense baseline 与 reconstruction 的公平性能比较。

### 6.2 RTX PRO 6000 解锁的实验

96GB VRAM 允许：

1. **faithful KVCOMM 规模**
   - 7B/8B BF16 weights 可完整驻留；
   - 不再依赖 AWQ 才能启动；
   - 能比较 exact、reconstruction 和 dense fallback。

2. **长上下文与大型 KV working set**
   - 按本地 Qwen3-8B 测得的约 0.14GB/1K token 粗略外推：
     - 64K KV 约 9GB；
     - 256K KV 约 36GB。
   - 实际还受模型、batch、allocator 和 runtime overhead 影响，但远超本地 1K 限制。

3. **HiCache/KVFlow**
   - GPU 内可以构造真正的 cache pressure curve；
   - 搭配 128–256GB host RAM 测 CPU backup、prefetch、eviction 和 H2D；
   - 可验证 priority 是否在 working set 超过 GPU capacity 后产生收益。

4. **RepoKV-MVCC**
   - 同时保留多个 repository snapshot/artifact version 的 logical/physical KV；
   - 测 cross-version exact alias、selective rematerialization 和 stale audit；
   - 真实回放多个 commit、branch 和 dirty worktree。

5. **固定 workflow**
   - 跑完整 `Architect -> Coder -> Debugger`；
   - 构造多个并发 workflow，验证 KVFlow prefetch 是否能被其他 ready request 隐藏；
   - 测 role/prefix 变化下 KV variance 和 reconstruction gate。

### 6.3 仍然不能证明的内容

RTX PRO 6000 不是 H100：

- SM120 kernel 路径与 H100 SM90 不同；
- RTX PRO 6000 使用 GDDR7，H100 使用 HBM；
- PCIe、NVLink、shared memory 和 kernel optimal point 不同；
- 单卡实例不能证明 multi-GPU、RDMA、PD disaggregation 或 tensor-parallel scaling。

因此可以声称：

```text
在 RTX PRO 6000 Blackwell 上验证系统机制和性能趋势
```

不能直接声称：

```text
复现 KVCOMM/KVFlow 的 H100 主结果
```

最终论文若需要硬件外推，应在 H100 上补一组 calibration experiment。

## 7. 推荐加入 workflow 的方式

### 7.1 本地控制面

本地继续负责：

- Git branch、commit 和 code review；
- 显式 `ccdd2023` GitHub 写操作；
- metadata、version catalog、dependency graph 单元测试；
- Git trace 采集和离线分析；
- 小模型 SM75 correctness；
- 实验矩阵生成；
- 结果下载、统计、图表和文档。

### 7.2 Vast GPU 执行面

Vast 只负责：

- GPU-dependent integration test；
- 7B/8B BF16；
- long-context/KV pressure；
- CPU↔GPU tier；
- throughput、latency、TTFT、H2D 和 quality benchmark；
- 多 workflow 并发。

### 7.3 每次实验必须固定的 manifest

```text
experiment_id
git_commit
docker_image_digest
model_id
model_revision
tokenizer_revision
prompt_template_fingerprint
driver_version
cuda_version
torch_version
sglang_version
gpu_name
gpu_count
gpu_memory
cpu_model
allocated_cpu_cores
allocated_system_ram
disk_bw
pcie_bw
network_bw
vast_offer_id
vast_machine_id
launch_args
random_seed
```

这可以防止 marketplace host 异构性污染结论。

## 8. 分阶段接入计划

### Phase V0：30–60 分钟兼容性 smoke test

使用 on-demand Secure Cloud/Verified offer 和官方 SGLang dev/release image：

```bash
nvidia-smi
df -h /workspace /dev/shm
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())
print(torch.cuda.mem_get_info())
PY
```

验收条件：

- GPU 精确型号和 96GB VRAM 正确；
- compute capability 为 `(12, 0)`；
- CUDA `>=12.8`；
- PyTorch、SGLang、FlashInfer 可 import；
- Qwen3-0.6B 启动并通过 health/generate；
- `/dev/shm` 和 system RAM 满足配置；
- direct SSH 与 30000 tunnel 正常；
- 结果可以下载，实例可以销毁。

### Phase V1：2–4 小时 8B 基线

- Qwen3-8B BF16 dense prefill；
- AWQ 作为本地对照，不作为唯一远程配置；
- exact prefix cache；
- HiCache disabled/enabled；
- pinned-memory H2D microbenchmark；
- 记录 TTFT、throughput、VRAM、host RAM 和 PCIe copy bandwidth。

### Phase V2：KVCOMM 与 RepoKV-MVCC

- canonical/base KV；
- anchor delta；
- RoPE relocation；
- exact/reconstruction/dense 三路径；
- Git commit trace；
- source/dependency invalidation；
- version alias；
- stale audit；
- 质量与性能联合 gate。

### Phase V3：论文级实验

- 固定同一 machine/offer 完成可比较主表；
- 跨至少两个 host 重复关键结果，确认不是单机偶然；
- 用 H100 补充少量 hardware calibration；
- 所有 compact raw data 下载到项目 artifact 存储后销毁实例。

## 9. 风险与操作规则

1. **不要把 Vast 当长期存储。**
   Stop 只停止 compute billing，storage 继续收费；destroy 会永久删除 container data。

2. **不要把唯一结果留在 volume。**
   Vast volume 绑定物理 host，不能跨机器迁移。

3. **不要把凭据写入公共 template。**
   使用 scoped key、account secrets 和 read-only token。

4. **不要把不同 host 的 raw throughput 直接混成同一组。**
   CPU、RAM、PCIe、disk、network 和 provider contention 都可能不同。

5. **首轮不要使用 interruptible。**
   correctness 和 baseline 用 on-demand；可 checkpoint 的重复实验再考虑 interruptible。

6. **不要长期使用 mutable image tag。**
   smoke test 可用 `dev`；正式实验必须固定 image digest 或 immutable tag。

7. **不要立即启用 DeepEP。**
   当前 Dockerfile 未为 SM120 编译该路径，先完成 dense single-GPU 主线。

## 10. 最终建议

Vast.ai RTX PRO 6000 对本项目不是可有可无的加速器，而是进入可信实验阶段所需的执行资源：

- 本地机器足以开发系统逻辑，但不足以验证论文核心规模。
- RTX PRO 6000 足以承担 7B/8B、长上下文、大 KV working set 和 CPU/GPU tier 的主要 prototype 实验。
- 它不能替代 H100 的最终硬件校准，也不应替代本地 Git、文档和结果管理。

因此固定策略为：

```text
Local-first development
+ commit-addressed remote execution
+ short-lived Vast instances
+ H100 calibration only for final claims
```

## 11. 主要资料

- Vast.ai CLI quickstart: <https://docs.vast.ai/cli/hello-world>
- Vast.ai Docker environment: <https://docs.vast.ai/guides/instances/docker-environment>
- Vast.ai connection modes: <https://docs.vast.ai/guides/instances/connect/overview>
- Vast.ai SSH: <https://docs.vast.ai/guides/instances/connect/ssh>
- Vast.ai networking: <https://docs.vast.ai/guides/instances/connect/networking>
- Vast.ai templates: <https://docs.vast.ai/guides/templates/introduction>
- Vast.ai template settings: <https://docs.vast.ai/guides/templates/template-settings>
- Vast.ai storage: <https://docs.vast.ai/guides/instances/storage/types>
- Vast.ai instance lifecycle: <https://docs.vast.ai/guides/instances/manage-instances>
- Vast.ai security: <https://docs.vast.ai/guides/reference/faq/security>
- Vast.ai API keys: <https://docs.vast.ai/guides/reference/api-keys>
- Vast.ai RTX PRO 6000 S: <https://vast.ai/rent-6000s>
- SGLang installation and Docker: <https://docs.sglang.io/docs/get-started/install>
- SGLang HiCache best practices: <https://docs.sglang.io/docs/advanced_features/hicache_best_practices>
- NVIDIA CUDA compute capability: <https://developer.nvidia.com/cuda/gpus>
