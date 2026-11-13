# Direct AAOCA segmentation

The command-line entry starts native-HU segmentation without an interactive menu or web service. Explicit visible GPU indices form the permitted pool. Each GPU runs one case at a time; cases may run on different GPUs concurrently, subject to CPU, RAM and I/O limits. The three TotalSegmentator tasks remain sequential within each case.

GPU UUIDs are parsed with the standard UUID library and represented as full GPU-prefixed identifiers. Both PyTorch spellings refer to the same device. Saved allocations remain limited to the inherited visible devices.

New direct jobs allow other compute processes on a selected GPU. Admission requires free memory of at least the requested working allowance plus a separate reserve. Defaults are 8 GiB plus 2 GiB. Each model launch rechecks this threshold and waits when it is not met. Memory can change concurrently; this is admission control, not an exclusive reservation or a hard allocation limit. Allocation failures stop the affected case and remain visible in logs.

The program reuses the runtime's independent environment, private license and model cache. Install, license, inventory, plan, check, run, status, logs, stop and resume are explicit commands. Run waits by default; --background submits a detached owner. Requests and code are frozen before dispatch. Resume uses the same frozen task. Status and logs do not create a runtime or launch inference.

Output remains a named, colored, overlap-preserving segmentation on the original CT grid. Verified final labels precede owned-mask cleanup. Inputs and private records remain outside the source checkout and publication whitelist.

The implementation reuses the existing case queue and frozen-job owner. A second independent scheduler would duplicate resource and recovery behavior; dispatching all cases to one shared CUDA process would lose per-case ownership and independent GPU binding.

Validation covers two generic UUID samples, explicit argument parsing, occupied-GPU admission, memory reserve, visibility restrictions, one actual local GPU, and one actual native CT without training. Multi-GPU hardware performance requires a machine with multiple allocated GPUs.
