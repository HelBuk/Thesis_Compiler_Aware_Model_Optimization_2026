
# import onnx
# from onnx import TensorProto, shape_inference

# MODEL = "../../models/yolov8n_fp16_fixed.onnx"  

# m = onnx.load(MODEL)
# m = shape_inference.infer_shapes(m)

# dtype = {}

# def record_vi(vi):
#     if vi.type.HasField("tensor_type"):
#         dtype[vi.name] = vi.type.tensor_type.elem_type

# for vi in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
#     record_vi(vi)

# for init in m.graph.initializer:
#     dtype[init.name] = init.data_type

# print("Nodes with FP32 IO:")
# for i, n in enumerate(m.graph.node):
#     fp32_in = [x for x in n.input if dtype.get(x) == TensorProto.FLOAT]
#     fp32_out = [x for x in n.output if dtype.get(x) == TensorProto.FLOAT]
#     if fp32_in or fp32_out:
#         print(f"[{i}] {n.op_type}  name={n.name or '-'}")
#         if fp32_in:
#             print("  FP32 inputs :", fp32_in)
#         if fp32_out:
#             print("  FP32 outputs:", fp32_out)

# print("\nCast nodes:")
# for i, n in enumerate(m.graph.node):
#     if n.op_type == "Cast":
#         to = next((a.i for a in n.attribute if a.name == "to"), None)
#         to_str = {TensorProto.FLOAT: "FP32", TensorProto.FLOAT16: "FP16"}.get(to, str(to))
#         print(f"[{i}] Cast name={n.name or '-'} -> {to_str}")


#Check fused layers:

import re
from collections import defaultdict

import onnx
import torch
import tvm
from tvm import relax, tir
from tvm.ir import GlobalVar, Op, structural_equal
from tvm.relax.frontend.onnx import from_onnx
from tvm.s_tir import meta_schedule as ms

ONNX_PATH = "../../models/yolov8n_fp16_fixed.onnx"
SHAPE_DICT = {"images": [1, 3, 640, 640]}
DTYPE_DICT = {"images": "float16"}  # set to float32 if your ONNX input is float32

print("Setting up cuda target")
cc_major, cc_minor = torch.cuda.get_device_capability(0)
TARGET = tvm.target.Target(
    {
        "kind": "cuda",
        "arch": f"sm_{cc_major}{cc_minor}",
        "max_num_threads": 1024,
        "max_threads_per_block": 1024,
        "thread_warp_size": 32,
        "max_shared_memory_per_block": 49152,
        "registers_per_block": 65536,
        "host": {
            "kind": "llvm",
            "mtriple": "aarch64-linux-gnu",
            "mcpu": "cortex-a78",
        },
    }
)

# TVM task/gvar token -> ONNX op type
NAME_TOKEN_TO_ONNX = {
    "conv2d_transpose": "ConvTranspose",
    "max_pool2d": "MaxPool",
    "resize2d": "Resize",
    "strided_slice": "Slice",
    "tir_sigmoid": "Sigmoid",
    "conv2d": "Conv",
    "softmax": "Softmax",
    "transpose": "Transpose",
    "concatenate": "Concat",
    "reshape": "Reshape",
    "split": "Split",
    "cast": "Cast",
    "multiply": "Mul",
    "divide": "Div",
    "subtract": "Sub",
    "add": "Add",
}

TOKEN_RE = re.compile(
    "(" + "|".join(sorted(NAME_TOKEN_TO_ONNX.keys(), key=len, reverse=True)) + r")(\d*)"
)


def build_onnx_occurrence_index(model):
    # (op_type, occurrence_idx) -> node
    idx = {}
    counters = defaultdict(int)
    for node in model.graph.node:
        k = counters[node.op_type]
        counters[node.op_type] += 1
        idx[(node.op_type, k)] = node
    return idx


def extract_tokens(name):
    out = []
    for m in TOKEN_RE.finditer(name):
        token = m.group(1)
        occ = int(m.group(2)) if m.group(2) else 0
        out.append((token, occ))
    return out


def resolve_onnx_nodes_from_name(name, onnx_occ_idx):
    refs = []
    for token, occ in extract_tokens(name):
        op_type = NAME_TOKEN_TO_ONNX[token]
        node = onnx_occ_idx.get((op_type, occ))
        if node is None:
            refs.append(f"{token}{occ} -> {op_type}[{occ}] <not found>")
            continue

        node_name = node.name if node.name else "<unnamed>"
        outs = ", ".join(node.output[:2]) if len(node.output) else "-"
        refs.append(f"{token}{occ} -> {node_name} ({op_type}[{occ}]) outputs: {outs}")

    # dedupe while keeping order
    deduped, seen = [], set()
    for r in refs:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return deduped


def iter_bindings(func: relax.Function):
    body = func.body
    if not isinstance(body, relax.SeqExpr):
        return
    for block in body.blocks:
        for binding in block.bindings:
            yield binding


def is_call_tir(expr):
    return (
        isinstance(expr, relax.Call)
        and isinstance(expr.op, Op)
        and expr.op.name == "relax.call_tir"
    )


def block_names(prim_func: tir.PrimFunc):
    names = []

    def visit(node):
        if isinstance(node, tir.SBlock):
            hint = getattr(node, "name_hint", None)
            if hint:
                names.append(hint)

    tir.stmt_functor.post_order_visit(prim_func.body, visit)

    seen = set()
    ordered = []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


onnx_model = onnx.load(ONNX_PATH)
onnx_occ_idx = build_onnx_occurrence_index(onnx_model)

mod = from_onnx(onnx_model, shape_dict=SHAPE_DICT, dtype_dict=DTYPE_DICT)

mod_tir = tvm.transform.Sequential(
    [
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(),
        relax.transform.FoldConstant(),
        relax.transform.FuseTIR(),
    ]
)(mod)

tasks = ms.relax_integration.extract_tasks(mod_tir, TARGET, params={})
normalize_mod = tvm.get_global_func("tvm.s_tir.meta_schedule.normalize_mod")

normalized_task_table = [
    (i, t.task_name, t.weight, normalize_mod(t.dispatched[0])) for i, t in enumerate(tasks)
]


def match_task(prim_func: tir.PrimFunc):
    normalized = normalize_mod(prim_func)
    for i, task_name, task_weight, task_mod in normalized_task_table:
        if structural_equal(normalized, task_mod):
            return i, task_name, task_weight
    return None, None, None


print("\n=== Ordered Flow (main execution order) ===")
print("Note: ONNX mapping is best-effort via token+occurrence index (e.g., conv2d40 -> Conv[40]).")

step = 0
for binding in iter_bindings(mod_tir["main"]):
    if not isinstance(binding, relax.VarBinding):
        continue
    if not is_call_tir(binding.value):
        continue

    gvar = binding.value.args[0]
    if not isinstance(gvar, GlobalVar):
        continue

    prim_func = mod_tir[gvar]
    task_idx, task_name, task_weight = match_task(prim_func)
    names = block_names(prim_func)

    print(f"\nStep {step:03d}")
    print(f"  call_tir gvar : {gvar.name_hint}")
    if task_idx is not None:
        print(f"  matched task  : #{task_idx} {task_name} (weight={task_weight})")
    else:
        print("  matched task  : <not found>")
    print(f"  TIR blocks    : {', '.join(names[:10])}" + (" ..." if len(names) > 10 else ""))

    # Prefer task_name for mapping (often richer for fused ops), fallback to gvar.name_hint.
    name_for_mapping = task_name if task_name is not None else gvar.name_hint
    refs = resolve_onnx_nodes_from_name(name_for_mapping, onnx_occ_idx)
    if refs:
        print("  ONNX origin   :")
        for r in refs[:12]:
            print(f"    - {r}")
    else:
        print("  ONNX origin   : <no token match>")

    step += 1

print(f"\nTotal call_tir steps in main: {step}")
print(f"Total unique extracted tasks: {len(tasks)}")
