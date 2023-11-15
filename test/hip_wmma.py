# Test for wmma GEMM using hip backend
# Derived from unit tests.

# Both this file and unit tests core dump with an invalid pointer
#
import numpy as np

import tvm
from tvm import te
from tvm import meta_schedule as ms
from tvm._ffi import register_func
from tvm.meta_schedule.builder import LocalBuilder
from tvm.target import Target

# get tensor intrin
from tvm.tir.tensor_intrin import rocm

# generate schedule for matrix multiply
def matmul_fp16(M:int, N:int, K:int, in_dtype: str, out_dtype:str):
    x = te.placeholder((M,K), name="X", dtype=in_dtype)
    y = te.placeholder((K,N), name="Y", dtype=in_dtype)
    k = te.reduce_axis((0,K), name="k")
    c = te.compute(
        (M,N),
        lambda i,j: te.sum(x[i][k].astype(out_dtype) * y[k][j].astype(out_dtype),axis=[k]),
        name="C",
        )
    return (x,y,c)

def initializer():
    @register_func("meta_schedule.builder.async_build")
    def async_build(mod, target, _params):
        from tvm.driver import build as tvm_build
        from tvm.tir.transform import RemoveWeightLayoutRewriteBlock

        # re-import here for local builder to register index_map_m16n8k8_matrix
        from tvm.tir.tensor_intrin import rocm

        mod = RemoveWeightLayoutRewriteBlock(skip_ndarray_rewrite=True)(mod)
        with tvm.transform.PassContext(config={"tir.use_async_copy":1}):
            rt_mod = tvm_build(mod, target=target)
        return rt_mod

def test_wmma_tune(tune_idx):
    M_list = [8192, 8192, 14336, 8192, 8192, 4864]
    K_list = [14336, 9728, 8192, 4864, 14336, 8192]
    N_list = [9728, 14336, 8192, 4864, 14336, 14336]

    M, N, K = M_list[tune_idx], N_list[tune_idx], K_list[tune_idx]

    print("========== Tuning Matrix Dim is M: %d, N: %d, K: %d: ==========" % (M,N,K))
    def tune(in_dtype,out_dtype):
        target = Target("hip")
        func = te.create_prim_func(
            matmul_fp16(M=M, N=N, K=K, in_dtype=in_dtype, out_dtype=out_dtype)
        ).with_attr({"global_symbol": "main"})

        space = ms.space_generator.PostOrderApply(
            sch_rules="rocm-matrixcore",
            postprocs="rocm-matrixcore",
            mutator_probs="rocm-matrixcore",
        )

        mod = tvm.IRModule({"main": func})
        work_dir = "./matrix_core_text"
        db = ms.tir_integration.tune_tir(
            mod = mod,
            target=target,
            work_dir=work_dir,
            max_trials_global=128,
            builder=LocalBuilder(
                f_build="meta_schedule.builder.async_build", initializer=initializer
            ),
            space=space,
        )
        sch = db.query_schedule(mod, target=target, workload_name="main")
        with tvm.transform.PassContext(config={"tir.use_async_copy":1}):
            rt_mod = tvm.build(sch.mod, target=target)
            print(rt.mod.imported_modules[0].get_source())
        a_np = np.random.uniform(0,1, size=(M,K)).astype(in_dtype)
        b_np = np.random.uniform(0,1, size=(K,N)).astype(in_dtype)
        dev = tvm.rocm(0)
        a_tvm = tvm.nd.array(a_np, device=tvm.rocm(0))
        b_tvm = tvm.nd.array(b_np, device=tvm.rocm(0))
        c_tvm = tvm.nd.array(np.empty((M,N)).astype(out_dtype), device=tvm.rocm(0))
        rt_mod(a_tvm,b_tvm,c_tvm)
        if M < 256 and N < 256 and K < 256:
            golden = np.matmul(a_np.astype(in_dtype), b_np.astype(in_dtype))
            tvm.testing.assert_allclose(golden, c_tvm.numpy(), atol=1e-3, rtol=1e-3)

    tune("float16", "float32")

if __name__ == "__main__":
    for tune_idx in range(0,6):
        test_wmma_tune(tune_idx)
