TVM tutorial examples adjusted for ROCm.

Following is current behavior:

tensor_expr_get_started.py:
   Basic functionality works.
   Sample case tries to save the module as a *.o file and this functionality
   is not implemented for ROCm.
   TVMError: Module[hip] does not support SaveToFile

   Update for 0.6:
   	  Save functionality is now implemented.

relay_quick_start.py:
   Crashes with an invalid ISA message:
   ### HCC STATUS_CHECK Error: HSA_STATUS_ERROR_INVALID_ISA (0x100f) at file:mcwamp_hsa.cpp line:1192
   Update for 0.6:
   	  Seem to have encountered same symptoms as
	  https://github.com/apache/incubator-tvm/issues/4371
	  also discussed at
	  https://discuss.tvm.ai/t/cannot-import-name-bilinear-sample-nchw/2510
	  Using the workaround for LD_LIBRARY_PATH gets around the issue (not unique to ROCm)
	  and no longer crashes with HCC STATUS CHECK error.

optimize/opt_conv_rocm.p:
   Runs and reports time, e.g. Convolution: 16.777117 ms

   Update for 0.6:
   	  Runs and reports time, now Convolution: 2983.906792 ms

autotune/tune_conv2d_rocm.py:
   Reports out of memory for all configurations.

   error: local memory limit exceeded (1188864) in default_function_kernel0
   error: local memory limit exceeded (4773888) in default_function_kernel0
   error: local memory limit exceeded (818688) in default_function_kernel0
   error: local memory limit exceeded (522240) in default_function_kernel0
   ### HCC STATUS_CHECK Error: HSA_STATUS_ERROR_INVALID_ARGUMENT (0x1001) at file:mcwamp_hsa.cpp line:1192

   Version did work Radeon VII (?).
   Need to understand if this is a valid test, but basic functionality is in place.

   Update for 0.6:
   	  No longer shows the "local memory limit exceeded" errors
	  A few warnings that "Unroll hint get ignore at CodeGenLLVM backend"
	  Probably still a number of invalid configs in the search space. In my sample run,
	  following were the 20 error codes:
	  	    error_no = 1 (invalid gpu kernel); 8 times
		    error_no = 6 (timeout); 10 times
		    error_no = 2 (success!); 2 times

autotune/tune_relay_rocm.py:
   Haven't tried RPC server.

topi/intro_topi.py:
   Examples work when ROCm is used as compiler/runtime instead of cuda.

language/scan.py:
   Examples work when ROCm is used as compiler/runtime instead of cuda.
   Relatively simple example.

language/reduction.py:
   Example works when ROCm is used as compiler/runtime instead of cuda.
   Relatively simple example.

   Update for 0.6:
   	  This continues to go without complaint.  Not sure what I missed before on correctness
	  issues for this one.

language/intrin_math.py:
   Example goes through without complaint.
   However, it relies on printing out the source to help verify the intrinsic
   placement and tough to see this in LLVM.  Need to step through this with
   cuda in parallel.
   intrinsic function was actually applied (grep in LLVM source code didn't show it)
   
