# Building
The Makefile defines what compiler is used. This line  ``` CC = nvcc ``` specifies that we'll use the [NVCC compiler.](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#axzz4fetCWOOG). I don't know how to install CUDA, because I used [paperspace.com](www.paperspace.com), so my machine came with CUDA already installed.

To build it, I just change to the correct directory and run the make command:

``` 
$ cd /home/paperspace/Documents/< this repo >/particles_gpu
$ make
nvcc -c -O3 -g gpu.cu
```
 

## GPU build commands
You have to make sure you're building your application for the correct GPU capabilities.
From the docs:
```
For instance, to compile an application for a GPU with compute capability 3.0, 
add the following flag to the compilation command:

-gencode arch=compute_30,code=sm_30
```
Where do those "compute_30" values come from?
You look at the [list of supported GPUs](https://developer.nvidia.com/cuda-gpus), and find yours. See the column on the right that says "Compute capability"?  

![compute capability](http://i.imgur.com/3uLLj9Z.png)

I didn't at first either. But it's there! Use that number to decide what to pass in to the compiler. For instance, for my Quadro M6000 GPU I should pass in "compute_61" and "sm_61"
[Source](http://docs.nvidia.com/cuda/cuda-gdb/#compiling-for-specific-gpus)

# Running
To run it locally on the Paperspace machine (as I was doing), you just run the executable file directly. Mine was called 'gpu':
```
$ ./gpu -n 200 -o gpu.txt
CPU-GPU copy time = 0.000431 seconds
n = 200, simulation time = 0.070571 seconds
```

# Random Helpful CUDA Notes
## Reading back a scalar from the GPU

```
__device__ double d_colAnswer;
...
typeof(d_colAnswer) colAnswer;
cudaMemcpyFromSymbol(&colAnswer, d_colAnswer, sizeof(colAnswer), 0, cudaMemcpyDeviceToHost);
printf("colAnswer: %f\n", colAnswer);
```
[Adapted from this StackOverflow post.](http://stackoverflow.com/questions/2619296/how-to-return-a-single-variable-from-a-cuda-kernel-function)

# Debugging
NVIDIA has created [a version of GDB that works with CUDA.](http://docs.nvidia.com/cuda/cuda-gdb/#axzz4artDaDSe)

1. Be sure to compile with the ```-g -G``` flags in your Makefile. [Source](http://docs.nvidia.com/cuda/cuda-gdb/#debug-compilation)
2. You can't debug the GPU while it's running, since Cuda-GDB would need to halt execution to show you the error, but the GPU is currently being used by the machine (to show you the screen and stuff). You can change this two ways. I did this way:
```
$ echo $CUDA_DEBUGGER_SOFTWARE_PREEMPTION

// Note that the above command has no return, since this value is not currently set in my environment.
// Allow the Cuda debugger to get control of the GPU for debugging:
$ export CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1
// Make sure it got set correctly
$ echo $CUDA_DEBUGGER_SOFTWARE_PREEMPTION
1 

// Yep.
```

Now, you can run the debugger and chase down your segfault:
```
cuda-gdb -ex run --args ./gpu -n 2000000 -o gpu.txt
```

Example output:
```
Starting program: /home/paperspace/Documents/molly/particles_gpu/./gpu -n 2000000 -o gpu.txt
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
[New Thread 0x7ffff64a2700 (LWP 4754)]
[New Thread 0x7ffff5ca1700 (LWP 4755)]
[New Thread 0x7ffff54a0700 (LWP 4756)]

Program received signal SIGSEGV, Segmentation fault.
0x0000000000402e36 in __fill_n_a<int*, int, int> (__value=<optimized out>, __n=10004569, __first=0x7ffffd9d3f18) at /usr/include/c++/4.8/bits/stl_algobase.h:749
749	      for (__decltype(__n + 0) __niter = __n;
(cuda-gdb) where
#0  0x0000000000402e36 in __fill_n_a<int*, int, int> (__value=<optimized out>, __n=10004569, __first=0x7ffffd9d3f18) at /usr/include/c++/4.8/bits/stl_algobase.h:749
#1  fill_n<int*, int, int> (__value=<optimized out>, __n=10004569, __first=0x7ffffd9d3f18) at /usr/include/c++/4.8/bits/stl_algobase.h:786
#2  main (argc=<optimized out>, argv=<optimized out>) at gpu.cu:180
```

