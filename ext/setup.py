from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vit_cuda',
    ext_modules=[
        CUDAExtension(
            name='vit_cuda',
            sources=[
                'binding.cpp',
                'attention_wrapper.cpp',
                'patch_embed_wrapper.cpp',
                'pos_encoding_wrapper.cpp',
                'mlp_wrapper.cpp',
                'layernorm_wrapper.cpp',
                'classifier_wrapper.cpp',
                'gemm_wrapper.cpp',
                'qkv_wrapper.cpp',
                'patch_embed.cu',
                'pos_encoding.cu',
                'attention.cu',
                'mlp.cu',
                'layernorm.cu',
                'classifier.cu',
                'gemm.cu',
                'qkv.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--allow-unsupported-compiler',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-use_fast_math',
                    '--expt-relaxed-constexpr',
                ],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
