from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="vit_cuda",
    version="0.1.0",
    packages=find_packages(),

    ext_modules=[
        CUDAExtension(
            name="vit_cuda._C",

            sources=[
                "ext/binding.cpp",
                "ext/attention_wrapper.cpp",
                "ext/patch_embed_wrapper.cpp",
                "ext/pos_encoding_wrapper.cpp",
                "ext/mlp_wrapper.cpp",
                "ext/layernorm_wrapper.cpp",
                "ext/classifier_wrapper.cpp",
                "ext/gemm_wrapper.cpp",
                "ext/qkv_wrapper.cpp",

                "ext/patch_embed.cu",
                "ext/pos_encoding.cu",
                "ext/attention.cu",
                "ext/mlp.cu",
                "ext/layernorm.cu",
                "ext/classifier.cu",
                "ext/gemm.cu",
                "ext/qkv.cu",
            ],

            extra_compile_args={
                "cxx": [
                    "-O3",
                ],

                "nvcc": [
                    "-O3",
                    "--allow-unsupported-compiler",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-use_fast_math",
                    "--expt-relaxed-constexpr",
                ],
            },
        ),
    ],

    cmdclass={
        "build_ext": BuildExtension,
    },
)
