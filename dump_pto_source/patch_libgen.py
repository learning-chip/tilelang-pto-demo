# patch https://github.com/tile-ai/tilelang-ascend/blob/d2405b5975edaa36166b702d2ef8c1952b7b7337/tilelang/jit/adapter/libgen.py
import os
import tempfile
import subprocess
from tilelang.env import TILELANG_TEMPLATE_PATH

def get_patched_compile_lib(src_dump_path="src.cpp", cmd_dump_path="compile_cmd.txt"):
    def patched_compile_lib(self, timeout: float = None):
        src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)
        libpath = src.name.replace(".cpp", ".so")
        ASCEND_HOME_PATH = os.environ["ASCEND_HOME_PATH"]
        TL_ROOT = os.environ["TL_ROOT"]
        if self.target == "ascendc" or self.target == "auto":
            command = [
                "bisheng",
                "--npu-arch=dav-2201",
                "-O2",
                "-std=c++17",
                "-xasc",
                f"-I{ASCEND_HOME_PATH}/include",
                f"-I{ASCEND_HOME_PATH}/include/experiment/msprof",
                f"-I{ASCEND_HOME_PATH}/include/experiment/runtime",
                f"-I{ASCEND_HOME_PATH}/pkg_inc",
                f"-I{ASCEND_HOME_PATH}/pkg_inc/runtime",
                f"-I{ASCEND_HOME_PATH}/pkg_inc/profiling",
                f"-I{TL_ROOT}/3rdparty/catlass/include",
                f"-I{TL_ROOT}/3rdparty/shmem/include",
                f"-I{TL_ROOT}/3rdparty/shmem/src/device",
                f"-DBACKEND_HYBM",
                "-I" + TILELANG_TEMPLATE_PATH,
                f"-L{ASCEND_HOME_PATH}/lib64",
                "-Wno-macro-redefined",
                "-Wno-ignored-attributes",
                "-Wno-non-c-typedef-for-linkage",
                "-lruntime",
                "-lascendcl",
                "-lm",
                "-ltiling_api",
                "-lplatform",
                "-lc_sec",
                "-ldl",
                "-fPIC",
                "--shared",
                src.name,
            ]
        elif self.target == "pto":
            ccec = "dav-c310" if self.platform == 'A5' else "dav-c220"
            memory = "REGISTER_BASE" if self.platform == 'A5' else "MEMORY_BASE"
            command = [
                "bisheng",
                f"--cce-aicore-arch={ccec}",
                f"-D{memory}",
                "-O2",
                "-std=gnu++17",
                "-xcce",
                "-mllvm",
                "-cce-aicore-stack-size=0x8000",
                "-mllvm",
                "-cce-aicore-function-stack-size=0x8000",
                "-mllvm",
                "-cce-aicore-record-overflow=true",
                "-mllvm",
                "-cce-aicore-addr-transform",
                "-mllvm",
                "-cce-aicore-dcci-insert-for-scalar=false",
                "-DL2_CACHE_HINT",
                "-I../../src/",
                f"-I{ASCEND_HOME_PATH}/include",
                f"-I{ASCEND_HOME_PATH}/include/experiment/msprof",
                f"-I{ASCEND_HOME_PATH}/include/experiment/runtime",
                f"-I/usr/local/Ascend/driver/kernel/inc",
                f"-I{TL_ROOT}/3rdparty/pto-isa/include",
                f"-I{ASCEND_HOME_PATH}/pkg_inc",
                f"-I{ASCEND_HOME_PATH}/pkg_inc/runtime",
                f"-I{ASCEND_HOME_PATH}/pkg_inc/profiling",
                f"-L{ASCEND_HOME_PATH}/lib64",
                "-I" + TILELANG_TEMPLATE_PATH,
                "-Wno-macro-redefined",
                "-Wno-ignored-attributes",
                "-lruntime",
                "-lstdc++",
                "-lascendcl",
                "-lm",
                "-ltiling_api",
                "-lplatform",
                "-lc_sec",
                "-ldl",
                "-fPIC",
                "--shared",
                src.name,
            ]
        command += ["-o", libpath]

        # NOTE: --- the patched part starts here ---
        print("dump source code to: ", src_dump_path)
        with open(src_dump_path, "w") as f:
            f.write(self.lib_code)

        print("dump compile command to: ", cmd_dump_path)
        with open(cmd_dump_path, "w") as f:
            f.write(" \\\n  ".join(command))
        # --- the patched part ends here ---

        src.write(self.lib_code)
        src.flush()
        try:
            ret = subprocess.run(command, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Compile kernel failed because of {e}") from e

        if ret.returncode != 0:
            raise RuntimeError(f"Compilation Failed! {command}")

        self.srcpath = src.name
        self.libpath = libpath

    return patched_compile_lib
