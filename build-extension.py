from __future__ import annotations

import os
import shutil
import sys

from pathlib import Path

from Cython.Build import cythonize
from setuptools import Distribution
from setuptools import Extension
from setuptools.command.build_ext import build_ext


if sys.platform == "win32":
    COMPILE_ARGS = ["/O2", "/fp:fast"]
    LINK_ARGS = []
    INCLUDE_DIRS = []
    LIBRARIES = []
else:
    COMPILE_ARGS = ["-march=native", "-O3", "-msse", "-msse2", "-mfma", "-mfpmath=sse"]
    LINK_ARGS = []
    INCLUDE_DIRS = []
    LIBRARIES = ["m"]


def build() -> None:

    extensions = []
    pyx_files = list(Path("smarttree").glob("*.pyx"))
    for pyx_file in pyx_files:
        module_name = f"smarttree.{pyx_file.stem}"
        extension = Extension(
            module_name,
            [str(pyx_file)],
            extra_compile_args=COMPILE_ARGS,
            extra_link_args=LINK_ARGS,
            include_dirs=INCLUDE_DIRS,
            libraries=LIBRARIES,
        )
        extensions.append(extension)

    ext_modules = cythonize(
        extensions,
        include_path=INCLUDE_DIRS,
        compiler_directives={"binding": True, "language_level": 3},
        build_dir="build",
    )

    distribution = Distribution({
        "name": "smarttree",
        "ext_modules": ext_modules
    })

    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    for output in cmd.get_outputs():
        output = Path(output)
        relative_extension = Path(".") / output.relative_to(cmd.build_lib)

        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)


if __name__ == "__main__":
    build()
