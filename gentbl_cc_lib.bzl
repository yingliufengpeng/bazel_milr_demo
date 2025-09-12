
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

def gentbl_cc_lib(name, type1, file_name, depnds):
    gentbl_cc_name = '{}_{}'.format(name, type1)
    gentbl_cc_library(
        name = gentbl_cc_name,
        tbl_outs = [
            (
                ["-gen-{}-decls".format(type1)],
                "{}.h.inc".format(file_name),
            ),
            (
                ["-gen-{}-defs".format(type1)],
                "{}.cpp.inc".format(file_name),
            ),
        ],
        tblgen = "@llvm-project//mlir:mlir-tblgen",
        td_file = "include/{}.td".format(file_name),
        deps = depnds,
    )

    return gentbl_cc_name

