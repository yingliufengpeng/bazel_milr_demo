
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

def gentbl_cc_lib(name, type1, file_name, depends, prefix=None):
    prefix = prefix or []
    base_dir = ''
    for e in prefix:
        base_dir += '{}/'.format(e)
    gentbl_cc_name = '{}_{}_{}'.format(name, type1, file_name)
    gentbl_cc_library(
        name = gentbl_cc_name,
        tbl_outs = [
            (
                ["-gen-{}-decls".format(type1)],
                "{}{}.h.inc".format(base_dir, file_name),
            ),
            (
                ["-gen-{}-defs".format(type1)],
                "{}{}.cpp.inc".format(base_dir, file_name),
            ),
        ],
        tblgen = "@llvm-project//mlir:mlir-tblgen",
        td_file = "include/{}{}.td".format(base_dir, file_name),
        deps = depends,
    )

    return gentbl_cc_name


def gentbl_pass_cc_lib(name, file_name, depends, prefix=None):
    prefix = prefix or []
    base_dir = ''
    for e in prefix:
        base_dir += '{}/'.format(e)
    gentbl_cc_name = '{}_{}'.format(name, 'pass')
    gentbl_cc_library(
        name = gentbl_cc_name,
        tbl_outs = [
            (
                ["-gen-pass-decls",],
                "{}{}.h.inc".format(base_dir, file_name),
            ),

        ],
        tblgen = "@llvm-project//mlir:mlir-tblgen",
        td_file = "include/{}{}.td".format(base_dir, file_name),
        deps = depends + [
#            "@llvm-project//mlir:OpBaseTdFiles",
            "@llvm-project//mlir:PassBaseTdFiles",
                                ],
    )

    return gentbl_cc_name
