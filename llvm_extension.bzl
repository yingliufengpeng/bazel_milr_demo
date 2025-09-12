# extension.bzl
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository", "new_local_repository")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def _my_extension_impl(mctx):
    debug = "debug/"
    if "win" in mctx.os.name:
        path = "D:/repo/llvm_repo/cd708029e0b2869e80abe31ddb175f7c35361f90"

    else:
        path = "/home/peng/src_repo/llvm_src/cd708029e0b2869e80abe31ddb175f7c35361f90"

    print("llvm info:    {} {}".format(mctx.os.name, path))

    #    local_repository(name = "llvm-project", path = path)
    new_local_repository(name = "llvm-project", path = path, build_file_content = "# empty")

    # Optional LLVM dependencies for performance
    maybe(
        http_archive,
        name = "llvm_zstd",
        build_file = "@llvm-project//utils/bazel/third_party_build:zstd.BUILD",
        sha256 = "5194fbfa781fcf45b98c5e849651aa7b3b0a008c6b72d4a0db760f3002291e94",
        strip_prefix = "zstd-1.5.0",
        urls = [
            "https://github.com/facebook/zstd/releases/download/v1.5.0/zstd-1.5.0.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "llvm_zlib",
        build_file = "@llvm-project//utils/bazel/third_party_build:zlib-ng.BUILD",
        sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
        strip_prefix = "zlib-ng-2.0.7",
        urls = [
            "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
        ],
    )

llvm = module_extension(implementation = _my_extension_impl)
