

def gen_ch_main_targets(chs, deps):
    for ch in chs:
        native.cc_binary(
            name = "{}_main".format(ch),
            srcs = [ch],
            copts = select({
                "@platforms//os:windows": ["/std:c++20"],
                "//conditions:default": ["-std=c++20"],
            }),
            deps = [

            ] + deps,
        )
