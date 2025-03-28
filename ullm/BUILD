################################################################################
#
# ullm BUILD
#
################################################################################

package(default_visibility = ["//:__subpackages__"])

# llama2 #######################################################################

cc_library(
    name = "llama2",
    srcs = ["llama2.c"],
    hdrs = [
        "llama2.h",
        "llama2-internal.h",
    ],
    deps = [
        "//sys:file",
        "//sys:memory",
        "//sys:time",
        "//util:log",
        "//util:status",
    ],
)

genrule(
    name = "models",
    srcs = [
       "@llama2//:tokenizer.bin",
       "@tinystories15M//file",
       "@tinystories110M//file",
    ],
    outs = [
       "tinystories15M.bin",
       "tinystories110M.bin",
       "tokenizer.bin",
    ],
    cmd = """
        cp $(location @tinystories15M//file) $(location :tinystories15M.bin);
        cp $(location @tinystories110M//file) $(location :tinystories110M.bin);
        cp $(location @llama2//:tokenizer.bin) $(location :tokenizer.bin);
    """,
)

cc_test(
    name = "llama2_test",
    srcs = ["llama2_test.cc"],
    data = [
        ":models",
    ],
    deps = [
        ":llama2",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
