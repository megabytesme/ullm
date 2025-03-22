################################################################################
#
# tools BUILD
#
################################################################################

package(default_visibility = ["//:__subpackages__"])

# ullm #########################################################################

cc_binary(
    name = "ullm",
    srcs = ["ullm.c"],
    data = [
        "//ullm:models",
    ],
    deps = [
        "//ullm:llama2",
        "//util:macros",
        "@cflags//:cflags",
    ],
)

# ullm-eswap ###################################################################

cc_binary(
    name = "ullm-eswap",
    srcs = ["ullm_eswap.c"],
    deps = [
        "//sys:file",
        "//util:log",
        "@cflags//:cflags",
    ],
)
