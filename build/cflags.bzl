################################################################################
#
# cflags BUILD
#
################################################################################

package(default_visibility = ["//visibility:public"])

# cflags #######################################################################

cc_library(
    name = "cflags",
    srcs = [
        "lib/c-flags.c",
        "lib/string-view.c",
    ],
    hdrs = [
        "lib/c-flags.h",
        "lib/string-view.h",
    ],
    includes = ["lib"],
)
