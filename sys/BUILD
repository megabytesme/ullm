################################################################################
#
# memory BUILD
#
################################################################################

package(default_visibility = ["//:__subpackages__"])

# file #########################################################################

cc_library(
    name = "file",
    srcs = ["file.c"],
    hdrs = ["file.h"],
    deps = [
        ":memory",
        "//util:log",
        "//util:status",
    ],
)

# memory #######################################################################

cc_library(
    name = "memory",
    srcs = ["memory.c"],
    hdrs = ["memory.h"],
    deps = [
        "//util:log",
    ],
)

# time #########################################################################

cc_library(
    name = "time",
    srcs = ["time.c"],
    hdrs = ["time.h"],
    deps = [
        "//util:status",
    ],
)
