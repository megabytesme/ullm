load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")

# llama2 #######################################################################

git_repository(
    name = "llama2",
    commit = "350e04fe35433e6d2941dce5a1f53308f87058eb",
    remote = "https://github.com/karpathy/llama2.c.git",
    build_file = "@//build:llama2.bzl",
)

http_file(
    name = "tinystories15M",
    url = "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
    integrity = "sha256-zVkGRNljhnorbloRB/UfrWY8QdecFJ++y7sflfqB9Jo=",
    downloaded_file_path = "tinystories15M.bin",
)
