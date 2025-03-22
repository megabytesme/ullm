# ullm

An effort to bring LLMs to very resource-constrained systems.

## Build, Test and Run

### Bazel

Bazel is the primary and modern way to build and test.

Build:

```
bazel build tools:ullm
```

Test:

```
bazel test ...
```

Run:

```
bazel run tools/ullm -- -p "The quick brown fox jumped. Where did they go?"
```

### Makefile

A Makefile is offered as a way to more easily tweak the build. It is not as
full-featured as Bazel.

Build:

```
make clean && make fetchdeps && make -j`nproc`
```

Run:

```
./out/ullm.elf -c out/stories15M.bin -t out/llama2.c/tokenizer.bin -p "The quick brown fox jumped. Where did he go?"
```

## Ports

### llama2.c

`ullm` contains a heavily modified version of the `llama2.c` project.

Forked from [llama2.c](https://github.com/karpathy/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb).
