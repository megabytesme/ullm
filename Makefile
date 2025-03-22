################################################################################
#
# POSIX Makefile
#
################################################################################

# Config #######################################################################

OUT := out

SRCS := \
		$(OUT)/c-flags/lib/c-flags.c \
		$(OUT)/c-flags/lib/string-view.c \
    ullm/llama2.c \
    util/log.c \
		util/status.c \
		sys/file.c \
		sys/memory.c \
    tools/ullm.c

OPT := 0

BIN := ullm

# build ########################################################################

BIN := $(OUT)/$(BIN)

OBJS := $(patsubst %.c, $(OUT)/%.o, $(SRCS))

CFLAGS := \
    -I . \
		-I out/c-flags/lib \
    -std=c99 \
    -Wall -Werror \
    -O$(OPT) \
    -ffunction-sections \
    -fdata-sections

LDFLAGS := \
    -Wl,-gc-sections \
		-lm

.PHONY:
all: $(BIN).elf
	size $(BIN).elf

.PHONY:
fetchdeps:
	@mkdir -p $(OUT)
	cd $(OUT); git clone https://github.com/DieTime/c-flags.git

$(BIN).elf: $(OBJS)
	cc $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT)/%.o: %.c
	@mkdir -p $(dir $@)
	cc $(CFLAGS) -c $< -o $@

.PHONY:
clean:
	rm -rf $(OUT)
