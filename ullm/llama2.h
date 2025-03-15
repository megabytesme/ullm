/*
 * MIT License
 *
 * Copyright (c) 2023 Andrej
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ULLM_ULLM_LLAMA2_H_
#define ULLM_ULLM_LLAMA2_H_

#include <stdint.h>

#include "util/status.h"

// Inference for Llama-2 Transformer model.

#ifdef __cplusplus
extern "C" {
#endif

// The hyperparameters of the architecture (the blueprint).
// Loaded from the model bin.
typedef struct {
  int32_t dim;
  int32_t hidden_dim;
  int32_t n_layers;
  int32_t n_heads;
  int32_t n_kv_heads;
  int32_t vocab_size;
  int32_t seq_len;
} __attribute__((packed)) UllmLlama2Config;

// The runtime config for the inference operation.
typedef struct {
  // The prompt to generate a response to.
  const char* prompt;

  // The path to the checkpoint file.
  const char* checkpoint_path;

  // The path to the tokenizer file.
  const char* tokenizer_path;

  // Model configuration.
  float temperature;
  float topp;
  unsigned int steps;

  // The source of entropy.
  uint64_t rng_seed;
} UllmLlama2RunConfig;

// Default initialize an UllmLlama2RunConfig.
void UllmLlama2RunConfigInit(UllmLlama2RunConfig* config);

// Runs UllmLlama2 inference with the supplied config.
UllmStatus UllmLlama2Generate(const UllmLlama2RunConfig* config);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ULLM_ULLM_LLAMA2_H_
