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

#ifndef ULLM_LLAMA2_H_
#define ULLM_LLAMA2_H_

#include <stdint.h>

// Inference for Llama-2 Transformer model.

#ifdef __cplusplus
extern "C" {
#endif

// The hyperparameters of the architecture (the blueprint).
// Loaded from the model bin.
typedef struct {
  // Transformer dimension.
  int32_t dim;
  // For ffn layers.
  int32_t hidden_dim;
  // Number of layers.
  int32_t n_layers;
  // Number of query heads.
  int32_t n_heads;
  // Number of key/value heads (can be < query heads because of multiquery).
  int32_t n_kv_heads;
  // vocabulary size, usually 256 (byte-level)
  int32_t vocab_size;
  // max sequence length
  int32_t seq_len;
} __attribute__((packed)) Llama2Config;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ULLM_LLAMA2_H_
