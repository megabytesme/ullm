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

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <endian.h>
#include <time.h>

#ifdef __ALTIVEC__
#include <altivec.h>
#define ULLM_ALTIVEC_ENABLED
#endif

#include "sys/memory.h"
#include "sys/time.h"
#include "ullm/llama2.h"
#include "util/log.h"
#include "util/macros.h"

#define ULLM_LOG_TAG "ullm.llama2"

// --- Forward Declarations for static functions ---
static void UllmLlama2FreeTokenizer(UllmLlama2State* state);
static void UllmLlama2Decode(const UllmLlama2RunConfig* config,
                            UllmLlama2State* state,
                            UllmLlama2Tokenizer* t,
                            int prev_token, int token);
static void UllmLlama2FreeTransformer(UllmLlama2Transformer* t);
// --- End Forward Declarations ---


static UllmStatus UllmLlama2MallocRunState(UllmLlama2Transformer* t) {
  UllmLlama2RunState* s = &t->state;
  UllmLlama2Config* p = &t->config;
  size_t alloc_size;

  if (p == NULL || t == NULL || s == NULL ) return ULLM_STATUS_INVALID_ARGUMENT;
  if (p->dim <= 0 || p->n_heads <= 0 || p->n_kv_heads <= 0 || p->n_layers <=0 || p->seq_len <= 0 || p->vocab_size <= 0 || p->hidden_dim <= 0) {
      ULOGE("Invalid config dimensions for MallocRunState");
      return ULLM_STATUS_INVALID_ARGUMENT;
  }
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  if (kv_dim <= 0 || p->dim % p->n_heads != 0) {
      ULOGE("Invalid kv_dim or head_size calculated in MallocRunState");
      return ULLM_STATUS_INVALID_ARGUMENT;
  }
  size_t kv_dim_s = (size_t)kv_dim;

  alloc_size = (size_t)p->dim * sizeof(float);
  s->x = UllmMemoryAlloc(alloc_size);
  if (!s->x) { ULOGE("Failed to allocate s->x"); goto oom_cleanup; }

  alloc_size = (size_t)p->dim * sizeof(float);
  s->xb = UllmMemoryAlloc(alloc_size);
  if (!s->xb) { ULOGE("Failed to allocate s->xb"); goto oom_cleanup; }

  alloc_size = (size_t)p->dim * sizeof(float);
  s->xb2 = UllmMemoryAlloc(alloc_size);
  if (!s->xb2) { ULOGE("Failed to allocate s->xb2"); goto oom_cleanup; }

  alloc_size = (size_t)p->hidden_dim * sizeof(float);
  s->hb = UllmMemoryAlloc(alloc_size);
  if (!s->hb) { ULOGE("Failed to allocate s->hb"); goto oom_cleanup; }

  alloc_size = (size_t)p->hidden_dim * sizeof(float);
  s->hb2 = UllmMemoryAlloc(alloc_size);
  if (!s->hb2) { ULOGE("Failed to allocate s->hb2"); goto oom_cleanup; }

  alloc_size = (size_t)p->dim * sizeof(float);
  s->q = UllmMemoryAlloc(alloc_size);
  if (!s->q) { ULOGE("Failed to allocate s->q"); goto oom_cleanup; }

  alloc_size = (size_t)p->n_layers * p->seq_len * kv_dim_s * sizeof(float);
  s->key_cache = UllmMemoryAlloc(alloc_size);
  if (!s->key_cache) { ULOGE("Failed to allocate s->key_cache"); goto oom_cleanup; }

  alloc_size = (size_t)p->n_layers * p->seq_len * kv_dim_s * sizeof(float);
  s->value_cache = UllmMemoryAlloc(alloc_size);
  if (!s->value_cache) { ULOGE("Failed to allocate s->value_cache"); goto oom_cleanup; }

  alloc_size = (size_t)p->n_heads * p->seq_len * sizeof(float);
  s->att = UllmMemoryAlloc(alloc_size);
  if (!s->att) { ULOGE("Failed to allocate s->att"); goto oom_cleanup; }

  alloc_size = (size_t)p->vocab_size * sizeof(float);
  s->logits = UllmMemoryAlloc(alloc_size);
  if (!s->logits) { ULOGE("Failed to allocate s->logits"); goto oom_cleanup; }

  return ULLM_STATUS_OK;

oom_cleanup:
    ULOGE("Failed to allocate one or more run state buffers (OOM?)");
    if(s->x) UllmMemoryFree(s->x);
    if(s->xb) UllmMemoryFree(s->xb);
    if(s->xb2) UllmMemoryFree(s->xb2);
    if(s->hb) UllmMemoryFree(s->hb);
    if(s->hb2) UllmMemoryFree(s->hb2);
    if(s->q) UllmMemoryFree(s->q);
    if(s->key_cache) UllmMemoryFree(s->key_cache);
    if(s->value_cache) UllmMemoryFree(s->value_cache);
    if(s->att) UllmMemoryFree(s->att);
    return ULLM_STATUS_OOM;
}


static UllmStatus ReadWeight(UllmFile* file,
    float** dst, size_t size, const char* name) {
  size_t buf_size = size * sizeof(float);
  *dst = NULL;
  if (size == 0 || buf_size == 0) {
      ULOGE("Attempted to allocate zero size for '%s'", name);
      return ULLM_STATUS_INVALID_ARGUMENT;
  }

  *dst = UllmMemoryAlloc(buf_size);

  if (*dst == NULL) {
    ULOGE("Failed to allocate %s (size %zu bytes)", name, buf_size);
    return ULLM_STATUS_OOM;
  }

  UllmStatus read_status = UllmFileRead(file, *dst, buf_size);
  if (read_status != ULLM_STATUS_OK) {
    ULOGE("Failed to read %s", name);
    UllmMemoryFree(*dst);
    *dst = NULL;
    return read_status;
  }

  #if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    uint32_t* ptr = (uint32_t*)(*dst);
    size_t count = size;
    for (size_t i = 0; i < count; ++i) {
        ptr[i] = __builtin_bswap32(ptr[i]);
    }
  #endif

  return ULLM_STATUS_OK;
}

static UllmStatus ReadWeights(UllmLlama2TransformerWeights *w,
    UllmLlama2Config* p, UllmFile* file, int shared_weights) {
  if (p == NULL || w == NULL || file == NULL) return ULLM_STATUS_INVALID_ARGUMENT;
  if (p->dim <= 0 || p->n_heads <= 0 || p->vocab_size <= 0 || p->n_layers <= 0 || p->hidden_dim <= 0 || p->n_kv_heads <= 0 || p->seq_len <= 0) {
      ULOGE("Invalid config dimensions for ReadWeights");
      return ULLM_STATUS_INVALID_ARGUMENT;
  }
  int head_size = p->dim / p->n_heads;
  if (head_size <= 0 || p->dim % p->n_heads != 0) {
      ULOGE("Invalid head_size calculated (%d)", head_size);
      return ULLM_STATUS_INVALID_ARGUMENT;
  }
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  if (kv_dim <= 0 || (p->dim * p->n_kv_heads) % p->n_heads != 0) {
      ULOGE("Invalid kv_dim calculated (%d)", kv_dim);
      return ULLM_STATUS_INVALID_ARGUMENT;
  }

  size_t n_layers_s = (size_t)p->n_layers;
  size_t dim_s = (size_t)p->dim;
  size_t hidden_dim_s = (size_t)p->hidden_dim;
  size_t vocab_size_s = (size_t)p->vocab_size;
  size_t seq_len_s = (size_t)p->seq_len;
  size_t head_size_s = (size_t)head_size;
  size_t kv_dim_s = (size_t)kv_dim;


  ULLM_RETURN_IF_ERROR(ReadWeight(file, &w->token_embedding_table,
      vocab_size_s * dim_s, "token_embedding_table"));
  ULLM_RETURN_IF_ERROR(ReadWeight(file, &w->rms_att_weight,
      n_layers_s * dim_s, "rms_att_weight"));
  ULLM_RETURN_IF_ERROR(ReadWeight(file, &w->wq,
      n_layers_s * dim_s * dim_s, "wq"));
  ULLM_RETURN_IF_ERROR(ReadWeight(file, &w->wk,
      n_layers_s * dim_s * kv_dim_s, "wk"));
  ULLM_RETURN_IF_ERROR(ReadWeight(file, &w->wv,
      n_layers_s * dim_s * kv_dim_s, "wv"));
  ULLM_RETURN_IF_ERROR(ReadWeight(file, &w->wo,
      n_layers_s * dim_s * dim_s, "wo"));
  ULLM_RETURN_IF_ERROR(ReadWeight(file, &w->rms_ffn_weight,
      n_layers_s * dim_s, "rms_ffn_weight"));
  ULLM_RETURN_IF_ERROR(ReadWeight(file, &w->w1,
      n_layers_s * dim_s * hidden_dim_s, "w1"));
  ULLM_RETURN_IF_ERROR(ReadWeight(file, &w->w2,
      n_layers_s * hidden_dim_s * dim_s, "w2"));
  ULLM_RETURN_IF_ERROR(ReadWeight(file, &w->w3,
      n_layers_s * dim_s * hidden_dim_s, "w3"));
  ULLM_RETURN_IF_ERROR(ReadWeight(file, &w->rms_final_weight,
      dim_s, "rms_final_weight"));

  uint64_t seek_offset = (uint64_t)seq_len_s * head_size_s / 2 * sizeof(float);
  ULLM_RETURN_IF_ERROR(UllmFileSeek(file, seek_offset));
  ULLM_RETURN_IF_ERROR(UllmFileSeek(file, seek_offset));


  if (shared_weights) {
    w->wcls = w->token_embedding_table;
  } else {
    ULLM_RETURN_IF_ERROR(ReadWeight(file, &w->wcls,
        vocab_size_s * dim_s, "wcls"));
  }

  return ULLM_STATUS_OK;
}

static UllmStatus UllmLlama2ReadCheckpoint(const UllmLlama2RunConfig* config,
    UllmLlama2State* state) {
  if (config == NULL || state == NULL) return ULLM_STATUS_INVALID_ARGUMENT;
  UllmFile checkpoint_file;
  checkpoint_file.fd = -1;
  ULLM_RETURN_IF_ERROR(UllmFileOpen(config->checkpoint_path, &checkpoint_file));

  UllmLlama2Transformer* t = &state->transformer;
  UllmStatus status = ULLM_STATUS_OK;

  ULLM_GOTO_IF_ERROR(cleanup, status, UllmFileRead(&checkpoint_file, &t->config.dim, sizeof(int32_t)));
  ULLM_GOTO_IF_ERROR(cleanup, status, UllmFileRead(&checkpoint_file, &t->config.hidden_dim, sizeof(int32_t)));
  ULLM_GOTO_IF_ERROR(cleanup, status, UllmFileRead(&checkpoint_file, &t->config.n_layers, sizeof(int32_t)));
  ULLM_GOTO_IF_ERROR(cleanup, status, UllmFileRead(&checkpoint_file, &t->config.n_heads, sizeof(int32_t)));
  ULLM_GOTO_IF_ERROR(cleanup, status, UllmFileRead(&checkpoint_file, &t->config.n_kv_heads, sizeof(int32_t)));
  ULLM_GOTO_IF_ERROR(cleanup, status, UllmFileRead(&checkpoint_file, &t->config.vocab_size, sizeof(int32_t)));
  ULLM_GOTO_IF_ERROR(cleanup, status, UllmFileRead(&checkpoint_file, &t->config.seq_len, sizeof(int32_t)));

  #if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    t->config.dim = __builtin_bswap32(t->config.dim);
    t->config.hidden_dim = __builtin_bswap32(t->config.hidden_dim);
    t->config.n_layers = __builtin_bswap32(t->config.n_layers);
    t->config.n_heads = __builtin_bswap32(t->config.n_heads);
    t->config.n_kv_heads = __builtin_bswap32(t->config.n_kv_heads);
    int32_t original_vocab_size = t->config.vocab_size;
    if (original_vocab_size != 0) {
        t->config.vocab_size = __builtin_bswap32(abs(original_vocab_size));
        if (original_vocab_size < 0) {
            t->config.vocab_size = -t->config.vocab_size;
        }
    } else {
        t->config.vocab_size = 0;
    }
    t->config.seq_len = __builtin_bswap32(t->config.seq_len);
    // ULOGI("Performed byte-swap on config for big-endian system."); // Removed debugging log
  #endif

  // negative vocab size is hacky way of signaling unshared weights. bit yikes.
  int shared_weights = t->config.vocab_size > 0 ? 1 : 0;
  t->config.vocab_size = abs(t->config.vocab_size);
  status = ReadWeights(&t->weights, &t->config,
      &checkpoint_file, shared_weights);

cleanup:
  if (checkpoint_file.fd >= 0) {
      UllmFileClose(&checkpoint_file);
  }
  return status;
}

static UllmStatus UllmLlama2BuildTransformer(const UllmLlama2RunConfig* config,
    UllmLlama2State* state) {
  ULLM_RETURN_IF_ERROR(UllmLlama2ReadCheckpoint(config, state));
  if (state->transformer.config.vocab_size <= 0) {
      ULOGE("Invalid vocab_size in config: %d", state->transformer.config.vocab_size);
      return ULLM_STATUS_INVALID_ARGUMENT;
  }
  if (state->transformer.config.seq_len <= 0) {
      ULOGE("Invalid seq_len in config: %d", state->transformer.config.seq_len);
      return ULLM_STATUS_INVALID_ARGUMENT;
  }

  if (config->steps > (unsigned int)state->transformer.config.seq_len) {
    ULOGI("steps (%u) > seq_len (%" PRId32 "), model output will be truncated",
        config->steps, state->transformer.config.seq_len);
  }

  return UllmLlama2MallocRunState(&state->transformer);
}

static void UllmLlama2FreeTransformer(UllmLlama2Transformer* t) {
    if (t == NULL) return;

  if(t->weights.token_embedding_table) UllmMemoryFree(t->weights.token_embedding_table);
  if(t->weights.rms_att_weight) UllmMemoryFree(t->weights.rms_att_weight);
  if(t->weights.rms_ffn_weight) UllmMemoryFree(t->weights.rms_ffn_weight);
  if(t->weights.wq) UllmMemoryFree(t->weights.wq);
  if(t->weights.wk) UllmMemoryFree(t->weights.wk);
  if(t->weights.wv) UllmMemoryFree(t->weights.wv);
  if(t->weights.wo) UllmMemoryFree(t->weights.wo);
  if(t->weights.w1) UllmMemoryFree(t->weights.w1);
  if(t->weights.w2) UllmMemoryFree(t->weights.w2);
  if(t->weights.w3) UllmMemoryFree(t->weights.w3);
  if(t->weights.rms_final_weight) UllmMemoryFree(t->weights.rms_final_weight);
  if (t->weights.wcls != NULL && t->weights.wcls != t->weights.token_embedding_table) {
    UllmMemoryFree(t->weights.wcls);
  }

  if(t->state.x) UllmMemoryFree(t->state.x);
  if(t->state.xb) UllmMemoryFree(t->state.xb);
  if(t->state.xb2) UllmMemoryFree(t->state.xb2);
  if(t->state.hb) UllmMemoryFree(t->state.hb);
  if(t->state.hb2) UllmMemoryFree(t->state.hb2);
  if(t->state.q) UllmMemoryFree(t->state.q);
  if(t->state.att) UllmMemoryFree(t->state.att);
  if(t->state.logits) UllmMemoryFree(t->state.logits);
  if(t->state.key_cache) UllmMemoryFree(t->state.key_cache);
  if(t->state.value_cache) UllmMemoryFree(t->state.value_cache);

  memset(t, 0, sizeof(UllmLlama2Transformer));
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

float sumsquares(const float* x, int size) {
  float ss = 0.0f;
  if (x == NULL || size <= 0) return 0.0f;
  for (int j = 0; j < size; j++) {
      ss += x[j] * x[j];
  }
  return ss;
}


void rmsnorm(float* o, float* x, const float* weight, int size) {
    if (o == NULL || x == NULL || weight == NULL || size <= 0) return;
    float ss = sumsquares(x, size);
    if (size <= 0) return;
    ss /= size;
    ss += 1e-5f;
    if (ss <= 0.0f) ss = 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}


void softmax(float* x, int size) {
  if (size <= 0 || x == NULL) return;
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  if (sum == 0.0f || isnan(sum) || isinf(sum)) {
      for (int i = 0; i < size; i++) {
          x[i] = 1.0f / size;
      }
  } else {
      for (int i = 0; i < size; i++) {
          x[i] /= sum;
      }
  }
}


#ifdef ULLM_ALTIVEC_ENABLED

void matmul(float* xout, const float* x, const float* w, int n, int d) {
  if (xout == NULL || x == NULL || w == NULL || n <= 0 || d <= 0 || n % 4 != 0) {
      if (n % 4 != 0) ULOGE("AltiVec matmul requires n (%d) to be multiple of 4", n);
      else ULOGE("Invalid arguments to AltiVec matmul");
      memset(xout, 0, d * sizeof(float));
      return;
  }
  // W (d,n) @ x (n,) -> xout (d,)
  const long addr_step = 4 * sizeof(float);
  const long x_addr_end = (long)n * sizeof(float);

  long w_row_offset = 0;
  for (int i = 0; i < d; ++i) {
    __attribute__ ((aligned(16))) float psum[4] = {};
    vector float sum_vec = vec_splats(0.0f);
    for (long x_addr_offset = 0; x_addr_offset < x_addr_end; x_addr_offset += addr_step) {
      if ((w_row_offset + x_addr_offset + addr_step) > (long)d * (long)n * sizeof(float)) {
          ULOGE("Potential out-of-bounds read in AltiVec matmul (w)");
          break;
      }
      vector float w_vec = vec_ld(w_row_offset + x_addr_offset, w);
      vector float x_vec = vec_ld(x_addr_offset, x);
      sum_vec = vec_madd(w_vec, x_vec, sum_vec);
    }

    vec_st(sum_vec, 0, psum);
    xout[i] = psum[0] + psum[1] + psum[2] + psum[3];
    w_row_offset += n * sizeof(float);
  }
}


#else // Original matmul, kept for non-AltiVec builds

void matmul(float* xout, const float* x, const float* w, int n, int d) {
  if (xout == NULL || x == NULL || w == NULL || n <= 0 || d <= 0) return;
  // W (d,n) @ x (n,) -> xout (d,)
  // n: input dimension
  // d: output dimension
  for (int i = 0; i < d; i++) {
      float val = 0.0f;
      for (int j = 0; j < n; j++) {
          val += w[i * n + j] * x[j];
      }
      xout[i] = val;
  }
}

#endif

float* forward(UllmLlama2Transformer* transformer, int token, int pos) {
  if (transformer == NULL) return NULL;
  UllmLlama2Config* p = &transformer->config;
  UllmLlama2TransformerWeights* w = &transformer->weights;
  UllmLlama2RunState* s = &transformer->state;
  if (p == NULL || w == NULL || s == NULL || s->x == NULL || w->token_embedding_table == NULL || s->logits == NULL) {
      ULOGE("Transformer state not properly initialized in forward pass.");
      return NULL;
  }

  float *x = s->x;
  int dim = p->dim;
  if (p->n_heads <= 0) {
      ULOGE("n_heads is zero or negative in forward pass!");
      return NULL;
  }

  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  if (kv_dim <= 0 || (p->dim * p->n_kv_heads) % p->n_heads != 0) {
      ULOGE("Invalid kv_dim (%d) calculated in forward pass", kv_dim);
      return NULL;
  }

  int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  if (kv_mul <= 0 || p->n_heads % p->n_kv_heads != 0) {
      ULOGE("Invalid kv_mul (%d) calculated in forward pass", kv_mul);
      return NULL;
  }

  int hidden_dim =  p->hidden_dim;
  int head_size = dim / p->n_heads;
  if (head_size <= 0 || dim % p->n_heads != 0) {
      ULOGE("Invalid head_size (%d) calculated in forward pass", head_size);
      return NULL;
  }


  if (token < 0 || token >= p->vocab_size) {
      ULOGE("Invalid token index %d in forward pass (vocab_size %d)", token, p->vocab_size);
      return NULL;
  }
  const float* content_row = w->token_embedding_table + token * dim;
  memcpy(x, content_row, dim * sizeof(*x));

  for(unsigned long long l = 0; l < p->n_layers; l++) {
    if (w->rms_att_weight == NULL || w->wq == NULL || w->wk == NULL || w->wv == NULL || w->wo == NULL ||
        w->rms_ffn_weight == NULL || w->w1 == NULL || w->w2 == NULL || w->w3 == NULL) {
        ULOGE("NULL *weight* pointer detected in layer %llu", l);
        return NULL;
    }
    if (s->xb == NULL || /* s->k == NULL || s->v == NULL || */ s->q == NULL || s->att == NULL ||
        s->key_cache == NULL || s->value_cache == NULL || s->xb2 == NULL || s->hb == NULL || s->hb2 == NULL) {
        ULOGE("NULL *state* pointer detected in layer %llu", l);
        return NULL;
    }

    rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

    int loff = l * p->seq_len * kv_dim;
    if (pos < 0 || pos >= p->seq_len) {
        ULOGE("Position %d out of bounds for KV cache (seq_len %d)", pos, p->seq_len);
        return NULL;
    }
    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;

    matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
    matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
    matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    for (int i = 0; i < dim; i+=2) {
      int head_dim = (head_size > 0) ? (i % head_size) : 0;
      float freq = (head_size > 0) ? (1.0f / powf(10000.0f, head_dim / (float)head_size)) : 0.0f;
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      int rotn = i < kv_dim ? 2 : 1;
      for (int v_idx = 0; v_idx < rotn; v_idx++) {
        float* vec = v_idx == 0 ? s->q : s->k;
        if (i + 1 < dim) {
            float v0 = vec[i];
            float v1 = vec[i+1];
            vec[i]   = v0 * fcr - v1 * fci;
            vec[i+1] = v0 * fci + v1 * fcr;
        } else if (i < dim) {
            vec[i] = vec[i] * fcr;
        }
      }
    }

    // multihead attention
    int h;
    for (h = 0; h < p->n_heads; h++) {
      float* q = s->q + h * head_size;
      float* att = s->att + h * p->seq_len;
      for (int t = 0; t <= pos; t++) {
        if (t < 0 || t >= p->seq_len) {
            ULOGE("Timestep %d out of bounds during attention calc (seq_len %d)", t, p->seq_len);
            continue;
        }
        int kv_group_idx = (kv_mul > 0) ? (h / kv_mul) : 0;
        int kv_group_offset = kv_group_idx * head_size;
        size_t k_offset = (size_t)loff + (size_t)t * kv_dim + kv_group_offset;
        if (k_offset + head_size > (size_t)p->n_layers * p->seq_len * kv_dim) {
            ULOGE("Key cache index out of bounds at layer %llu, head %d, timestep %d", l, h, t);
            continue;
        }
        float* k = s->key_cache + k_offset;

        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
            score += q[i] * k[i];
        }
        if (head_size > 0) {
          score /= sqrtf((float)head_size);
        } else {
          score = 0.0f;
        }
        att[t] = score;
      }

      softmax(att, pos + 1);

      float* xb = s->xb + h * head_size;
      memset(xb, 0, head_size * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        if (t < 0 || t >= p->seq_len) continue;

        int kv_group_idx = (kv_mul > 0) ? (h / kv_mul) : 0;
        int kv_group_offset = kv_group_idx * head_size;
        size_t v_offset = (size_t)loff + (size_t)t * kv_dim + kv_group_offset;
        if (v_offset + head_size > (size_t)p->n_layers * p->seq_len * kv_dim) {
            ULOGE("Value cache index out of bounds at layer %llu, head %d, timestep %d", l, h, t);
            continue;
        }
        float* v = s->value_cache + v_offset;
        float a = att[t];
        for (int i = 0; i < head_size; i++) {
          xb[i] += a * v[i];
        }
      }
    }

    matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

    for (int i = 0; i < dim; i++) {
      x[i] += s->xb2[i];
    }

    rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

    matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
    matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

    // SwiGLU non-linearity
    for (int i = 0; i < hidden_dim; i++) {
      float val = s->hb[i];
      // silu(x)=x*sigma(x), where sigma(x) is the logistic sigmoid
      val *= (1.0f / (1.0f + expf(-val)));
      val *= s->hb2[i];
      s->hb[i] = val;
    }

    matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

    for (int i = 0; i < dim; i++) {
      x[i] += s->xb[i];
    }
  }

  if (w->rms_final_weight == NULL || w->wcls == NULL) {
        ULOGE("NULL weight pointer detected before final classification");
        return NULL;
  }

  rmsnorm(x, x, w->rms_final_weight, dim);

  matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
  return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

int compare_tokens(const void *a, const void *b) {
    if (a == NULL || b == NULL) return 0;
    UllmLlama2TokenIndex* token_a = (UllmLlama2TokenIndex*)a;
    UllmLlama2TokenIndex* token_b = (UllmLlama2TokenIndex*)b;
    if (token_a->str == NULL && token_b->str == NULL) return 0;
    if (token_a->str == NULL) return -1;
    if (token_b->str == NULL) return 1;
    return strcmp(token_a->str, token_b->str);
}

UllmStatus UllmLlama2BuildTokenizer(const UllmLlama2RunConfig* config,
    UllmLlama2State* state) {
  if (config == NULL || state == NULL) return ULLM_STATUS_INVALID_ARGUMENT;
  UllmLlama2Tokenizer* t = &state->tokenizer;
  const int32_t vocab_size = state->transformer.config.vocab_size;
  memset(t, 0, sizeof(UllmLlama2Tokenizer));

  if (vocab_size <= 0) {
      ULOGE("Invalid vocab_size (%d) for BuildTokenizer", vocab_size);
      return ULLM_STATUS_INVALID_ARGUMENT;
  }

  t->vocab = (char**)UllmMemoryAlloc(vocab_size * sizeof(char*));
  if (t->vocab == NULL) {
    ULOGE("Failed to allocate vocab buffer");
    return ULLM_STATUS_OOM;
  }
  memset(t->vocab, 0, vocab_size * sizeof(char*));


  t->vocab_scores = (float*)UllmMemoryAlloc(vocab_size * sizeof(float));
  if (t->vocab_scores == NULL) {
    ULOGE("Failed to allocate vocab_scores buffer");
    UllmMemoryFree(t->vocab);
    t->vocab = NULL;
    return ULLM_STATUS_OOM;
  }

  t->sorted_vocab = UllmMemoryAlloc(vocab_size * sizeof(UllmLlama2TokenIndex));
  if (t->sorted_vocab == NULL) {
    ULOGE("Failed to allocate sorted_vocab buffer");
    UllmMemoryFree(t->vocab);
    UllmMemoryFree(t->vocab_scores);
    t->vocab = NULL;
    t->vocab_scores = NULL;
    return ULLM_STATUS_OOM;
  }

  UllmFile tokenizer_file;
  tokenizer_file.fd = -1;
  UllmStatus status = ULLM_STATUS_OK;
  ULLM_GOTO_IF_ERROR(cleanup, status, UllmFileOpen(config->tokenizer_path, &tokenizer_file));


  uint32_t max_token_length = 0;

  ULLM_GOTO_IF_ERROR(cleanup, status, UllmFileRead(&tokenizer_file,
      &max_token_length, sizeof(uint32_t)));
  #if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
      max_token_length = __builtin_bswap32(max_token_length);
      // ULOGI("Performed byte-swap on max_token_length."); // Removed debugging log
  #endif
  if (max_token_length == 0 || max_token_length > 1024) {
      ULOGE("Invalid max_token_length read from tokenizer: %u", max_token_length);
      status = ULLM_STATUS_INVALID_ARGUMENT;
      goto cleanup;
  }

  size_t token_buffer_size = max_token_length * 2 + 1 + 1;
  t->token_buffer = UllmMemoryAlloc(token_buffer_size);
  if (t->token_buffer == NULL) {
    ULOGE("Failed to allocate token_buffer");
    t->token_buffer_capacity = 0;
    status = ULLM_STATUS_OOM;
    goto cleanup;
  }
  t->token_buffer_capacity = token_buffer_size;


  for (int i = 0; i < vocab_size; i++) {
    float score_val;
    ULLM_GOTO_IF_ERROR(cleanup, status, UllmFileRead(&tokenizer_file,
        &score_val, sizeof(float)));
    #if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
        uint32_t* score_ptr = (uint32_t*)&score_val;
        *score_ptr = __builtin_bswap32(*score_ptr);
    #endif
    t->vocab_scores[i] = score_val;

    uint32_t len;
    ULLM_GOTO_IF_ERROR(cleanup, status, UllmFileRead(&tokenizer_file,
        &len, sizeof(uint32_t)));
    #if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
        len = __builtin_bswap32(len);
    #endif

    if (len > max_token_length) {
        ULOGE("Token length %u exceeds max_token_length %u at index %d", len, max_token_length, i);
        status = ULLM_STATUS_INVALID_ARGUMENT;
        goto cleanup;
    }

    t->vocab[i] = (char *)UllmMemoryAlloc(len + 1);
    if (t->vocab[i] == NULL) {
      ULOGE("Failed to alloc vocab memory for index %d", i);
      status = ULLM_STATUS_OOM;
      goto cleanup;
    }

    ULLM_GOTO_IF_ERROR(cleanup, status, UllmFileRead(&tokenizer_file,
        t->vocab[i], len));
    t->vocab[i][len] = '\0'; // add the string terminating token
  }

  for (int i = 0; i < vocab_size; i++) {
    t->sorted_vocab[i].str = t->vocab[i];
    t->sorted_vocab[i].id = i;
  }
  qsort(t->sorted_vocab, vocab_size, sizeof(UllmLlama2TokenIndex), compare_tokens);

cleanup:
  if (tokenizer_file.fd >= 0) {
    UllmFileClose(&tokenizer_file);
  }
  if (status != ULLM_STATUS_OK) {
      UllmLlama2FreeTokenizer(state);
  }
  return status;
}


static void UllmLlama2FreeTokenizer(UllmLlama2State* state) {
  if (state == NULL) return;
  UllmLlama2Tokenizer* t = &state->tokenizer;
  if (t == NULL) return;

  if (t->vocab) {
    int vocab_size_to_free = (state->transformer.config.vocab_size > 0) ? state->transformer.config.vocab_size : 0;
    for (int i = 0; i < vocab_size_to_free; i++) {
        if(t->vocab[i]) UllmMemoryFree(t->vocab[i]);
    }
    UllmMemoryFree(t->vocab);
    t->vocab = NULL;
  }
  if (t->vocab_scores) {
      UllmMemoryFree(t->vocab_scores);
      t->vocab_scores = NULL;
  }
  if (t->sorted_vocab) {
      UllmMemoryFree(t->sorted_vocab);
      t->sorted_vocab = NULL;
  }
  if (t->token_buffer) {
      UllmMemoryFree(t->token_buffer);
      t->token_buffer = NULL;
  }
  t->token_buffer_capacity = 0;
}


static void UllmLlama2EmitPiece(const UllmLlama2RunConfig* config,
    const char *piece) {
  if (config == NULL || config->output_callback == NULL || piece == NULL || piece[0] == '\0'
      || (piece[1] == '\0' && !isprint((unsigned char)piece[0]) && !isspace((unsigned char)piece[0]))) {
    return;
  }

  config->output_callback(piece, config->cookie);
}

static void UllmLlama2Decode(const UllmLlama2RunConfig* config,
                            UllmLlama2State* state,
                            UllmLlama2Tokenizer* t,
                            int prev_token, int token) {
  if (state == NULL || t == NULL || t->vocab == NULL) {
      ULOGE("Invalid state or tokenizer in UllmLlama2Decode");
      return;
  }

  if (token < 0 || token >= state->transformer.config.vocab_size) {
      ULOGE("Token index %d out of bounds (vocab_size %d)", token, state->transformer.config.vocab_size);
      return;
  }
  const char *piece = t->vocab[token];
  if (piece == NULL) {
      ULOGE("Vocab entry for token %d is NULL", token);
      return;
  }

  // following BOS (1) token strip leading whitespace
  if (prev_token == 1 && piece[0] == ' ') {
    piece++;
  }

  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    char byte_piece[2] = {0};
    byte_piece[0] = (char)byte_val;
    UllmLlama2EmitPiece(config, byte_piece);
  } else {
    UllmLlama2EmitPiece(config, piece);
  }
}


int str_lookup(const char *str, UllmLlama2TokenIndex *sorted_vocab, int vocab_size) {
  // efficiently find the perfect match for str in vocab, return its index or -1 if not found
  if (str == NULL || sorted_vocab == NULL || vocab_size <= 0) return -1;
  UllmLlama2TokenIndex tok = { .str = str };
  UllmLlama2TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(UllmLlama2TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}

static void UllmLlama2Encode(const UllmLlama2RunConfig* config,
    UllmLlama2State* state, int8_t bos, int8_t eos, int *tokens,
    int *n_tokens) {
  if (config == NULL || state == NULL || tokens == NULL || n_tokens == NULL) {
      if (n_tokens) *n_tokens = 0;
      return;
  }

  // encode the string text (input) into an upper-bound preallocated tokens[] array
  // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
  int32_t vocab_size = state->transformer.config.vocab_size;
  UllmLlama2Tokenizer* t = &state->tokenizer;
  size_t str_len = 0;
  size_t token_buffer_cap = state->tokenizer.token_buffer_capacity;
  if (t == NULL || t->vocab == NULL || t->sorted_vocab == NULL || t->token_buffer == NULL || token_buffer_cap == 0 || vocab_size <= 0) {
      ULOGE("Tokenizer not properly initialized for encoding.");
      *n_tokens = 0;
      return;
  }

  *n_tokens = 0;

  if (bos) {
    if (1 < vocab_size) {
        tokens[(*n_tokens)++] = 1;
    } else {
        ULOGE("Vocab size too small (%d) to include BOS token.", vocab_size);
    }
  }

  // add_dummy_prefix is true by default
  if (config->prompt != NULL && config->prompt[0] != '\0') {
    int dummy_prefix = str_lookup(" ", t->sorted_vocab, vocab_size);
    if (dummy_prefix != -1) {
        tokens[(*n_tokens)++] = dummy_prefix;
    } else {
        ULOGI("Dummy prefix space token not found in vocab");
    }
  }

  // process the raw (UTF-8) byte sequence of the input string
  if (config->prompt != NULL) {
      for (const char *c = config->prompt; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
          str_len = 0;
        }

        if (str_len < token_buffer_cap - 1) {
            t->token_buffer[str_len++] = *c;
            t->token_buffer[str_len] = '\0';
        } else {
            ULOGI("Token buffer overflow potential during encoding, character might be split.");
            str_len = 0;
            if ((*c & 0xC0) != 0x80 && *c >= 0) {
                if ((unsigned char)(*c) + 3 < vocab_size) {
                    tokens[(*n_tokens)++] = (unsigned char)(*c) + 3;
                } else {
                    ULOGE("Byte fallback resulted in invalid token index for char '%c'", *c);
                    tokens[(*n_tokens)++] = 0; // UNK token
                }
            }
            continue;
        }

        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
          continue;
        }

        int id = str_lookup(t->token_buffer, t->sorted_vocab, vocab_size);

        if (id != -1) {
          tokens[(*n_tokens)++] = id;
        } else {
          // byte_fallback encoding
          for (int i=0; i < str_len; i++) {
            if ((unsigned char)t->token_buffer[i] + 3 < vocab_size) {
                tokens[(*n_tokens)++] = (unsigned char)t->token_buffer[i] + 3;
            } else {
                ULOGE("Byte fallback resulted in invalid token index for byte 0x%02x", (unsigned char)t->token_buffer[i]);
                tokens[(*n_tokens)++] = 0; // UNK token
            }
          }
        }
        str_len = 0;
      }
  }

  // merge the best consecutive pair each iteration
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;

    for (int i=0; i < (*n_tokens-1); i++) {
      if (tokens[i] < 0 || tokens[i] >= vocab_size || tokens[i+1] < 0 || tokens[i+1] >= vocab_size) {
          ULOGE("Invalid token index during merge check at pos %d", i);
          continue;
      }
      if (t->vocab[tokens[i]] == NULL || t->vocab[tokens[i+1]] == NULL) {
          ULOGE("NULL vocab entry during merge check at pos %d", i);
          continue;
      }

      snprintf(t->token_buffer, token_buffer_cap, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
      t->token_buffer[token_buffer_cap - 1] = '\0';

      int id = str_lookup(t->token_buffer, t->sorted_vocab, vocab_size);
      if (id != -1 && id < vocab_size && t->vocab_scores != NULL && t->vocab_scores[id] > best_score) {
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    if (best_idx == -1) {
      break;
    }

    tokens[best_idx] = best_id;
    for (int i = best_idx+1; i < (*n_tokens-1); i++) {
      tokens[i] = tokens[i+1];
    }
    if (*n_tokens > 0) {
      (*n_tokens)--;
    }
  }

  if (eos) {
    if (2 < vocab_size) {
        tokens[(*n_tokens)++] = 2;
    } else {
        ULOGE("Vocab size too small (%d) to include EOS token.", vocab_size);
    }
  }
}


// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    if (n <= 0 || probabilities == NULL) return -1;
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    if (n <= 0 || probabilities == NULL) return -1;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare_prob_index(const void* a, const void* b) {
    if (a == NULL || b == NULL) return 0;
    UllmLlama2ProbIndex* a_ = (UllmLlama2ProbIndex*) a;
    UllmLlama2ProbIndex* b_ = (UllmLlama2ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, UllmLlama2ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling")
    if (n <= 0 || probabilities == NULL || probindex == NULL) return -1;

    int n0 = 0;
    const float cutoff = (topp > 0.0f && topp < 1.0f && n > 1) ? (1.0f - topp) / (n - 1) : 0.0f;
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            if (n0 < n) {
              probindex[n0].index = i;
              probindex[n0].prob = probabilities[i];
              n0++;
            } else {
              ULOGI("Probindex buffer full during top-p filtering, possible issue.");
              break;
            }
        }
    }

    if (n0 == 0) {
        int max_idx = sample_argmax(probabilities, n);
        return max_idx;
    }

    qsort(probindex, n0, sizeof(UllmLlama2ProbIndex), compare_prob_index);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    if (last_idx < 0) last_idx = 0;
    if (last_idx >= n0) last_idx = n0 - 1;

    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}


static UllmStatus UllmLlama2BuildSampler(const UllmLlama2RunConfig* config,
    UllmLlama2State* state) {
  if (config == NULL || state == NULL) return ULLM_STATUS_INVALID_ARGUMENT;
  UllmLlama2Sampler* sampler = &state->sampler;
  memset(sampler, 0, sizeof(UllmLlama2Sampler));
  sampler->rng_state = config->rng_seed;
  if (state->transformer.config.vocab_size <= 0) {
      ULOGE("Cannot build sampler with invalid vocab_size: %d", state->transformer.config.vocab_size);
      return ULLM_STATUS_INVALID_ARGUMENT;
  }
  sampler->probindex = UllmMemoryAlloc(state->transformer.config.vocab_size
      * sizeof(UllmLlama2ProbIndex));
  if (sampler->probindex == NULL) {
    ULOGE("Failed to allocate probindex");
    return ULLM_STATUS_OOM;
  }

  return ULLM_STATUS_OK;
}

static void UllmLlama2FreeSampler(UllmLlama2Sampler* sampler) {
    if(sampler && sampler->probindex) {
      UllmMemoryFree(sampler->probindex);
      sampler->probindex = NULL;
    }
}

uint32_t random_u32(uint64_t *state) {
  // xorshift rng
  if (state == NULL) return 0;
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(uint64_t *state) { // random float32 in [0,1)
  if (state == NULL) return 0.0f;
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(const UllmLlama2RunConfig* config, UllmLlama2State* state, float* logits) {
  if (config == NULL || state == NULL || logits == NULL) {
      ULOGE("NULL argument passed to sample function");
      return 2; // Return EOS
  }
  UllmLlama2Sampler* sampler = &state->sampler;
  const int32_t vocab_size = state->transformer.config.vocab_size;
  int next;

  if (sampler == NULL || sampler->probindex == NULL || vocab_size <= 0) {
      ULOGE("Sampler not properly initialized or invalid vocab size.");
      return 2; // Return EOS
  }


  if (config->temperature == 0.0f) {
    next = sample_argmax(logits, vocab_size);
  } else {
    if (config->temperature != 0.0f) {
        for (int q=0; q < vocab_size; q++) { logits[q] /= config->temperature; }
    }
    softmax(logits, vocab_size);
    float coin = random_f32(&sampler->rng_state);
    if (config->topp <= 0 || config->topp >= 1) {
      next = sample_mult(logits, vocab_size, coin);
    } else {
      next = sample_topp(logits, vocab_size, config->topp, sampler->probindex, coin);
    }
  }
  if (next < 0 || next >= vocab_size) {
      ULOGE("Sampling produced invalid token index %d", next);
      return 2; // Return EOS
  }
  return next;
}

static UllmStatus UllmLlama2ValidateConfig(const UllmLlama2RunConfig* config) {
  if (config == NULL) {
      ULOGE("config must not be NULL");
      return ULLM_STATUS_INVALID_ARGUMENT;
  }

  if (config->temperature < 0.0) {
    ULOGE("temperature must not be negative");
    return ULLM_STATUS_INVALID_ARGUMENT;
  }

  if (config->topp < 0.0f || config->topp > 1.0f) {
    ULOGE("topp must be between 0.0f and 1.0f");
    return ULLM_STATUS_INVALID_ARGUMENT;
  }

  if (config->steps == 0) {
    ULOGE("steps must be greater than 0");
    return ULLM_STATUS_INVALID_ARGUMENT;
  }

  if (config->checkpoint_path == NULL || strlen(config->checkpoint_path) == 0) {
      ULOGE("checkpoint_path must be provided");
      return ULLM_STATUS_INVALID_ARGUMENT;
  }

  if (config->tokenizer_path == NULL || strlen(config->tokenizer_path) == 0) {
      ULOGE("tokenizer_path must be provided");
      return ULLM_STATUS_INVALID_ARGUMENT;
  }


  return ULLM_STATUS_OK;
}

// ----------------------------------------------------------------------------
// Public API

void UllmLlama2RunConfigInit(UllmLlama2RunConfig* config) {
  if (config == NULL) return;
  memset(config, 0, sizeof(UllmLlama2RunConfig));
  config->temperature = 1.0f;
  config->topp = 0.9f;
  config->steps = 256;
  config->rng_seed = (uint64_t)time(NULL);
  if (config->rng_seed == (uint64_t)-1) config->rng_seed = 1234;
}

UllmStatus UllmLlama2Init(const UllmLlama2RunConfig* config,
    UllmLlama2State* state) {
  if (state == NULL) return ULLM_STATUS_INVALID_ARGUMENT;
  memset(state, 0, sizeof(UllmLlama2State));

  ULLM_RETURN_IF_ERROR(UllmLlama2ValidateConfig(config));
  ULLM_RETURN_IF_ERROR(UllmLlama2BuildTransformer(config, state));

  UllmStatus status = UllmLlama2BuildTokenizer(config, state);
  if (status != ULLM_STATUS_OK) {
      UllmLlama2FreeTransformer(&state->transformer);
      return status;
  }

  status = UllmLlama2BuildSampler(config, state);
  if (status != ULLM_STATUS_OK) {
      UllmLlama2FreeTokenizer(state);
      UllmLlama2FreeTransformer(&state->transformer);
      return status;
  }

  return ULLM_STATUS_OK;
}


UllmStatus UllmLlama2Generate(const UllmLlama2RunConfig* config,
    UllmLlama2State* state) {
  UllmStatus status = ULLM_STATUS_OK;
  int* prompt_tokens = NULL;
  int num_prompt_tokens = 0;

  if (config == NULL || state == NULL) {
      return ULLM_STATUS_INVALID_ARGUMENT;
  }
  if (state->tokenizer.vocab == NULL || state->sampler.probindex == NULL || state->transformer.config.seq_len <= 0) {
      ULOGE("Llama2 state not fully initialized.");
      return ULLM_STATUS_INVALID_ARGUMENT;
  }

  size_t prompt_len = (config->prompt != NULL) ? strlen(config->prompt) : 0;
  size_t max_possible_tokens = prompt_len + 3;
  size_t prompt_tokens_alloc_size = max_possible_tokens * sizeof(int);
  prompt_tokens = UllmMemoryAlloc(prompt_tokens_alloc_size);
  if (prompt_tokens == NULL) {
    ULOGE("Failed to allocate prompt tokens");
    return ULLM_STATUS_OOM;
  }

  uint64_t start_time_ns = UllmTimeNanos();

  UllmLlama2Encode(config, state, 1, 0, prompt_tokens, &num_prompt_tokens);

  if (num_prompt_tokens <= 0) {
    ULOGE("Prompt encoding failed or resulted in zero tokens.");
    status = ULLM_STATUS_INVALID_ARGUMENT;
    goto gen_cleanup;
  }

  int next = -1;
  int token = prompt_tokens[0];
  unsigned int pos = 0;
  unsigned int max_steps = (config->steps < (unsigned int)state->transformer.config.seq_len) ? config->steps : (unsigned int)state->transformer.config.seq_len;

  while (pos < max_steps) {
    float* logits = forward(&state->transformer, token, pos);
    if (logits == NULL) {
        ULOGE("Forward pass returned NULL logits at pos %u", pos);
        status = ULLM_STATUS_OOM;
        goto gen_cleanup;
    }

    if (pos < num_prompt_tokens - 1) {
      next = prompt_tokens[pos + 1];
    } else {
      next = sample(config, state, logits);
    }
    pos++;

    if (next == 2) { // EOS token
        break;
    }
    if (next < 0 || next >= state->transformer.config.vocab_size) {
        ULOGE("Sampler returned invalid token index %d at pos %u", next, pos -1);
        status = ULLM_STATUS_INVALID_ARGUMENT;
        goto gen_cleanup;
    }

    UllmLlama2Decode(config, state, &state->tokenizer, token, next);
    token = next;
  }

  UllmLlama2EmitPiece(config, "\n");

  int prompt_tokens_consumed = (num_prompt_tokens > 0) ? num_prompt_tokens -1 : 0;
  int generated_tokens = (pos > (unsigned int)prompt_tokens_consumed) ? pos - prompt_tokens_consumed : 0;


  if (generated_tokens > 0) {
    uint64_t end_time_ns = UllmTimeNanos();
    double elapsed_s = (end_time_ns - start_time_ns) / 1000000000.0;

    if (elapsed_s > 0) {
      double token_rate = generated_tokens / elapsed_s;
      ULOGI("Complete: %d tokens / %.2fs = %.2f token/s", generated_tokens, elapsed_s, token_rate);
    } else {
        ULOGI("Complete: %d tokens / <0.01s", generated_tokens);
    }
  } else {
    ULOGI("Complete: No tokens generated after prompt.");
  }

gen_cleanup:
  if (prompt_tokens) UllmMemoryFree(prompt_tokens);
  return status;
}

void UllmLlama2Deinit(UllmLlama2State* state) {
  if (state == NULL) return;
  UllmLlama2FreeSampler(&state->sampler);
  UllmLlama2FreeTokenizer(state);
  UllmLlama2FreeTransformer(&state->transformer);
}
