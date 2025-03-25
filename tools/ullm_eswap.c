/*
 * Copyright 2025 Andrew Rossignol andrew.rossignol@gmail.com
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <string.h>

#include "c-flags.h"
#include "util/log.h"
#include "sys/file.h"

#define ULLM_LOG_TAG "ullm.eswap"

int main(int argc, char** argv) {
  if (argc > 0) {
    c_flags_set_application_name(argv[0]);
  }

  char** checkpoint_path = c_flag_string(
      "checkpoint_path", "c", "checkpoint path", "");
  char** tokenizer_path = c_flag_string(
      "tokenizer_path", "t", "tokenizer path", "");
  bool *help = c_flag_bool(
      "help", "h", "show usage", false);
  c_flags_parse(&argc, &argv, false);

  UllmFile file;
  UllmStatus status = ULLM_STATUS_OK;
  if (*help) {
    c_flags_usage();
    return -1;
  } else if (strlen(*checkpoint_path) > 0) {
    status = UllmFileOpen(*checkpoint_path, &file);
    if (status != ULLM_STATUS_OK) {
      ULOGE("Failed to map checkpoint: %s", UllmStatusToString(status));
      goto cleanup;
    }

    if ((file.size % 4) != 0) {
      ULOGE("Checkpoint size must be a multiple of 4");
      status = ULLM_STATUS_INVALID_ARGUMENT;
      goto cleanup;
    }

    for (uint64_t i = 0; i < file.size; i+= sizeof(uint32_t)) {
      uint32_t le_value = 0;
      ULLM_GOTO_IF_ERROR(cleanup, status, UllmFileRead(&file,
          &le_value, sizeof(uint32_t)));
      const uint32_t be_value = __builtin_bswap32(le_value);
      fwrite(&be_value, sizeof(uint32_t), 1, stdout);
    }
  } else if (strlen(*tokenizer_path) > 0) {
    status = UllmFileOpen(*tokenizer_path, &file);
    if (status != ULLM_STATUS_OK) {
      ULOGE("Failed to map checkpoint: %s", UllmStatusToString(status));
      goto cleanup;
    }

    uint32_t le_value = 0;
    uint64_t offset = 0;
    ULLM_GOTO_IF_ERROR(cleanup, status, UllmFileRead(&file,
        &le_value, sizeof(uint32_t)));
    offset += sizeof(uint32_t);
    uint32_t be_value = __builtin_bswap32(le_value);
    fwrite(&be_value, sizeof(uint32_t), 1, stdout);

    while (offset < file.size) {
      ULLM_GOTO_IF_ERROR(cleanup, status, UllmFileRead(&file,
          &le_value, sizeof(uint32_t)));
      offset += sizeof(uint32_t);
      be_value = __builtin_bswap32(le_value);
      fwrite(&be_value, sizeof(uint32_t), 1, stdout);

      ULLM_GOTO_IF_ERROR(cleanup, status, UllmFileRead(&file,
          &le_value, sizeof(uint32_t)));
      offset += sizeof(uint32_t);
      be_value = __builtin_bswap32(le_value);
      fwrite(&be_value, sizeof(uint32_t), 1, stdout);
      for (uint32_t i = 0; i < le_value; i++) {
        char c;
        ULLM_GOTO_IF_ERROR(cleanup, status, UllmFileRead(&file, &c, sizeof(c)));
        offset += sizeof(char);
        fwrite(&c, sizeof(char), 1, stdout);
      }
    }
  } else {
    c_flags_usage();
    return -1;
  }

cleanup:
  UllmFileClose(&file);
  return status == ULLM_STATUS_OK ? 0 : -1;
}
