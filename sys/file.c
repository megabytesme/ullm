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

#include "sys/file.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sys/memory.h"
#include "util/log.h"

#define ULLM_LOG_TAG "ullm.file"

typedef struct UllmFileHandle {
  int fd;
  const char* ptr;
  uint64_t size;
} UllmFileHandle;

UllmStatus UllmFileMap(const char* path, UllmFileHandle** handle,
    const char** ptr, uint64_t* size) {
  *handle = UllmMemoryAlloc(sizeof(UllmFileHandle));
  memset(*handle, 0, sizeof(UllmFileHandle));

  struct stat st;
  if (stat(path, &st) != 0) {
    ULOGE("Failed to stat file '%s': %s (%d)", path, strerror(errno), errno);
    return ULLM_STATUS_IO_ERROR;
  }

  (*handle)->fd = open(path, O_RDONLY);
  if ((*handle)->fd < 0) {
    ULOGE("Failed to open file '%s': %s (%d)", path, strerror(errno), errno);
    return ULLM_STATUS_IO_ERROR;
  }

  *size = st.st_size;
  *ptr = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, (*handle)->fd, 0);
  if (*ptr == MAP_FAILED) {
    ULOGE("Failed to mmap '%s': %s (%d)", path, strerror(errno), errno);
    return ULLM_STATUS_IO_ERROR;
  }

  (*handle)->ptr = *ptr;
  (*handle)->size = st.st_size;
  ULOGI("Mapped file '%s' with size %" PRIu64 ", handle %p",
      path, (*handle)->size, *handle);
  return ULLM_STATUS_OK;
}

void UllmFileUnmap(UllmFileHandle* handle) {
  if (handle == NULL) {
    return;
  }

  if (handle->ptr != NULL) {
    int status = munmap((void*)handle->ptr, handle->size);
    if (status != 0) {
      ULOGE("Failed to unmap file: %s (%d), handle %p",
          strerror(errno), errno, handle);
    } else {
      ULOGI("Unmapped file, handle %p", handle);
    }
  }

  if (handle->fd >= 0) {
    close(handle->fd);
  }

  UllmMemoryFree(handle);
}

UllmStatus UllmFileRead(const UllmFileHandle* file, uint64_t* offset,
    void* dst, uint64_t size) {
  if ((file->size - *offset) < size) {
    ULOGE("File truncated when reading %" PRIu64 " from offset %" PRIu64,
        size, *offset);
    return ULLM_STATUS_INVALID_ARGUMENT;
  }

  memcpy(dst, &file->ptr[*offset], size);
  *offset = *offset + size;
  return ULLM_STATUS_OK;
}
