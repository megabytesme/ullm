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

#ifndef ULLM_SYS_FILE_H_
#define ULLM_SYS_FILE_H_

#include <stddef.h>
#include <stdint.h>

#include "util/status.h"

#ifdef __cplusplus
extern "C" {
#endif

// A forward declaration of a file handle type.
typedef struct UllmFileHandle UllmFileHandle;

// Opens the supplied file, maps into memory and populates the size or
// returns an error.
UllmStatus UllmFileMap(const char* path, UllmFileHandle** handle,
    const char** ptr, uint64_t* size);

// Unmaps a file and closes it. This invlidates any pointers to file contents.
void UllmFileUnmap(UllmFileHandle* handle);

// Reads the supplied size into a destination buffer.
UllmStatus UllmFileRead(const UllmFileHandle* file, uint64_t* offset,
    void* dst, uint64_t size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ULLM_SYS_FILE_H_
