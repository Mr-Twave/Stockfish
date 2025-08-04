
/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef MEMORY_H_INCLUDED
#define MEMORY_H_INCLUDED

#include "types.h"
#include <cstddef>

namespace Stockfish {

#ifdef USE_NUMA_TT
using NumaNodeID = int;
constexpr NumaNodeID NumaNodeIDAll = -1;
#endif

void* std_aligned_alloc(size_t alignment, size_t size);

#ifdef USE_NUMA_TT
void* aligned_large_pages_alloc(size_t allocSize, NumaNodeID node);
#else
void* aligned_large_pages_alloc(size_t allocSize);
#endif

void std_aligned_free(void* ptr);
void aligned_large_pages_free(void* mem);
bool has_large_pages();

}  // namespace Stockfish

#endif  // #ifndef MEMORY_H_INCLUDED
