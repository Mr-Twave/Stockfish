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

#ifndef THREAD_H_INCLUDED
#define THREAD_H_INCLUDED

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "numa.h"
#include "position.h"
#include "search.h"
#include "types.h"

namespace Stockfish {

class OptionsMap;

// The Thread class encapsulates a single thread of execution.
class Thread {

    std::mutex              mutex;
    std::condition_variable cv;
    size_t                  idx;
    bool                    exit = false, searching = true;
    NativeThread            stdThread;
    std::function<void()>   jobFunc;

   public:
    explicit Thread(Search::SharedState&, std::unique_ptr<Search::ISearchManager>, 
                    size_t, OptionalThreadToNumaNodeBinder);
    virtual ~Thread();

    void start_searching();
    void clear_worker();
    void wait_for_search_finished();
    void run_custom_job(std::function<void()>);
    void ensure_network_replicated();

    size_t                      nthreads;
    std::unique_ptr<Search::Worker> worker;
    NumaReplicatedAccessToken   numaAccessToken;

   private:
    void idle_loop();
};


// The ThreadPool class manages all threads.
class ThreadPool {

   public:
    explicit ThreadPool(Search::SharedState& ss);

    void start_thinking(const OptionsMap&, Position&, StateListPtr&, Search::LimitsType);
    void clear();
    // Note: Takes SharedState by reference via the constructor because it now
    // owns non-copyable TT resources in NUMA builds.
    void set(const NumaConfig&, const Search::SearchManager::UpdateContext&);

    Search::SearchManager* main_manager();
    Thread* main_thread() const { return threads.front().get(); }
    uint64_t               nodes_searched() const;
    uint64_t               tb_hits() const;
    void                   start_searching();
    void                   wait_for_search_finished() const;
    std::vector<size_t>    get_bound_thread_count_by_numa_node() const;
    void                   ensure_network_replicated();

    std::atomic_bool stop, abortedSearch, increaseDepth;

    auto cbegin() const noexcept { return threads.cbegin(); }
    auto cend() const noexcept { return threads.cend(); }
    auto begin() noexcept { return threads.begin(); }
    auto end() noexcept { return threads.end(); }
    size_t size() const noexcept { return threads.size(); }
    bool   empty() const noexcept { return threads.empty(); }

    Thread* operator[](size_t i) { return threads[i].get(); }
    const Thread* operator[](size_t i) const { return threads[i].get(); }

    void run_on_thread(size_t threadId, std::function<void()>);
    void wait_on_thread(size_t threadId);
    size_t num_threads() const;

#ifdef USE_NUMA_TT
    // This atomic is now managed by the ThreadPool via SharedState
    // for NUMA builds, ensuring a single source of truth for entry age.
    std::atomic<uint8_t>& generation8;
#endif

   private:
    StateListPtr                       setupStates;
    std::vector<std::unique_ptr<Thread>> threads;
    std::vector<NumaIndex>               boundThreadToNumaNode;
    Search::SharedState&                 sharedState; // Holds TTs and other shared resources

    template<typename T>
    uint64_t accumulate(std::atomic<T> Search::Worker::*member) const {
        uint64_t sum = 0;
        for (auto&& th : threads)
            sum += (th->worker.get()->*member).load(std::memory_order_relaxed);
        return sum;
    }
};

extern ThreadPool Threads;

}  // namespace Stockfish

#endif  // #ifndef THREAD_H_INCLUDED
