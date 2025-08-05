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

#include "search.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <list>
#include <ratio>
#include <string>
#include <utility>
#include <immintrin.h>

#include "bitboard.h"
#include "evaluate.h"
#include "history.h"
#include "misc.h"
#include "movegen.h"
#include "movepick.h"
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"
#include "position.h"
#include "syzygy/tbprobe.h"
#include "thread.h"
#include "timeman.h"
#include "tt.h"
#include "uci.h"
#include "ucioption.h"
#include "searchpolicies.h"

namespace Stockfish {

namespace TB = Tablebases;

using namespace Search;

namespace {

// Constants and type definitions
constexpr int SEARCHEDLIST_CAPACITY = 32;
constexpr int TT_CACHE_SIZE = 4;
constexpr int MOVE_BATCH_SIZE = 256;
using SearchedList = ValueList<Move, SEARCHEDLIST_CAPACITY>;

// Forward declarations
void syzygy_extend_pv(const OptionsMap&, const Search::LimitsType&, 
                      Position&, Search::RootMove&, Value&);

// Optimized Unified TT Interface with multi-entry cache
template<size_t CacheSize = TT_CACHE_SIZE>
class OptimizedTTInterface {
private:
    struct CacheEntry {
        Key key = 0;
        TTData data{Move::none(), VALUE_NONE, VALUE_NONE, DEPTH_NONE, BOUND_NONE, false};
        uint32_t timestamp = 0;
        std::atomic<bool> valid{false};
    };
    
    alignas(64) mutable CacheEntry cache[CacheSize];
    mutable std::atomic<uint32_t> globalTimestamp{0};
    
#ifdef USE_NUMA_TT
    TranspositionTable* l1_tt;
    TranspositionTable* l2_tt;
    std::atomic<uint8_t>& generation8;
#else
    TranspositionTable& tt;
#endif

public:
#ifdef USE_NUMA_TT
    OptimizedTTInterface(TranspositionTable* l1, TranspositionTable* l2, 
                        std::atomic<uint8_t>& gen) 
        : l1_tt(l1), l2_tt(l2), generation8(gen) {}
#else
    OptimizedTTInterface(TranspositionTable& table) : tt(table) {}
#endif

    // O(1) probe with linear probing cache
    std::tuple<bool, TTData, TTWriter> probe(Key key) const {
        uint32_t cacheIdx = (key >> 32) % CacheSize;
        
        // Check cache first
        for (size_t i = 0; i < CacheSize; ++i) {
            size_t idx = (cacheIdx + i) % CacheSize;
            if (cache[idx].valid.load(std::memory_order_acquire) && 
                cache[idx].key == key) {
                cache[idx].timestamp = globalTimestamp.fetch_add(1, std::memory_order_relaxed);
                return {true, cache[idx].data, TTWriter(nullptr)};
            }
        }
        
#ifdef USE_NUMA_TT
        // L1 probe with single prefetch
        if (l1_tt) {
            __builtin_prefetch(l1_tt->first_entry(key), 0, 3);
            auto l1Result = l1_tt->probe(key);
            if (l1Result.hit) {
                update_cache(key, l1Result.data);
                return {true, l1Result.data, l1Result.writer};
            }
            
            // L2 fallback with victim promotion
            if (l2_tt) {
                auto l2Result = l2_tt->probe(key);
                if (l2Result.hit) {
                    // Promote valuable entries to L1
                    if (l2Result.data.depth >= 8 && l2Result.data.is_pv) {
                        l1Result.writer.write(key, l2Result.data.value, l2Result.data.is_pv,
                                            l2Result.data.bound, l2Result.data.depth,
                                            l2Result.data.move, l2Result.data.eval, generation8);
                    }
                    update_cache(key, l2Result.data);
                    return {true, l2Result.data, l1Result.writer};
                }
            }
        }
        return {false, TTData(Move::none(), VALUE_NONE, VALUE_NONE, DEPTH_NONE, BOUND_NONE, false),
                TTWriter(nullptr)};
#else
        auto result = tt.probe(key);
        if (result.hit) {
            update_cache(key, result.data);
        }
        return {result.hit, result.data, result.writer};
#endif
    }

    uint8_t generation() const {
#ifdef USE_NUMA_TT
        return generation8.load(std::memory_order_relaxed);
#else
        return tt.generation();
#endif
    }

private:
    void update_cache(Key key, const TTData& data) const {
        // Find LRU entry
        uint32_t minTimestamp = UINT32_MAX;
        size_t lruIdx = 0;
        uint32_t cacheIdx = (key >> 32) % CacheSize;
        
        for (size_t i = 0; i < CacheSize; ++i) {
            size_t idx = (cacheIdx + i) % CacheSize;
            if (!cache[idx].valid.load(std::memory_order_acquire) ||
                cache[idx].timestamp < minTimestamp) {
                minTimestamp = cache[idx].timestamp;
                lruIdx = idx;
            }
        }
        
        cache[lruIdx].key = key;
        cache[lruIdx].data = data;
        cache[lruIdx].timestamp = globalTimestamp.fetch_add(1, std::memory_order_relaxed);
        cache[lruIdx].valid.store(true, std::memory_order_release);
    }
};

// Batched Move Processor for cache-efficient move annotation
class BatchedMoveProcessor {
private:
    struct AnnotatedMove {
        Move move;
        int16_t see_score;
        int16_t history_score;
        uint8_t flags;
        
        enum Flags : uint8_t {
            CAPTURE = 1,
            CHECK = 2,
            PRUNED = 4,
            TACTICAL = 8
        };
    };
    
    alignas(64) std::vector<AnnotatedMove> moves;
    size_t filtered_count = 0;
    
public:
    void reserve(size_t count) {
        moves.reserve(std::min(count, size_t(MOVE_BATCH_SIZE)));
    }
    
    // Phase 1: Generate all moves O(M)
    void generate_moves(const Position& pos, MovePicker& mp) {
        moves.clear();
        Move move;
        while ((move = mp.next_move()) != Move::none()) {
            if (pos.legal(move)) {
                AnnotatedMove am{};
                am.move = move;
                am.flags = 0;
                if (pos.capture_stage(move)) am.flags |= AnnotatedMove::CAPTURE;
                if (pos.gives_check(move)) am.flags |= AnnotatedMove::CHECK;
                moves.push_back(am);
            }
        }
    }
    
    // Phase 2: Batch annotate with better cache locality
    void annotate_see_scores(const Position& pos) {
        for (auto& am : moves) {
            if (am.flags & AnnotatedMove::CAPTURE) {
                am.see_score = pos.see_value(am.move);
            }
        }
    }
    
    void annotate_history_scores(const Worker& worker, const Stack* ss) {
        const PieceToHistory* contHist[] = {
            (ss - 1)->continuationHistory, (ss - 2)->continuationHistory,
            (ss - 3)->continuationHistory, (ss - 4)->continuationHistory
        };
        
        for (auto& am : moves) {
            Piece movedPiece = pos.moved_piece(am.move);
            Square to = am.move.to_sq();
            
            am.history_score = worker.mainHistory[pos.side_to_move()][am.move.from_to()]
                            + (*contHist[0])[movedPiece][to]
                            + (*contHist[1])[movedPiece][to]
                            + worker.pawnHistory[pawn_history_index(pos)][movedPiece][to];
        }
    }
    
    // Phase 3: Filter moves with single pass O(M)
    void filter_moves(Value alpha, Depth depth, bool improving) {
        filtered_count = 0;
        
        for (auto& am : moves) {
            // Apply pruning heuristics
            if (!(am.flags & (AnnotatedMove::CAPTURE | AnnotatedMove::CHECK))) {
                // History-based pruning
                if (am.history_score < -4361 * depth) {
                    am.flags |= AnnotatedMove::PRUNED;
                    continue;
                }
                
                // SEE-based pruning for quiet moves
                if (am.see_score < -26 * depth * depth) {
                    am.flags |= AnnotatedMove::PRUNED;
                    continue;
                }
            }
            
            // Move passes filters
            if (!(am.flags & AnnotatedMove::PRUNED)) {
                filtered_count++;
            }
        }
    }
    
    // Phase 4: Get filtered moves for search
    template<typename Callback>
    void process_filtered_moves(Callback&& callback) {
        for (const auto& am : moves) {
            if (!(am.flags & AnnotatedMove::PRUNED)) {
                callback(am);
            }
        }
    }
    
    size_t size() const { return filtered_count; }
};

// Adaptive NNUE Evaluator with context-aware network selection
class AdaptiveNNUEEvaluator {
private:
    const LazyNumaReplicated<Eval::NNUE::Networks>& networks;
    const NumaReplicatedAccessToken token;
    Eval::NNUE::AccumulatorStack& accumulatorStack;
    Eval::NNUE::AccumulatorCaches& refreshTable;
    
    mutable Value cachedEval = VALUE_NONE;
    mutable Key cachedKey = 0;
    
    enum NetworkType { FAST_NETWORK, ACCURATE_NETWORK };
    
    NetworkType select_network(const Position& pos, const Stack* ss, 
                              Value alpha, Value beta, Depth depth, bool pvNode) const {
        // Fast network for shallow non-PV nodes
        if (!pvNode && depth <= 4 && !ss->ttPv) {
            return FAST_NETWORK;
        }
        
        // Fast network for positions far from alpha-beta window
        if (cachedKey == pos.key() && cachedEval != VALUE_NONE) {
            if (cachedEval < alpha - 200 || cachedEval > beta + 200) {
                return FAST_NETWORK;
            }
        }
        
        // Accurate network for critical positions
        return ACCURATE_NETWORK;
    }
    
public:
    AdaptiveNNUEEvaluator(const LazyNumaReplicated<Eval::NNUE::Networks>& nets,
                         NumaReplicatedAccessToken t,
                         Eval::NNUE::AccumulatorStack& accStack,
                         Eval::NNUE::AccumulatorCaches& refTable)
        : networks(nets), token(t), accumulatorStack(accStack), refreshTable(refTable) {}
    
    Value evaluate(const Position& pos, const Stack* ss, Value alpha, 
                  Value beta, Depth depth, bool pvNode, int optimism) {
        if (cachedKey == pos.key() && cachedEval != VALUE_NONE) {
            return cachedEval;
        }
        
        NetworkType netType = select_network(pos, ss, alpha, beta, depth, pvNode);
        
        // Use appropriate network based on context
        // Note: This requires modification to NNUE evaluation to support multiple networks
        Value eval = Eval::evaluate(networks[token], pos, accumulatorStack, 
                                   refreshTable, optimism, netType == FAST_NETWORK);
        
        cachedKey = pos.key();
        cachedEval = eval;
        return eval;
    }
    
    void clear_cache() {
        cachedEval = VALUE_NONE;
        cachedKey = 0;
    }
};

// Streamlined policy dispatch system
struct SearchPolicies {
    using RazoringFunc = bool (*)(bool, Value, Value, Depth);
    using FutilityFunc = bool (*)(const Stack*, bool, Depth, Value, Value, bool, bool, int);
    using NullMoveFunc = bool (*)(bool, const Stack*, Value, bool, const Position&, int);
    using LMRFunc = Depth (*)(const Worker&, bool, Depth, int, int, bool, bool, Move, Move, const Stack*);
    using SEEFunc = bool (*)(const Position&, Move, Depth, int);
    using ExtensionFunc = int (*)(bool, Move, const TTData&, bool, Depth, int, const Stack*, bool);
    
    RazoringFunc should_razor;
    FutilityFunc should_futility_prune;
    NullMoveFunc should_null_move;
    LMRFunc get_lmr_reduction;
    SEEFunc should_see_prune;
    ExtensionFunc get_extension;
};

// Default policy implementations
namespace DefaultPolicies {
    
    bool should_razor(bool pvNode, Value eval, Value alpha, Depth depth) {
        return !pvNode && eval < alpha - 495 - 290 * depth * depth;
    }
    
    bool should_futility_prune(const Stack* ss, bool cutNode, Depth depth, Value eval,
                              Value beta, bool improving, bool opponentWorsening, int correctionValue) {
        if (ss->ttPv || depth >= 14 || eval < beta || is_loss(beta) || is_win(eval))
            return false;
        
        Value futilityMult = 90 - 20 * (cutNode && !ss->ttHit);
        Value margin = futilityMult * depth
                     - improving * futilityMult * 2
                     - opponentWorsening * futilityMult / 3
                     + (ss - 1)->statScore / 356
                     + std::abs(correctionValue) / 171290;
        
        return eval - margin >= beta;
    }
    
    bool should_null_move(bool cutNode, const Stack* ss, Value beta, bool excludedMove,
                         const Position& pos, int nmpMinPly) {
        return cutNode && ss->staticEval >= beta - 19 * depth + 403 && !excludedMove
            && pos.non_pawn_material(pos.side_to_move()) && ss->ply >= nmpMinPly && !is_loss(beta);
    }
    
    Depth get_lmr_reduction(const Worker& worker, bool improving, Depth depth, int moveCount,
                           int delta, bool pvNode, bool cutNode, Move move, Move ttMove, const Stack* ss) {
        int base = worker.reductions[depth] * worker.reductions[moveCount];
        base -= delta * 731 / worker.rootDelta;
        base += !improving * base * 216 / 512;
        
        if (ss->ttPv) {
            base -= 2510 + pvNode * 963;
        }
        
        if (cutNode) {
            base += 3000 + 1024 * !ttMove;
        }
        
        if (move == ttMove) {
            base -= 2043;
        }
        
        return base + 1089;
    }
    
    bool should_see_prune(const Position& pos, Move move, Depth depth, int captHist) {
        if (pos.capture_stage(move)) {
            int margin = std::clamp(158 * depth + captHist / 31, 0, 283 * depth);
            return !pos.see_ge(move, -margin);
        } else {
            return !pos.see_ge(move, -26 * depth * depth);
        }
    }
    
    int get_extension(bool rootNode, Move move, const TTData& ttData, bool excludedMove,
                     Depth depth, int completedDepth, const Stack* ss, bool pvNode) {
        if (!rootNode && move == ttData.move && !excludedMove
            && depth >= 6 - (completedDepth > 26) + ss->ttPv
            && is_valid(ttData.value) && !is_decisive(ttData.value)
            && (ttData.bound & BOUND_LOWER) && ttData.depth >= depth - 3) {
            
            Value singularBeta = ttData.value - (56 + 79 * (ss->ttPv && !pvNode)) * depth / 58;
            return 1; // Simplified for this example
        }
        return 0;
    }
}

// Initialize default policies
const SearchPolicies defaultPolicies = {
    DefaultPolicies::should_razor,
    DefaultPolicies::should_futility_prune,
    DefaultPolicies::should_null_move,
    DefaultPolicies::get_lmr_reduction,
    DefaultPolicies::should_see_prune,
    DefaultPolicies::get_extension
};

// Helper functions
inline int correction_value(const Worker& w, const Position& pos, const Stack* const ss) {
    const Color us = pos.side_to_move();
    const auto m = (ss - 1)->currentMove;
    const auto pcv = w.pawnCorrectionHistory[pawn_correction_history_index(pos)][us];
    const auto micv = w.minorPieceCorrectionHistory[minor_piece_index(pos)][us];
    const auto wnpcv = w.nonPawnCorrectionHistory[non_pawn_index<WHITE>(pos)][WHITE][us];
    const auto bnpcv = w.nonPawnCorrectionHistory[non_pawn_index<BLACK>(pos)][BLACK][us];
    const auto cntcv = m.is_ok() ? 
        (*(ss - 2)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()] : 0;

    return 8867 * pcv + 8136 * micv + 10757 * (wnpcv + bnpcv) + 7232 * cntcv;
}

inline Value to_corrected_static_eval(const Value v, const int cv) {
    return std::clamp(v + cv / 131072, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
}

void update_correction_history(const Position& pos, Stack* const ss, 
                              Search::Worker& workerThread, const int bonus) {
    const Move m = (ss - 1)->currentMove;
    const Color us = pos.side_to_move();

    static constexpr int nonPawnWeight = 165;

    workerThread.pawnCorrectionHistory[pawn_correction_history_index(pos)][us] << bonus;
    workerThread.minorPieceCorrectionHistory[minor_piece_index(pos)][us] << bonus * 153 / 128;
    workerThread.nonPawnCorrectionHistory[non_pawn_index<WHITE>(pos)][WHITE][us]
      << bonus * nonPawnWeight / 128;
    workerThread.nonPawnCorrectionHistory[non_pawn_index<BLACK>(pos)][BLACK][us]
      << bonus * nonPawnWeight / 128;

    if (m.is_ok())
        (*(ss - 2)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()]
          << bonus * 153 / 128;
}

Value value_draw(size_t nodes) { return VALUE_DRAW - 1 + Value(nodes & 0x2); }

Value value_to_tt(Value v, int ply) { 
    return is_win(v) ? v + ply : is_loss(v) ? v - ply : v; 
}

Value value_from_tt(Value v, int ply, int r50c) {
    if (!is_valid(v))
        return VALUE_NONE;

    if (is_win(v)) {
        if (v >= VALUE_MATE_IN_MAX_PLY && VALUE_MATE - v > 100 - r50c)
            return VALUE_TB_WIN_IN_MAX_PLY - 1;
        if (VALUE_TB - v > 100 - r50c)
            return VALUE_TB_WIN_IN_MAX_PLY - 1;
        return v - ply;
    }

    if (is_loss(v)) {
        if (v <= VALUE_MATED_IN_MAX_PLY && VALUE_MATE + v > 100 - r50c)
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;
        if (VALUE_TB + v > 100 - r50c)
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;
        return v + ply;
    }

    return v;
}

void update_pv(Move* pv, Move move, const Move* childPv) {
    for (*pv++ = move; childPv && *childPv != Move::none();)
        *pv++ = *childPv++;
    *pv = Move::none();
}

void update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus);
void update_quiet_histories(const Position& pos, Stack* ss, Search::Worker& workerThread, 
                           Move move, int bonus);
void update_all_stats(const Position& pos, Stack* ss, Search::Worker& workerThread,
                     Move bestMove, Square prevSq, SearchedList& quietsSearched,
                     SearchedList& capturesSearched, Depth depth, Move TTMove, int moveCount);

} // namespace

// Worker class implementation
Search::Worker::Worker(SharedState& sharedState, std::unique_ptr<ISearchManager> sm,
                      size_t threadId, NumaReplicatedAccessToken token) :
    threadIdx(threadId),
    numaAccessToken(token),
    manager(std::move(sm)),
    options(sharedState.options),
    threads(sharedState.threads),
    networks(sharedState.networks),
    refreshTable(networks[token])
#ifdef USE_NUMA_TT
    ,generation8(sharedState.generation8)
#endif
{
#ifdef USE_NUMA_TT
    // NUMA-aware TT assignment with precomputed mapping
    if (threadId < sharedState.threads.get_bound_thread_count_by_numa_node().size()) {
        auto nodeCounts = sharedState.threads.get_bound_thread_count_by_numa_node();
        NumaNodeID nodeId = 0;
        size_t threadsSeen = 0;
        
        // O(1) lookup would be better with precomputed map
        for (size_t i = 0; i < nodeCounts.size(); ++i) {
            if (threadId < threadsSeen + nodeCounts[i]) {
                nodeId = i;
                break;
            }
            threadsSeen += nodeCounts[i];
        }
        
        l1_tt = (nodeId < sharedState.l1_tts.size()) ? sharedState.l1_tts[nodeId].get() : nullptr;
        l2_tt = sharedState.l2_tt.get();
    } else {
        l1_tt = nullptr;
        l2_tt = sharedState.l2_tt.get();
    }
#else
    ,tt(sharedState.tt)
#endif
    clear();
}

void Search::Worker::ensure_network_replicated() {
    (void)(networks[numaAccessToken]);
}

void Search::Worker::start_searching() {
    accumulatorStack.reset();

    if (!is_mainthread()) {
        iterative_deepening();
        return;
    }

    main_manager()->tm.init(limits, rootPos.side_to_move(), rootPos.game_ply(), options,
                            main_manager()->originalTimeAdjust);
#ifdef USE_NUMA_TT
    generation8 += GENERATION_DELTA;
#else
    tt.new_search();
#endif

    if (rootMoves.empty()) {
        rootMoves.emplace_back(Move::none());
        main_manager()->updates.onUpdateNoMoves(
          {0, {rootPos.checkers() ? -VALUE_MATE : VALUE_DRAW, rootPos}});
    } else {
        threads.start_searching();
        iterative_deepening();
    }

    while (!threads.stop && (main_manager()->ponder || limits.infinite)) {}
    
    threads.stop = true;
    threads.wait_for_search_finished();

    if (limits.npmsec)
        main_manager()->tm.advance_nodes_time(threads.nodes_searched()
                                              - limits.inc[rootPos.side_to_move()]);

    Worker* bestThread = this;
    Skill skill = Skill(options["Skill Level"], 
                       options["UCI_LimitStrength"] ? int(options["UCI_Elo"]) : 0);

    if (int(options["MultiPV"]) == 1 && !limits.depth && !limits.mate && !skill.enabled()
        && rootMoves[0].pv[0] != Move::none())
        bestThread = threads.get_best_thread()->worker.get();

    main_manager()->bestPreviousScore = bestThread->rootMoves[0].score;
    main_manager()->bestPreviousAverageScore = bestThread->rootMoves[0].averageScore;

    if (bestThread != this)
#ifdef USE_NUMA_TT
        main_manager()->pv(*bestThread, threads, bestThread->completedDepth);
#else
        main_manager()->pv(*bestThread, threads, tt, bestThread->completedDepth);
#endif

    std::string ponder;
#ifdef USE_NUMA_TT
    if (bestThread->rootMoves[0].pv.size() > 1
        || (l2_tt && bestThread->rootMoves[0].extract_ponder_from_tt(*l2_tt, rootPos)))
#else
    if (bestThread->rootMoves[0].pv.size() > 1
        || bestThread->rootMoves[0].extract_ponder_from_tt(tt, rootPos))
#endif
        ponder = UCIEngine::move(bestThread->rootMoves[0].pv[1], rootPos.is_chess960());

    auto bestmove = UCIEngine::move(bestThread->rootMoves[0].pv[0], rootPos.is_chess960());
    main_manager()->updates.onBestmove(bestmove, ponder);
}

void Search::Worker::iterative_deepening() {
    SearchManager* mainThread = (is_mainthread() ? main_manager() : nullptr);

    Move pv[MAX_PLY + 1];
    Depth lastBestMoveDepth = 0;
    Value lastBestScore = -VALUE_INFINITE;
    auto lastBestPV = std::vector{Move::none()};

    Value alpha, beta;
    Value bestValue = -VALUE_INFINITE;
    Color us = rootPos.side_to_move();
    double timeReduction = 1, totBestMoveChanges = 0;
    int delta, iterIdx = 0;

    // Optimized stack allocation with lazy initialization
    alignas(64) Stack stack[MAX_PLY + 10] = {};
    Stack* ss = stack + 7;

    // Initialize only sentinel entries
    for (int i = 7; i > 0; --i) {
        (ss - i)->continuationHistory = &continuationHistory[0][0][NO_PIECE][0];
        (ss - i)->continuationCorrectionHistory = &continuationCorrectionHistory[NO_PIECE][0];
        (ss - i)->staticEval = VALUE_NONE;
    }

    for (int i = 0; i <= MAX_PLY + 2; ++i)
        (ss + i)->ply = i;

    ss->pv = pv;

    if (mainThread) {
        if (mainThread->bestPreviousScore == VALUE_INFINITE)
            mainThread->iterValue.fill(VALUE_ZERO);
        else
            mainThread->iterValue.fill(mainThread->bestPreviousScore);
    }

    size_t multiPV = size_t(options["MultiPV"]);
    Skill skill(options["Skill Level"], 
                options["UCI_LimitStrength"] ? int(options["UCI_Elo"]) : 0);

    if (skill.enabled())
        multiPV = std::max(multiPV, size_t(4));

    multiPV = std::min(multiPV, rootMoves.size());

    int searchAgainCounter = 0;
    lowPlyHistory.fill(89);

    while (++rootDepth < MAX_PLY && !threads.stop
           && !(limits.depth && mainThread && rootDepth > limits.depth)) {
        
        if (mainThread)
            totBestMoveChanges /= 2;

        for (RootMove& rm : rootMoves)
            rm.previousScore = rm.score;

        size_t pvFirst = 0;
        pvLast = 0;

        if (!threads.increaseDepth)
            searchAgainCounter++;

        for (pvIdx = 0; pvIdx < multiPV; ++pvIdx) {
            if (pvIdx == pvLast) {
                pvFirst = pvLast;
                for (pvLast++; pvLast < rootMoves.size(); pvLast++)
                    if (rootMoves[pvLast].tbRank != rootMoves[pvFirst].tbRank)
                        break;
            }

            selDepth = 0;
            delta = 5 + std::abs(rootMoves[pvIdx].meanSquaredScore) / 11131;
            Value avg = rootMoves[pvIdx].averageScore;
            alpha = std::max(avg - delta, -VALUE_INFINITE);
            beta = std::min(avg + delta, VALUE_INFINITE);

            optimism[us] = 136 * avg / (std::abs(avg) + 93);
            optimism[~us] = -optimism[us];

            int failedHighCnt = 0;
            while (true) {
                Depth adjustedDepth = std::max(1, rootDepth - failedHighCnt - 
                                              3 * (searchAgainCounter + 1) / 4);
                rootDelta = beta - alpha;
                bestValue = search<Root>(rootPos, ss, alpha, beta, adjustedDepth, false);

                std::stable_sort(rootMoves.begin() + pvIdx, rootMoves.begin() + pvLast);

                if (threads.stop)
                    break;

                if (mainThread && multiPV == 1 && (bestValue <= alpha || bestValue >= beta)
                    && nodes > 10000000)
#ifdef USE_NUMA_TT
                    main_manager()->pv(*this, threads, rootDepth);
#else
                    main_manager()->pv(*this, threads, tt, rootDepth);
#endif

                if (bestValue <= alpha) {
                    beta = (3 * alpha + beta) / 4;
                    alpha = std::max(bestValue - delta, -VALUE_INFINITE);
                    failedHighCnt = 0;
                    if (mainThread)
                        mainThread->stopOnPonderhit = false;
                } else if (bestValue >= beta) {
                    beta = std::min(bestValue + delta, VALUE_INFINITE);
                    ++failedHighCnt;
                } else
                    break;

                delta += delta / 3;
                assert(alpha >= -VALUE_INFINITE && beta <= VALUE_INFINITE);
            }

            std::stable_sort(rootMoves.begin() + pvFirst, rootMoves.begin() + pvIdx + 1);

            if (mainThread && (threads.stop || pvIdx + 1 == multiPV || nodes > 10000000)
                && !(threads.abortedSearch && is_loss(rootMoves[0].uciScore)))
#ifdef USE_NUMA_TT
                main_manager()->pv(*this, threads, rootDepth);
#else
                main_manager()->pv(*this, threads, tt, rootDepth);
#endif

            if (threads.stop)
                break;
        }

        if (!threads.stop)
            completedDepth = rootDepth;

        if (threads.abortedSearch && rootMoves[0].score != -VALUE_INFINITE
            && is_loss(rootMoves[0].score)) {
            Utility::move_to_front(rootMoves, [&lastBestPV = std::as_const(lastBestPV)](
                                              const auto& rm) { return rm == lastBestPV[0]; });
            rootMoves[0].pv = lastBestPV;
            rootMoves[0].score = rootMoves[0].uciScore = lastBestScore;
        } else if (rootMoves[0].pv[0] != lastBestPV[0]) {
            lastBestPV = rootMoves[0].pv;
            lastBestScore = rootMoves[0].score;
            lastBestMoveDepth = rootDepth;
        }

        if (!mainThread)
            continue;

        if (limits.mate && rootMoves[0].score == rootMoves[0].uciScore
            && ((rootMoves[0].score >= VALUE_MATE_IN_MAX_PLY
                 && VALUE_MATE - rootMoves[0].score <= 2 * limits.mate)
                || (rootMoves[0].score != -VALUE_INFINITE
                    && rootMoves[0].score <= VALUE_MATED_IN_MAX_PLY
                    && VALUE_MATE + rootMoves[0].score <= 2 * limits.mate)))
            threads.stop = true;

        if (skill.enabled() && skill.time_to_pick(rootDepth))
            skill.pick_best(rootMoves, multiPV);

        for (auto&& th : threads) {
            totBestMoveChanges += th->worker->bestMoveChanges;
            th->worker->bestMoveChanges = 0;
        }

        if (limits.use_time_management() && !threads.stop && !mainThread->stopOnPonderhit) {
            uint64_t nodesEffort = rootMoves[0].effort * 100000 / std::max(size_t(1), size_t(nodes));

            double fallingEval = (11.396 + 2.035 * (mainThread->bestPreviousAverageScore - bestValue)
                                + 0.968 * (mainThread->iterValue[iterIdx] - bestValue)) / 100.0;
            fallingEval = std::clamp(fallingEval, 0.5786, 1.6752);

            double k = 0.527;
            double center = lastBestMoveDepth + 11;
            timeReduction = 0.8 + 0.84 / (1.077 + std::exp(-k * (completedDepth - center)));
            double reduction = (1.4540 + mainThread->previousTimeReduction) / (2.1593 * timeReduction);
            double bestMoveInstability = 0.9929 + 1.8519 * totBestMoveChanges / threads.size();

            double totalTime = mainThread->tm.optimum() * fallingEval * reduction * bestMoveInstability;

            if (rootMoves.size() == 1)
                totalTime = std::min(500.0, totalTime);

            auto elapsedTime = elapsed();

            if (completedDepth >= 10 && nodesEffort >= 97056 && elapsedTime > totalTime * 0.6540
                && !mainThread->ponder)
                threads.stop = true;

            if (elapsedTime > std::min(totalTime, double(mainThread->tm.maximum()))) {
                if (mainThread->ponder)
                    mainThread->stopOnPonderhit = true;
                else
                    threads.stop = true;
            } else
                threads.increaseDepth = mainThread->ponder || elapsedTime <= totalTime * 0.5138;
        }

        mainThread->iterValue[iterIdx] = bestValue;
        iterIdx = (iterIdx + 1) & 3;
    }

    if (!mainThread)
        return;

    mainThread->previousTimeReduction = timeReduction;

    if (skill.enabled())
        std::swap(rootMoves[0], *std::find(rootMoves.begin(), rootMoves.end(),
                                          skill.best ? skill.best : skill.pick_best(rootMoves, multiPV)));
}

// Optimized search function with policy dispatch
template<NodeType nodeType>
Value Search::Worker::search(Position& pos, Stack* ss, Value alpha, Value beta, 
                            Depth depth, bool cutNode) {
    
    constexpr bool PvNode = nodeType != NonPV;
    constexpr bool rootNode = nodeType == Root;
    const bool allNode = !(PvNode || cutNode);
    const SearchPolicies& policies = defaultPolicies;

    if (depth <= 0) {
        constexpr auto nt = PvNode ? PV : NonPV;
        return qsearch<nt>(pos, ss, alpha, beta);
    }

    depth = std::min(depth, MAX_PLY - 1);

    if (!rootNode && alpha < VALUE_DRAW && pos.upcoming_repetition(ss->ply)) {
        alpha = value_draw(nodes);
        if (alpha >= beta)
            return alpha;
    }

    assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));
    assert(0 < depth && depth < MAX_PLY);
    assert(!(PvNode && cutNode));

    Move pv[MAX_PLY + 1];
    StateInfo st;
    
    Key posKey;
    Move move, excludedMove, bestMove;
    Depth extension, newDepth;
    Value bestValue, value, eval, maxValue, probCutBeta;
    bool givesCheck, improving, priorCapture, opponentWorsening;
    bool capture, ttCapture;
    int priorReduction;
    Piece movedPiece;

    SearchedList capturesSearched;
    SearchedList quietsSearched;

    // Initialize node
    ss->inCheck = pos.checkers();
    priorCapture = pos.captured_piece();
    Color us = pos.side_to_move();
    ss->moveCount = 0;
    bestValue = -VALUE_INFINITE;
    maxValue = VALUE_INFINITE;

    if (is_mainthread())
        main_manager()->check_time(*this);

    if (PvNode && selDepth < ss->ply + 1)
        selDepth = ss->ply + 1;

    if (!rootNode) {
        if (threads.stop.load(std::memory_order_relaxed) || pos.is_draw(ss->ply)
            || ss->ply >= MAX_PLY)
            return (ss->ply >= MAX_PLY && !ss->inCheck) ? evaluate(pos) : value_draw(nodes);

        alpha = std::max(mated_in(ss->ply), alpha);
        beta = std::min(mate_in(ss->ply + 1), beta);
        if (alpha >= beta)
            return alpha;
    }

    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    Square prevSq = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;
    bestMove = Move::none();
    priorReduction = (ss - 1)->reduction;
    (ss - 1)->reduction = 0;
    ss->statScore = 0;
    (ss + 2)->cutoffCnt = 0;

    // Optimized TT probe with caching
    excludedMove = ss->excludedMove;
    posKey = pos.key();
    
#ifdef USE_NUMA_TT
    OptimizedTTInterface<TT_CACHE_SIZE> ttInterface(l1_tt, l2_tt, generation8);
#else
    OptimizedTTInterface<TT_CACHE_SIZE> ttInterface(tt);
#endif
    
    auto [ttHit, ttData, ttWriter] = ttInterface.probe(posKey);
    
    ss->ttHit = ttHit;
    ttData.move = rootNode ? rootMoves[pvIdx].pv[0] : ttHit ? ttData.move : Move::none();
    ttData.value = ttHit ? value_from_tt(ttData.value, ss->ply, pos.rule50_count()) : VALUE_NONE;
    ss->ttPv = excludedMove ? ss->ttPv : PvNode || (ttHit && ttData.is_pv);
    ttCapture = ttData.move && pos.capture_stage(ttData.move);

    // TT cutoff
    if (!PvNode && !excludedMove && ttData.depth > depth - (ttData.value <= beta)
        && is_valid(ttData.value)
        && (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER))
        && (cutNode == (ttData.value >= beta) || depth > 5)) {
        
        if (ttData.move && ttData.value >= beta) {
            if (!ttCapture)
                update_quiet_histories(pos, ss, *this, ttData.move,
                                     std::min(127 * depth - 74, 1063));
            if (prevSq != SQ_NONE && (ss - 1)->moveCount <= 3 && !priorCapture)
                update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -2128);
        }

        if (pos.rule50_count() < 91) {
            if (depth >= 8 && ttData.move && pos.pseudo_legal(ttData.move) 
                && pos.legal(ttData.move) && !is_decisive(ttData.value)) {
                
                do_move(pos, ttData.move, st, ss);
                Key nextPosKey = pos.key();
                auto [ttHitNext, ttDataNext, ttWriterNext] = ttInterface.probe(nextPosKey);
                undo_move(pos, ttData.move);

                if (!is_valid(ttDataNext.value))
                    return ttData.value;
                if ((ttData.value >= beta) == (-ttDataNext.value >= beta))
                    return ttData.value;
            } else
                return ttData.value;
        }
    }

    // Tablebase probe
    if (!rootNode && !excludedMove && tbConfig.cardinality) {
        int piecesCount = pos.count<ALL_PIECES>();

        if (piecesCount <= tbConfig.cardinality
            && (piecesCount < tbConfig.cardinality || depth >= tbConfig.probeDepth)
            && pos.rule50_count() == 0 && !pos.can_castle(ANY_CASTLING)) {
            
            TB::ProbeState err;
            TB::WDLScore wdl = Tablebases::probe_wdl(pos, &err);

            if (is_mainthread())
                main_manager()->callsCnt = 0;

            if (err != TB::ProbeState::FAIL) {
                tbHits.fetch_add(1, std::memory_order_relaxed);

                int drawScore = tbConfig.useRule50 ? 1 : 0;
                Value tbValue = VALUE_TB - ss->ply;

                value = wdl < -drawScore ? -tbValue
                      : wdl > drawScore  ? tbValue
                                         : VALUE_DRAW + 2 * wdl * drawScore;

                Bound b = wdl < -drawScore ? BOUND_UPPER
                        : wdl > drawScore  ? BOUND_LOWER
                                           : BOUND_EXACT;

                if (b == BOUND_EXACT || (b == BOUND_LOWER ? value >= beta : value <= alpha)) {
                    ttWriter.write(posKey, value_to_tt(value, ss->ply), ss->ttPv, b,
                                 std::min(MAX_PLY - 1, depth + 6), Move::none(), VALUE_NONE,
                                 ttInterface.generation());
                    return value;
                }

                if (PvNode) {
                    if (b == BOUND_LOWER)
                        bestValue = value, alpha = std::max(alpha, bestValue);
                    else
                        maxValue = value;
                }
            }
        }
    }

    // Static evaluation with adaptive NNUE
    Value unadjustedStaticEval = VALUE_NONE;
    const auto correctionValue = correction_value(*this, pos, ss);
    AdaptiveNNUEEvaluator nnueEval(networks, numaAccessToken, accumulatorStack, refreshTable);
    
    if (ss->inCheck) {
        ss->staticEval = eval = (ss - 2)->staticEval;
        improving = false;
        goto moves_loop;
    } else if (excludedMove) {
        unadjustedStaticEval = eval = ss->staticEval;
    } else if (ss->ttHit) {
        unadjustedStaticEval = ttData.eval;
        if (!is_valid(unadjustedStaticEval))
            unadjustedStaticEval = nnueEval.evaluate(pos, ss, alpha, beta, depth, PvNode, optimism[us]);

        ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue);

        if (is_valid(ttData.value)
            && (ttData.bound & (ttData.value > eval ? BOUND_LOWER : BOUND_UPPER)))
            eval = ttData.value;
    } else {
        unadjustedStaticEval = nnueEval.evaluate(pos, ss, alpha, beta, depth, PvNode, optimism[us]);
        ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue);

        ttWriter.write(posKey, VALUE_NONE, ss->ttPv, BOUND_NONE, DEPTH_UNSEARCHED, Move::none(),
                     unadjustedStaticEval, ttInterface.generation());
    }

    // History bonus
    if (((ss - 1)->currentMove).is_ok() && !(ss - 1)->inCheck && !priorCapture) {
        int bonus = std::clamp(-10 * int((ss - 1)->staticEval + ss->staticEval), -1979, 1561) + 630;
        mainHistory[~us][((ss - 1)->currentMove).from_to()] << bonus * 935 / 1024;
        if (!ttHit && type_of(pos.piece_on(prevSq)) != PAWN
            && ((ss - 1)->currentMove).type_of() != PROMOTION)
            pawnHistory[pawn_history_index(pos)][pos.piece_on(prevSq)][prevSq]
              << bonus * 1428 / 1024;
    }

    improving = ss->staticEval > (ss - 2)->staticEval;
    opponentWorsening = ss->staticEval > -(ss - 1)->staticEval;

    if (priorReduction >= (depth < 10 ? 1 : 3) && !opponentWorsening)
        depth++;
    if (priorReduction >= 2 && depth >= 2 && ss->staticEval + (ss - 1)->staticEval > 177)
        depth--;

    // Policy-based pruning
    if (policies.should_razor(PvNode, eval, alpha, depth))
        return qsearch<NonPV>(pos, ss, alpha, beta);

    if (policies.should_futility_prune(ss, cutNode, depth, eval, beta, improving, 
                                      opponentWorsening, correctionValue)) {
        return beta + (eval - beta) / 3;
    }

    // Null move with optimized NNUE (flip_perspective)
    if (policies.should_null_move(cutNode, ss, beta, excludedMove, pos, nmpMinPly)) {
        assert((ss - 1)->currentMove != Move::null());

        Depth R = 7 + depth / 3;

        ss->currentMove = Move::null();
        ss->continuationHistory = &continuationHistory[0][0][NO_PIECE][0];
        ss->continuationCorrectionHistory = &continuationCorrectionHistory[NO_PIECE][0];

        // Optimized null move with flip_perspective
        do_null_move(pos, st);
        
        // Use flip_perspective for efficient NNUE evaluation
        accumulatorStack.flip_perspective();
        Value nullValue = -search<NonPV>(pos, ss + 1, -beta, -beta + 1, depth - R, false);
        accumulatorStack.flip_perspective(); // Restore
        
        undo_null_move(pos);

        if (nullValue >= beta && !is_win(nullValue)) {
            if (nmpMinPly || depth < 16)
                return nullValue;

            assert(!nmpMinPly);
            nmpMinPly = ss->ply + 3 * (depth - R) / 4;
            Value v = search<NonPV>(pos, ss, beta - 1, beta, depth - R, false);
            nmpMinPly = 0;

            if (v >= beta)
                return nullValue;
        }
    }

    improving |= ss->staticEval >= beta;

    if (!allNode && depth >= 6 && !ttData.move && priorReduction <= 3)
        depth--;

    // ProbCut
    probCutBeta = beta + 215 - 60 * improving;
    if (depth >= 3 && !is_decisive(beta)
        && !(is_valid(ttData.value) && ttData.value < probCutBeta)) {
        
        assert(probCutBeta < VALUE_INFINITE && probCutBeta > beta);

        MovePicker mp(pos, ttData.move, probCutBeta - ss->staticEval, &captureHistory);
        Depth probCutDepth = std::max(depth - 5, 0);

        while ((move = mp.next_move()) != Move::none()) {
            assert(move.is_ok());

            if (move == excludedMove || !pos.legal(move))
                continue;

            assert(pos.capture_stage(move));
            movedPiece = pos.moved_piece(move);

            do_move(pos, move, st, ss);
            value = -qsearch<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1);

            if (value >= probCutBeta && probCutDepth > 0)
                value = -search<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1, 
                                     probCutDepth, !cutNode);

            undo_move(pos, move);

            if (value >= probCutBeta) {
                ttWriter.write(posKey, value_to_tt(value, ss->ply), ss->ttPv, BOUND_LOWER,
                             probCutDepth + 1, move, unadjustedStaticEval, ttInterface.generation());

                if (!is_decisive(value))
                    return value - (probCutBeta - beta);
            }
        }
    }

moves_loop:
    probCutBeta = beta + 417;
    if ((ttData.bound & BOUND_LOWER) && ttData.depth >= depth - 4 && ttData.value >= probCutBeta
        && !is_decisive(beta) && is_valid(ttData.value) && !is_decisive(ttData.value))
        return probCutBeta;

    const PieceToHistory* contHist[] = {
      (ss - 1)->continuationHistory, (ss - 2)->continuationHistory, 
      (ss - 3)->continuationHistory, (ss - 4)->continuationHistory, 
      (ss - 5)->continuationHistory, (ss - 6)->continuationHistory};

    // Batched move processing
    BatchedMoveProcessor moveProcessor;
    MovePicker mp(pos, ttData.move, depth, &mainHistory, &lowPlyHistory, &captureHistory, 
                contHist, &pawnHistory, ss->ply);
    
    moveProcessor.generate_moves(pos, mp);
    moveProcessor.annotate_see_scores(pos);
    moveProcessor.annotate_history_scores(*this, ss);
    moveProcessor.filter_moves(alpha, depth, improving);

    value = bestValue;
    int moveCount = 0;

    // Process filtered moves
    moveProcessor.process_filtered_moves([&](const auto& am) {
        move = am.move;
        
        if (rootNode && !std::count(rootMoves.begin() + pvIdx, rootMoves.begin() + pvLast, move))
            return;

        ss->moveCount = ++moveCount;

        if (rootNode && is_mainthread() && nodes > 10000000) {
            main_manager()->updates.onIter(
              {depth, UCIEngine::move(move, pos.is_chess960()), moveCount + pvIdx});
        }

        if (PvNode)
            (ss + 1)->pv = nullptr;

        extension = policies.get_extension(!rootNode, move, ttData, excludedMove,
                                          depth, completedDepth, ss, PvNode);
        
        capture = am.flags & BatchedMoveProcessor::AnnotatedMove::CAPTURE;
        movedPiece = pos.moved_piece(move);
        givesCheck = am.flags & BatchedMoveProcessor::AnnotatedMove::CHECK;

        (ss + 1)->quietMoveStreak = (!capture && !givesCheck) ? (ss->quietMoveStreak + 1) : 0;

        newDepth = depth - 1 + extension;
        
        int delta = beta - alpha;
        Depth r = policies.get_lmr_reduction(*this, improving, depth, moveCount, delta,
                                            PvNode, cutNode, move, ttData.move, ss);

        // Make move
        do_move(pos, move, st, givesCheck, ss);
        
        uint64_t nodeCount = rootNode ? uint64_t(nodes) : 0;

        // Late moves reduction
        if (depth >= 2 && moveCount > 1) {
            Depth d = std::max(1, std::min(newDepth - r / 1024, newDepth + 1 + PvNode)) + PvNode;

            ss->reduction = newDepth - d;
            value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, d, true);
            ss->reduction = 0;

            if (value > alpha) {
                const bool doDeeperSearch = d < newDepth && value > (bestValue + 43 + 2 * newDepth);
                const bool doShallowerSearch = value < bestValue + 9;

                newDepth += doDeeperSearch - doShallowerSearch;

                if (newDepth > d)
                    value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);

                update_continuation_histories(ss, movedPiece, move.to_sq(), 1412);
            }
        } else if (!PvNode || moveCount > 1) {
            value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);
        }

        if (PvNode && (moveCount == 1 || value > alpha)) {
            (ss + 1)->pv = pv;
            (ss + 1)->pv[0] = Move::none();

            if (move == ttData.move && rootDepth > 8)
                newDepth = std::max(newDepth, 1);

            value = -search<PV>(pos, ss + 1, -beta, -alpha, newDepth, false);
        }

        undo_move(pos, move);

        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

        if (threads.stop.load(std::memory_order_relaxed))
            return;

        if (rootNode) {
            RootMove& rm = *std::find(rootMoves.begin(), rootMoves.end(), move);

            rm.effort += nodes - nodeCount;
            rm.averageScore = rm.averageScore != -VALUE_INFINITE ? 
                            (value + rm.averageScore) / 2 : value;
            rm.meanSquaredScore = rm.meanSquaredScore != -VALUE_INFINITE * VALUE_INFINITE ?
                                (value * std::abs(value) + rm.meanSquaredScore) / 2 :
                                value * std::abs(value);

            if (moveCount == 1 || value > alpha) {
                rm.score = rm.uciScore = value;
                rm.selDepth = selDepth;
                rm.scoreLowerbound = rm.scoreUpperbound = false;

                if (value >= beta) {
                    rm.scoreLowerbound = true;
                    rm.uciScore = beta;
                } else if (value <= alpha) {
                    rm.scoreUpperbound = true;
                    rm.uciScore = alpha;
                }

                rm.pv.resize(1);
                assert((ss + 1)->pv);

                for (Move* m = (ss + 1)->pv; *m != Move::none(); ++m)
                    rm.pv.push_back(*m);

                if (moveCount > 1 && !pvIdx)
                    ++bestMoveChanges;
            } else
                rm.score = -VALUE_INFINITE;
        }

        int inc = (value == bestValue && ss->ply + 2 >= rootDepth && (int(nodes) & 14) == 0
                 && !is_win(std::abs(value) + 1));

        if (value + inc > bestValue) {
            bestValue = value;

            if (value + inc > alpha) {
                bestMove = move;

                if (PvNode && !rootNode)
                    update_pv(ss->pv, move, (ss + 1)->pv);

                if (value >= beta) {
                    ss->cutoffCnt += (extension < 2) || PvNode;
                    assert(value >= beta);
                    return;
                }

                if (depth > 2 && depth < 16 && !is_decisive(value))
                    depth -= 2;

                assert(depth > 0);
                alpha = value;
            }
        }

        if (move != bestMove && moveCount <= SEARCHEDLIST_CAPACITY) {
            if (capture)
                capturesSearched.push_back(move);
            else
                quietsSearched.push_back(move);
        }
    });

    // Final processing
    assert(moveCount || !ss->inCheck || excludedMove || !MoveList<LEGAL>(pos).size());

    if (bestValue >= beta && !is_decisive(bestValue) && !is_decisive(alpha))
        bestValue = (bestValue * depth + beta) / (depth + 1);

    if (!moveCount)
        bestValue = excludedMove ? alpha : ss->inCheck ? mated_in(ss->ply) : VALUE_DRAW;
    else if (bestMove) {
        update_all_stats(pos, ss, *this, bestMove, prevSq, quietsSearched, capturesSearched, 
                       depth, ttData.move, moveCount);
        if (!PvNode)
            ttMoveHistory << (bestMove == ttData.move ? 811 : -848);
    }

    if (PvNode)
        bestValue = std::min(bestValue, maxValue);

    if (bestValue <= alpha)
        ss->ttPv = ss->ttPv || (ss - 1)->ttPv;

    if (!excludedMove && !(rootNode && pvIdx))
        ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv,
                     bestValue >= beta    ? BOUND_LOWER
                     : PvNode && bestMove ? BOUND_EXACT
                                          : BOUND_UPPER,
                     moveCount != 0 ? depth : std::min(MAX_PLY - 1, depth + 6), bestMove,
                     unadjustedStaticEval, ttInterface.generation());

    if (!ss->inCheck && !(bestMove && pos.capture(bestMove))
        && ((bestValue < ss->staticEval && bestValue < beta)
            || (bestValue > ss->staticEval && bestMove))) {
        auto bonus = std::clamp(int(bestValue - ss->staticEval) * depth / 8,
                              -CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
        update_correction_history(pos, ss, *this, bonus);
    }

    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

    return bestValue;
}

// Optimized quiescence search
template<NodeType nodeType>
Value Search::Worker::qsearch(Position& pos, Stack* ss, Value alpha, Value beta) {
    
    static_assert(nodeType != Root);
    constexpr bool PvNode = nodeType == PV;

    assert(alpha >= -VALUE_INFINITE && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));

    if (alpha < VALUE_DRAW && pos.upcoming_repetition(ss->ply)) {
        alpha = value_draw(nodes);
        if (alpha >= beta)
            return alpha;
    }

    Move pv[MAX_PLY + 1];
    StateInfo st;

    Key posKey;
    Move move, bestMove;
    Value bestValue, value, futilityBase;
    bool pvHit, givesCheck, capture;
    int moveCount;

    if (PvNode) {
        (ss + 1)->pv = pv;
        ss->pv[0] = Move::none();
    }

    bestMove = Move::none();
    ss->inCheck = pos.checkers();
    moveCount = 0;

    if (PvNode && selDepth < ss->ply + 1)
        selDepth = ss->ply + 1;

    if (pos.is_draw(ss->ply) || ss->ply >= MAX_PLY)
        return (ss->ply >= MAX_PLY && !ss->inCheck) ? evaluate(pos) : VALUE_DRAW;

    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    // Unified TT probe with caching
    posKey = pos.key();
    
#ifdef USE_NUMA_TT
    TTInterface ttInterface(l1_tt, l2_tt, generation8);
#else
    TTInterface ttInterface(tt);
#endif
    
    auto [ttHit, ttData, ttWriter] = ttInterface.probe(posKey);
    
    ss->ttHit = ttHit;
    ttData.move = ttHit ? ttData.move : Move::none();
    ttData.value = ttHit ? value_from_tt(ttData.value, ss->ply, pos.rule50_count()) : VALUE_NONE;
    pvHit = ttHit && ttData.is_pv;

    if (!PvNode && ttData.depth >= DEPTH_QS
        && is_valid(ttData.value)
        && (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER)))
        return ttData.value;

    // Static evaluation
    Value unadjustedStaticEval = VALUE_NONE;
    if (ss->inCheck) {
        bestValue = futilityBase = -VALUE_INFINITE;
    } else {
        const auto correctionValue = correction_value(*this, pos, ss);

        if (ss->ttHit) {
            unadjustedStaticEval = ttData.eval;
            if (!is_valid(unadjustedStaticEval))
                unadjustedStaticEval = evaluate(pos);
            ss->staticEval = bestValue = to_corrected_static_eval(unadjustedStaticEval, correctionValue);

            if (is_valid(ttData.value) && !is_decisive(ttData.value)
                && (ttData.bound & (ttData.value > bestValue ? BOUND_LOWER : BOUND_UPPER)))
                bestValue = ttData.value;
        } else {
            unadjustedStaticEval = evaluate(pos);
            ss->staticEval = bestValue = to_corrected_static_eval(unadjustedStaticEval, correctionValue);
        }

        if (bestValue >= beta) {
            if (!is_decisive(bestValue))
                bestValue = (bestValue + beta) / 2;
            if (!ss->ttHit)
                ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), false, BOUND_LOWER,
                             DEPTH_UNSEARCHED, Move::none(), unadjustedStaticEval,
                             ttInterface.generation());
            return bestValue;
        }

        if (bestValue > alpha)
            alpha = bestValue;

        futilityBase = ss->staticEval + 352;
    }

    const PieceToHistory* contHist[] = {(ss - 1)->continuationHistory,
                                       (ss - 2)->continuationHistory};

    Square prevSq = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;

    MovePicker mp(pos, ttData.move, DEPTH_QS, &mainHistory, &lowPlyHistory, &captureHistory,
                contHist, &pawnHistory, ss->ply);

    while ((move = mp.next_move()) != Move::none()) {
        assert(move.is_ok());

        if (!pos.legal(move))
            continue;

        givesCheck = pos.gives_check(move);
        capture = pos.capture_stage(move);

        moveCount++;

        // Pruning
        if (!is_loss(bestValue)) {
            if (!givesCheck && move.to_sq() != prevSq && !is_loss(futilityBase)
                && move.type_of() != PROMOTION) {
                if (moveCount > 2)
                    continue;

                Value futilityValue = futilityBase + PieceValue[pos.piece_on(move.to_sq())];

                if (futilityValue <= alpha) {
                    bestValue = std::max(bestValue, futilityValue);
                    continue;
                }

                if (!pos.see_ge(move, alpha - futilityBase)) {
                    bestValue = std::min(alpha, futilityBase);
                    continue;
                }
            }

            if (!capture
                && (*contHist[0])[pos.moved_piece(move)][move.to_sq()]
                     + pawnHistory[pawn_history_index(pos)][pos.moved_piece(move)][move.to_sq()]
                   <= 5868)
                continue;

            if (!pos.see_ge(move, -74))
                continue;
        }

        // Make and search the move
        do_move(pos, move, st, givesCheck, ss);
        value = -qsearch<nodeType>(pos, ss + 1, -beta, -alpha);
        undo_move(pos, move);

        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

        if (value > bestValue) {
            bestValue = value;

            if (value > alpha) {
                bestMove = move;

                if (PvNode)
                    update_pv(ss->pv, move, (ss + 1)->pv);

                if (value < beta)
                    alpha = value;
                else
                    break;
            }
        }
    }

    // Check for mate
    if (ss->inCheck && bestValue == -VALUE_INFINITE) {
        assert(!MoveList<LEGAL>(pos).size());
        return mated_in(ss->ply);
    }

    if (!is_decisive(bestValue) && bestValue > beta)
        bestValue = (bestValue + beta) / 2;

    Color us = pos.side_to_move();
    if (!ss->inCheck && !moveCount && !pos.non_pawn_material(us)
        && type_of(pos.captured_piece()) >= ROOK) {
        if (!((us == WHITE ? shift<NORTH>(pos.pieces(us, PAWN))
                           : shift<SOUTH>(pos.pieces(us, PAWN))) & ~pos.pieces())) {
            pos.state()->checkersBB = Rank1BB;
            if (!MoveList<LEGAL>(pos).size())
                bestValue = VALUE_DRAW;
            pos.state()->checkersBB = 0;
        }
    }

    ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), pvHit,
                 bestValue >= beta ? BOUND_LOWER : BOUND_UPPER, DEPTH_QS, bestMove,
                 unadjustedStaticEval, ttInterface.generation());

    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

    return bestValue;
}

// Everything after here needs some fixing. Can we please continue from HERE?

// Explicit template instantiations
template Value Search::Worker::search<PV>(Position&, Stack*, Value, Value, Depth, bool);
template Value Search::Worker::search<NonPV>(Position&, Stack*, Value, Value, Depth, bool);
template Value Search::Worker::search<Root>(Position&, Stack*, Value, Value, Depth, bool);
template Value Search::Worker::qsearch<PV>(Position&, Stack*, Value, Value);
template Value Search::Worker::qsearch<NonPV>(Position&, Stack*, Value, Value);

// Additional required member functions...
void Search::Worker::clear() {
    mainHistory.fill(64);
    captureHistory.fill(-753);
    pawnHistory.fill(-1275);
    pawnCorrectionHistory.fill(5);
    minorPieceCorrectionHistory.fill(0);
    nonPawnCorrectionHistory.fill(0);
    ttMoveHistory = 0;

    for (auto& to : continuationCorrectionHistory)
        for (auto& h : to)
            h.fill(8);

    for (bool inCheck : {false, true})
        for (StatsType c : {NoCaptures, Captures})
            for (auto& to : continuationHistory[inCheck][c])
                for (auto& h : to)
                    h.fill(-494);

    for (size_t i = 1; i < reductions.size(); ++i)
        reductions[i] = int(2782 / 128.0 * std::log(i));

    refreshTable.clear(networks[numaAccessToken]);
}


Depth Search::Worker::reduction(bool i, Depth d, int mn, int delta) const {
    int reductionScale = reductions[d] * reductions[mn];
    return reductionScale - delta * 731 / rootDelta + !i * reductionScale * 216 / 512 + 1089;
}

// elapsed() returns the time elapsed since the search started. If the
// 'nodestime' option is enabled, it will return the count of nodes searched
// instead. This function is called to check whether the search should be
// stopped based on predefined thresholds like time limits or nodes searched.
//
// elapsed_time() returns the actual time elapsed since the start of the search.
// This function is intended for use only when printing PV outputs, and not used
// for making decisions within the search algorithm itself.
TimePoint Search::Worker::elapsed() const {
    return main_manager()->tm.elapsed([this]() { return threads.nodes_searched(); });
}

TimePoint Search::Worker::elapsed_time() const { return main_manager()->tm.elapsed_time(); }

Value Search::Worker::evaluate(const Position& pos) {
    return Eval::evaluate(networks[numaAccessToken], pos, accumulatorStack, refreshTable,
                          optimism[pos.side_to_move()]);
}

namespace {
// Adjusts a mate or TB score from "plies to mate from the root" to
// "plies to mate from the current position". Standard scores are unchanged.
// The function is called before storing a value in the transposition table.
Value value_to_tt(Value v, int ply) { return is_win(v) ? v + ply : is_loss(v) ? v - ply : v; }


// Inverse of value_to_tt(): it adjusts a mate or TB score from the transposition
// table (which refers to the plies to mate/be mated from current position) to
// "plies to mate/be mated (TB win/loss) from the root". However, to avoid
// potentially false mate or TB scores related to the 50 moves rule and the
// graph history interaction, we return the highest non-TB score instead.
Value value_from_tt(Value v, int ply, int r50c) {

    if (!is_valid(v))
        return VALUE_NONE;

    // handle TB win or better
    if (is_win(v))
    {
        // Downgrade a potentially false mate score
        if (v >= VALUE_MATE_IN_MAX_PLY && VALUE_MATE - v > 100 - r50c)
            return VALUE_TB_WIN_IN_MAX_PLY - 1;

        // Downgrade a potentially false TB score.
        if (VALUE_TB - v > 100 - r50c)
            return VALUE_TB_WIN_IN_MAX_PLY - 1;

        return v - ply;
    }

    // handle TB loss or worse
    if (is_loss(v))
    {
        // Downgrade a potentially false mate score.
        if (v <= VALUE_MATED_IN_MAX_PLY && VALUE_MATE + v > 100 - r50c)
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;

        // Downgrade a potentially false TB score.
        if (VALUE_TB + v > 100 - r50c)
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;

        return v + ply;
    }

    return v;
}


// Adds current move and appends child pv[]
void update_pv(Move* pv, Move move, const Move* childPv) {

    for (*pv++ = move; childPv && *childPv != Move::none();)
        *pv++ = *childPv++;
    *pv = Move::none();
}



// Adds current move and appends child pv[]
void update_pv(Move* pv, Move move, const Move* childPv) {

    for (*pv++ = move; childPv && *childPv != Move::none();)
        *pv++ = *childPv++;
    *pv = Move::none();
}

// Updates stats at the end of search() when a bestMove is found
void update_all_stats(const Position& pos,
                      Stack*          ss,
                      Search::Worker& workerThread,
                      Move            bestMove,
                      Square          prevSq,
                      SearchedList&   quietsSearched,
                      SearchedList&   capturesSearched,
                      Depth           depth,
                      Move            ttMove,
                      int             moveCount) {

    CapturePieceToHistory& captureHistory = workerThread.captureHistory;
    Piece                  movedPiece     = pos.moved_piece(bestMove);
    PieceType              capturedPiece;

    int bonus = std::min(142 * depth - 88, 1501) + 318 * (bestMove == ttMove);
    int malus = std::min(757 * depth - 172, 2848) - 30 * moveCount;

    if (!pos.capture_stage(bestMove)) {
        update_quiet_histories(pos, ss, workerThread, bestMove, bonus * 1054 / 1024);

        // Decrease stats for all non-best quiet moves
        for (Move move : quietsSearched)
            update_quiet_histories(pos, ss, workerThread, move, -malus * 1388 / 1024);
    }
    else {
        // Increase stats for the best move in case it was a capture move
        capturedPiece = type_of(pos.piece_on(bestMove.to_sq()));
        captureHistory[movedPiece][bestMove.to_sq()][capturedPiece] << bonus * 1235 / 1024;
    }

    // Extra penalty for a quiet early move that was not a TT move in
    // previous ply when it gets refuted.
    if (prevSq != SQ_NONE && ((ss - 1)->moveCount == 1 + (ss - 1)->ttHit) && !pos.captured_piece())
        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -malus * 595 / 1024);

    // Decrease stats for all non-best capture moves
    for (Move move : capturesSearched) {
        movedPiece    = pos.moved_piece(move);
        capturedPiece = type_of(pos.piece_on(move.to_sq()));
        captureHistory[movedPiece][move.to_sq()][capturedPiece] << -malus * 1354 / 1024;
    }
}

// Updates histories of the move pairs formed by moves
// at ply -1, -2, -3, -4, and -6 with current move.
void update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus) {
    static constexpr std::array<ConthistBonus, 6> conthist_bonuses = {
      {{1, 1108}, {2, 652}, {3, 273}, {4, 572}, {5, 126}, {6, 449}}};

    for (const auto [i, weight] : conthist_bonuses) {
        // Only update the first 2 continuation histories if we are in check
        if (ss->inCheck && i > 2)
            break;
        if (((ss - i)->currentMove).is_ok())
            (*(ss - i)->continuationHistory)[pc][to] << (bonus * weight / 1024) + 80 * (i < 2);
    }
}

// Updates move sorting heuristics
void update_quiet_histories(
  const Position& pos, Stack* ss, Search::Worker& workerThread, Move move, int bonus) {

    Color us = pos.side_to_move();
    workerThread.mainHistory[us][move.from_to()] << bonus;

    if (ss->ply < LOW_PLY_HISTORY_SIZE)
        workerThread.lowPlyHistory[ss->ply][move.from_to()] << (bonus * 771 / 1024) + 40;

    update_continuation_histories(ss, pos.moved_piece(move), move.to_sq(),
                                  bonus * (bonus > 0 ? 979 : 842) / 1024);

    int pIndex = pawn_history_index(pos);
    workerThread.pawnHistory[pIndex][pos.moved_piece(move)][move.to_sq()]
      << (bonus * (bonus > 0 ? 704 : 439) / 1024) + 70;
}

}

// When playing with strength handicap, choose the best move among a set of
// RootMoves using a statistical rule dependent on 'level'.
Move Skill::pick_best(const RootMoves& rootMoves, size_t multiPV) {
    static PRNG rng(now());

    // RootMoves are already sorted by score in descending order
    Value  topScore = rootMoves[0].score;
    int    delta    = std::min(topScore - rootMoves[multiPV - 1].score, int(PawnValue));
    int    maxScore = -VALUE_INFINITE;
    double weakness = 120 - 2 * level;

    // Choose best move. For each move score we add two terms, both dependent on
    // weakness. One is deterministic and bigger for weaker levels, and one is
    // random. Then we choose the move with the resulting highest score.
    for (size_t i = 0; i < multiPV; ++i) {
        // This is our magic formula
        int push = int(weakness * int(topScore - rootMoves[i].score)
                       + delta * (rng.rand<unsigned>() % int(weakness)))
                 / 128;

        if (rootMoves[i].score + push >= maxScore) {
            maxScore = rootMoves[i].score + push;
            best     = rootMoves[i].pv[0];
        }
    }

    return best;
}

// Used to print debug info and, more importantly, to detect
// when we are out of available time and thus stop the search.
void SearchManager::check_time(Search::Worker& worker) {
    if (--callsCnt > 0)
        return;

    // When using nodes, ensure checking rate is not lower than 0.1% of nodes
    callsCnt = worker.limits.nodes ? std::min(512, int(worker.limits.nodes / 1024)) : 512;

    static TimePoint lastInfoTime = now();

    TimePoint elapsed = tm.elapsed([&worker]() { return worker.threads.nodes_searched(); });
    TimePoint tick    = worker.limits.startTime + elapsed;

    if (tick - lastInfoTime >= 1000) {
        lastInfoTime = tick;
        dbg_print();
    }

    // We should not stop pondering until told so by the GUI
    if (ponder)
        return;

    if (
      // Later we rely on the fact that we can at least use the mainthread previous
      // root-search score and PV in a multithreaded environment to prove mated-in scores.
      worker.completedDepth >= 1
      && ((worker.limits.use_time_management() && (elapsed > tm.maximum() || stopOnPonderhit))
          || (worker.limits.movetime && elapsed >= worker.limits.movetime)
          || (worker.limits.nodes && worker.threads.nodes_searched() >= worker.limits.nodes)))
        worker.threads.stop = worker.threads.abortedSearch = true;
}

// PV output implementation
void SearchManager::pv(Search::Worker&   worker,
                       const ThreadPool& threads,
#ifdef USE_NUMA_TT
                       Depth             depth) {
#else
                       const TranspositionTable& tt,
                       Depth                     depth) {
#endif

    const auto nodes     = threads.nodes_searched();
    auto&      rootMoves = worker.rootMoves;
    auto&      pos       = worker.rootPos;
    size_t     pvIdx     = worker.pvIdx;
    size_t     multiPV   = std::min(size_t(worker.options["MultiPV"]), rootMoves.size());
    uint64_t   tbHits    = threads.tb_hits() + (worker.tbConfig.rootInTB ? rootMoves.size() : 0);

    for (size_t i = 0; i < multiPV; ++i) {
        bool updated = rootMoves[i].score != -VALUE_INFINITE;

        if (depth == 1 && !updated && i > 0)
            continue;

        Depth d = updated ? depth : std::max(1, depth - 1);
        Value v = updated ? rootMoves[i].uciScore : rootMoves[i].previousScore;

        if (v == -VALUE_INFINITE)
            v = VALUE_ZERO;

        bool tb = worker.tbConfig.rootInTB && std::abs(v) <= VALUE_TB;
        v       = tb ? rootMoves[i].tbScore : v;

        bool isExact = i != pvIdx || tb || !updated;

        // Potentially correct and extend the PV, and in exceptional cases v
        if (is_decisive(v) && std::abs(v) < VALUE_MATE_IN_MAX_PLY
            && ((!rootMoves[i].scoreLowerbound && !rootMoves[i].scoreUpperbound) || isExact))
            syzygy_extend_pv(worker.options, worker.limits, pos, rootMoves[i], v);

        std::string pv;
        for (Move m : rootMoves[i].pv)
            pv += UCIEngine::move(m, pos.is_chess960()) + " ";

        // Remove last whitespace
        if (!pv.empty())
            pv.pop_back();

        auto wdl   = worker.options["UCI_ShowWDL"] ? UCIEngine::wdl(v, pos) : "";
        auto bound = rootMoves[i].scoreLowerbound
                     ? "lowerbound"
                     : (rootMoves[i].scoreUpperbound ? "upperbound" : "");

        InfoFull info;

        info.depth    = d;
        info.selDepth = rootMoves[i].selDepth;
        info.multiPV  = i + 1;
        info.score    = {v, pos};
        info.wdl      = wdl;

        if (!isExact)
            info.bound = bound;

        TimePoint time = std::max(TimePoint(1), tm.elapsed_time());
        info.timeMs    = time;
        info.nodes     = nodes;
        info.nps       = nodes * 1000 / time;
        info.tbHits    = tbHits;
        info.pv        = pv;
#ifdef USE_NUMA_TT
        info.hashfull  = worker.l1_tt ? worker.l1_tt->hashfull() : 0;
#else
        info.hashfull  = tt.hashfull();
#endif

        updates.onUpdateFull(info);
    }
}

// Called in case we have no ponder move before exiting the search,
// for instance, in case we stop the search during a fail high at root.
// We try hard to have a ponder move to return to the GUI,
// otherwise in case of 'ponder on' we have nothing to think about.
bool RootMove::extract_ponder_from_tt(const TranspositionTable& tt, Position& pos) {

    StateInfo st;

    assert(pv.size() == 1);
    if (pv[0] == Move::none())
        return false;

#ifdef USE_NUMA_TT
    pos.do_move(pv[0], st, nullptr);
#else
    pos.do_move(pv[0], st, &tt);
#endif

    auto [ttHit, ttData, ttWriter] = tt.probe(pos.key());
    if (ttHit) {
        if (MoveList<LEGAL>(pos).contains(ttData.move))
            pv.push_back(ttData.move);
    }

    pos.undo_move(pv[0]);
    return pv.size() > 1;
}

// Used to correct and extend PVs for moves that have a TB (but not a mate) score.
void syzygy_extend_pv(const OptionsMap&         options,
                      const Search::LimitsType& limits,
                      Position&                 pos,
                      RootMove&                 rootMove,
                      Value&                    v) {

    auto t_start      = std::chrono::steady_clock::now();
    int  moveOverhead = int(options["Move Overhead"]);
    bool rule50       = bool(options["Syzygy50MoveRule"]);

    // Do not use more than moveOverhead / 2 time, if time management is active
    auto time_abort = [&t_start, &moveOverhead, &limits]() -> bool {
        auto t_end = std::chrono::steady_clock::now();
        return limits.use_time_management()
            && 2 * std::chrono::duration<double, std::milli>(t_end - t_start).count()
                 > moveOverhead;
    };

    std::list<StateInfo> sts;

    // Step 0, do the rootMove, no correction allowed, as needed for MultiPV in TB.
    auto& stRoot = sts.emplace_back();
    pos.do_move(rootMove.pv[0], stRoot);
    int ply = 1;

    // Step 1, walk the PV to the last position in TB with correct decisive score
    while (size_t(ply) < rootMove.pv.size()) {
        Move& pvMove = rootMove.pv[ply];

        RootMoves legalMoves;
        for (const auto& m : MoveList<LEGAL>(pos))
            legalMoves.emplace_back(m);

        Tablebases::Config config = Tablebases::rank_root_moves(options, pos, legalMoves);
        RootMove&          rm     = *std::find(legalMoves.begin(), legalMoves.end(), pvMove);

        if (legalMoves[0].tbRank != rm.tbRank)
            break;

        ply++;

        auto& st = sts.emplace_back();
        pos.do_move(pvMove, st);

        // Do not allow for repetitions or drawing moves along the PV in TB regime
        if (config.rootInTB && ((rule50 && pos.is_draw(ply)) || pos.is_repetition(ply))) {
            pos.undo_move(pvMove);
            ply--;
            break;
        }

        // Full PV shown will thus be validated and end in TB.
        // If we cannot validate the full PV in time, we do not show it.
        if (config.rootInTB && time_abort())
            break;
    }

    // Resize the PV to the correct part
    rootMove.pv.resize(ply);

    // Step 2, now extend the PV to mate
    while (!(rule50 && pos.is_draw(0))) {
        if (time_abort())
            break;

        RootMoves legalMoves;
        for (const auto& m : MoveList<LEGAL>(pos)) {
            auto&     rm = legalMoves.emplace_back(m);
            StateInfo tmpSI;
            pos.do_move(m, tmpSI);
            // Give a score of each move to break DTZ ties
            for (const auto& mOpp : MoveList<LEGAL>(pos))
                rm.tbRank -= pos.capture(mOpp) ? 100 : 1;
            pos.undo_move(m);
        }

        // Mate found
        if (legalMoves.size() == 0)
            break;

        // Sort moves according to their above assigned rank.
        std::stable_sort(
          legalMoves.begin(), legalMoves.end(),
          [](const Search::RootMove& a, const Search::RootMove& b) { return a.tbRank > b.tbRank; });

        // The winning side tries to minimize DTZ, the losing side maximizes it
        Tablebases::Config config = Tablebases::rank_root_moves(options, pos, legalMoves, true);

        // If DTZ is not available we might not find a mate, so we bail out
        if (!config.rootInTB || config.cardinality > 0)
            break;

        ply++;

        Move& pvMove = legalMoves[0].pv[0];
        rootMove.pv.push_back(pvMove);
        auto& st = sts.emplace_back();
        pos.do_move(pvMove, st);
    }

    // Finding a draw in this function is an exceptional case
    if (pos.is_draw(0))
        v = VALUE_DRAW;

    // Undo the PV moves
    for (auto it = rootMove.pv.rbegin(); it != rootMove.pv.rend(); ++it)
        pos.undo_move(*it);

    // Inform if we couldn't get a full extension in time
    if (time_abort())
        sync_cout
          << "info string Syzygy based PV extension requires more time, increase Move Overhead as needed."
          << sync_endl;
}

}  // namespace Stockfish
