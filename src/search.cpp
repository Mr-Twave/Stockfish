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

#include "bitboard.h"
#include "evaluate.h"
#include "history.h"
#include "misc.h"
#include "movegen.h"
#include "movepick.h"
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"
#include "position.h"
#include "searchpolicies.h"
#include "syzygy/tbprobe.h"
#include "thread.h"
#include "timeman.h"
#include "tt.h"
#include "uci.h"
#include "ucioption.h"

namespace Stockfish {

namespace TB = Tablebases;

void syzygy_extend_pv(const OptionsMap&         options,
                      const Search::LimitsType& limits,
                      Stockfish::Position&      pos,
                      Stockfish::Search::RootMove& rootMove,
                      Value&                    v);

using namespace Search;

namespace {

// Compile-time selection of TT architecture
#ifdef USE_NUMA_TT
constexpr bool IsNumaTT = true;
#else
constexpr bool IsNumaTT = false;
#endif

constexpr int SEARCHEDLIST_CAPACITY = 32;
using SearchedList = ValueList<Move, SEARCHEDLIST_CAPACITY>;

// (*Scalers):
// The values with Scaler asterisks have proven non-linear scaling.
// They are optimized to time controls of 180 + 1.8 and longer.

int correction_value(const Worker& w, const Position& pos, const Stack* const ss) {
    const Color us    = pos.side_to_move();
    const auto  m     = (ss - 1)->currentMove;
    const auto  pcv   = w.pawnCorrectionHistory[pawn_correction_history_index(pos)][us];
    const auto  micv  = w.minorPieceCorrectionHistory[minor_piece_index(pos)][us];
    const auto  wnpcv = w.nonPawnCorrectionHistory[non_pawn_index<WHITE>(pos)][WHITE][us];
    const auto  bnpcv = w.nonPawnCorrectionHistory[non_pawn_index<BLACK>(pos)][BLACK][us];
    const auto  cntcv =
      m.is_ok() ? (*(ss - 2)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()]
                : 0;

    return 8867 * pcv + 8136 * micv + 10757 * (wnpcv + bnpcv) + 7232 * cntcv;
}

// Add correctionHistory value to raw staticEval and guarantee evaluation
// does not hit the tablebase range.
Value to_corrected_static_eval(const Value v, const int cv) {
    return std::clamp(v + cv / 131072, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
}

void update_correction_history(const Position& pos,
                               Stack* const    ss,
                               Search::Worker& workerThread,
                               const int       bonus) {
    const Move  m  = (ss - 1)->currentMove;
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

// Add a small random component to draw evaluations to avoid 3-fold blindness
Value value_draw(size_t nodes) { return VALUE_DRAW - 1 + Value(nodes & 0x2); }
Value value_to_tt(Value v, int ply);
Value value_from_tt(Value v, int ply, int r50c);
void  update_pv(Move* pv, Move move, const Move* childPv);
void  update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus);
void  update_quiet_histories(
   const Position& pos, Stack* ss, Search::Worker& workerThread, Move move, int bonus);
void update_all_stats(const Position& pos,
                      Stack* ss,
                      Search::Worker& workerThread,
                      Move            bestMove,
                      Square          prevSq,
                      SearchedList&   quietsSearched,
                      SearchedList&   capturesSearched,
                      Depth           depth,
                      Move            TTMove,
                      int             moveCount);

}  // namespace

Worker::Worker(SharedState&                sharedState,
               std::unique_ptr<ISearchManager> sm,
               size_t                          threadId,
               NumaReplicatedAccessToken       token) :
    // Unpack the SharedState struct into member variables
    threadIdx(threadId),
    numaAccessToken(token),
    manager(std::move(sm)),
    options(sharedState.options),
    threads(sharedState.threads),
    networks(sharedState.networks),
    refreshTable(networks[token])
#ifdef USE_NUMA_TT
    ,
    generation8(sharedState.generation8)
#else
    ,
    tt(sharedState.tt)
#endif
{
    clear();
}

void Worker::ensure_network_replicated() {
    // Access once to force lazy initialization.
    // We do this because we want to avoid initialization during search.
    (void) (networks[numaAccessToken]);
}

void Worker::start_searching() {

    accumulatorStack.reset();

    // Non-main threads go directly to iterative_deepening()
    if (!is_mainthread())
    {
        iterative_deepening();
        return;
    }

    main_manager()->tm.init(limits, rootPos.side_to_move(), rootPos.game_ply(), options,
                            main_manager()->originalTimeAdjust);
    
#ifdef USE_NUMA_TT
    if (is_mainthread())
        threads.generation8 += GENERATION_DELTA;
#else
    tt.new_search();
#endif

    if (rootMoves.empty())
    {
        rootMoves.emplace_back(Move::none());
        main_manager()->updates.onUpdateNoMoves(
          {0, {rootPos.checkers() ? -VALUE_MATE : VALUE_DRAW, rootPos}});
    }
    else
    {
        threads.start_searching();
        iterative_deepening();
    }

    // When we reach the maximum depth, we can arrive here without a raise of
    // threads.stop. However, if we are pondering or in an infinite search,
    // the UCI protocol states that we shouldn't print the best move before the
    // GUI sends a "stop" or "ponderhit" command.
    while (!threads.stop && (main_manager()->ponder || limits.infinite))
    {}  // Busy wait for a stop or a ponder reset

    // Stop the threads if not already stopped
    threads.stop = true;

    // Wait until all threads have finished
    threads.wait_for_search_finished();

    // When playing in 'nodes as time' mode, subtract the searched nodes
    if (limits.npmsec)
        main_manager()->tm.advance_nodes_time(threads.nodes_searched()
                                              - limits.inc[rootPos.side_to_move()]);

    Worker* bestThread = this;
    Skill   skill =
      Skill(options["Skill Level"], options["UCI_LimitStrength"] ? int(options["UCI_Elo"]) : 0);

    if (int(options["MultiPV"]) == 1 && !limits.depth && !limits.mate && !skill.enabled()
        && rootMoves[0].pv[0] != Move::none())
        bestThread = threads.get_best_thread()->worker.get();

    main_manager()->bestPreviousScore        = bestThread->rootMoves[0].score;
    main_manager()->bestPreviousAverageScore = bestThread->rootMoves[0].averageScore;

    // Send again PV info if we have a new best thread
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

// Main iterative deepening loop
void Worker::iterative_deepening() {

    SearchManager* mainThread = (is_mainthread() ? main_manager() : nullptr);

    Move pv[MAX_PLY + 1];

    Depth lastBestMoveDepth = 0;
    Value lastBestScore     = -VALUE_INFINITE;
    auto  lastBestPV        = std::vector{Move::none()};

    Value  alpha, beta;
    Value  bestValue     = -VALUE_INFINITE;
    Color  us            = rootPos.side_to_move();
    double timeReduction = 1, totBestMoveChanges = 0;
    int    delta, iterIdx = 0;

    Stack  stack[MAX_PLY + 10] = {};
    Stack* ss                  = stack + 7;

    for (int i = 7; i > 0; --i)
    {
        (ss - i)->continuationHistory =
          &continuationHistory[0][0][NO_PIECE][0];  // Use as a sentinel
        (ss - i)->continuationCorrectionHistory = &continuationCorrectionHistory[NO_PIECE][0];
        (ss - i)->staticEval                    = VALUE_NONE;
    }

    for (int i = 0; i <= MAX_PLY + 2; ++i)
        (ss + i)->ply = i;

    ss->pv = pv;

    if (mainThread)
    {
        if (mainThread->bestPreviousScore == VALUE_INFINITE)
            mainThread->iterValue.fill(VALUE_ZERO);
        else
            mainThread->iterValue.fill(mainThread->bestPreviousScore);
    }

    size_t multiPV = size_t(options["MultiPV"]);
    Skill skill(options["Skill Level"], options["UCI_LimitStrength"] ? int(options["UCI_Elo"]) : 0);

    // When playing with strength handicap enable MultiPV search
    if (skill.enabled())
        multiPV = std::max(multiPV, size_t(4));

    multiPV = std::min(multiPV, rootMoves.size());

    int searchAgainCounter = 0;
    lowPlyHistory.fill(89);

    // Iterative deepening loop
    while (++rootDepth < MAX_PLY && !threads.stop
           && !(limits.depth && mainThread && rootDepth > limits.depth))
    {
        // Age out PV variability metric
        if (mainThread)
            totBestMoveChanges /= 2;

        // Save the last iteration's scores
        for (RootMove& rm : rootMoves)
            rm.previousScore = rm.score;

        size_t pvFirst = 0;
        pvLast         = 0;

        if (!threads.increaseDepth)
            searchAgainCounter++;

        // MultiPV loop
        for (pvIdx = 0; pvIdx < multiPV; ++pvIdx)
        {
            if (pvIdx == pvLast)
            {
                pvFirst = pvLast;
                for (pvLast++; pvLast < rootMoves.size(); pvLast++)
                    if (rootMoves[pvLast].tbRank != rootMoves[pvFirst].tbRank)
                        break;
            }

            // Reset selDepth for each PV line
            selDepth = 0;

            // Reset aspiration window
            delta     = 5 + std::abs(rootMoves[pvIdx].meanSquaredScore) / 11131;
            Value avg = rootMoves[pvIdx].averageScore;
            alpha     = std::max(avg - delta, -VALUE_INFINITE);
            beta      = std::min(avg + delta, VALUE_INFINITE);

            // Adjust optimism
            optimism[us]  = 136 * avg / (std::abs(avg) + 93);
            optimism[~us] = -optimism[us];

            // Aspiration search
            int failedHighCnt = 0;
            while (true)
            {
                Depth adjustedDepth =
                  std::max(1, rootDepth - failedHighCnt - 3 * (searchAgainCounter + 1) / 4);
                rootDelta = beta - alpha;

                // Main search call with policy-based design
                bestValue = search<Root,
                                   DefaultRazoringPolicy,
                                   DefaultFutilityPolicy,
                                   DefaultNMPolicy,
                                   DefaultLMRPolicy,
                                   DefaultSEEPolicy,
                                   DefaultExtensionPolicy>(rootPos, ss, alpha, beta, adjustedDepth, false);

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

                // Aspiration window adjustment
                if (bestValue <= alpha)
                {
                    beta  = (3 * alpha + beta) / 4;
                    alpha = std::max(bestValue - delta, -VALUE_INFINITE);

                    failedHighCnt = 0;
                    if (mainThread)
                        mainThread->stopOnPonderhit = false;
                }
                else if (bestValue >= beta)
                {
                    beta = std::min(bestValue + delta, VALUE_INFINITE);
                    ++failedHighCnt;
                }
                else
                    break;

                delta += delta / 3;

                assert(alpha >= -VALUE_INFINITE && beta <= VALUE_INFINITE);
            }

            // Sort PV lines
            std::stable_sort(rootMoves.begin() + pvFirst, rootMoves.begin() + pvIdx + 1);

            if (mainThread
                && (threads.stop || pvIdx + 1 == multiPV || nodes > 10000000)
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

        // Handle unproven mated-in scores
        if (threads.abortedSearch && rootMoves[0].score != -VALUE_INFINITE
            && is_loss(rootMoves[0].score))
        {
            Utility::move_to_front(rootMoves, [&lastBestPV = std::as_const(lastBestPV)](
                                                const auto& rm) { return rm == lastBestPV[0]; });
            rootMoves[0].pv    = lastBestPV;
            rootMoves[0].score = rootMoves[0].uciScore = lastBestScore;
        }
        else if (rootMoves[0].pv[0] != lastBestPV[0])
        {
            lastBestPV        = rootMoves[0].pv;
            lastBestScore     = rootMoves[0].score;
            lastBestMoveDepth = rootDepth;
        }

        if (!mainThread)
            continue;

        // Check for mate
        if (limits.mate && rootMoves[0].score == rootMoves[0].uciScore
            && ((rootMoves[0].score >= VALUE_MATE_IN_MAX_PLY
                 && VALUE_MATE - rootMoves[0].score <= 2 * limits.mate)
                || (rootMoves[0].score != -VALUE_INFINITE
                    && rootMoves[0].score <= VALUE_MATED_IN_MAX_PLY
                    && VALUE_MATE + rootMoves[0].score <= 2 * limits.mate)))
            threads.stop = true;

        // Skill level
        if (skill.enabled() && skill.time_to_pick(rootDepth))
            skill.pick_best(rootMoves, multiPV);

        // Count best move changes
        for (auto&& th : threads)
        {
            totBestMoveChanges += th->worker->bestMoveChanges;
            th->worker->bestMoveChanges = 0;
        }

        // Time management
        if (limits.use_time_management() && !threads.stop && !mainThread->stopOnPonderhit)
        {
            uint64_t nodesEffort =
              rootMoves[0].effort * 100000 / std::max(size_t(1), size_t(nodes));

            double fallingEval =
              (11.396 + 2.035 * (mainThread->bestPreviousAverageScore - bestValue)
               + 0.968 * (mainThread->iterValue[iterIdx] - bestValue))
              / 100.0;
            fallingEval = std::clamp(fallingEval, 0.5786, 1.6752);

            // Time reduction
            double k      = 0.527;
            double center = lastBestMoveDepth + 11;
            timeReduction = 0.8 + 0.84 / (1.077 + std::exp(-k * (completedDepth - center)));
            double reduction =
              (1.4540 + mainThread->previousTimeReduction) / (2.1593 * timeReduction);
            double bestMoveInstability = 0.9929 + 1.8519 * totBestMoveChanges / threads.size();

            double totalTime =
              mainThread->tm.optimum() * fallingEval * reduction * bestMoveInstability;

            // Cap for single legal move
            if (rootMoves.size() == 1)
                totalTime = std::min(500.0, totalTime);

            auto elapsedTime = elapsed();

            if (completedDepth >= 10 && nodesEffort >= 97056 && elapsedTime > totalTime * 0.6540
                && !mainThread->ponder)
                threads.stop = true;

            // Stop search if time exceeded
            if (elapsedTime > std::min(totalTime, double(mainThread->tm.maximum())))
            {
                if (mainThread->ponder)
                    mainThread->stopOnPonderhit = true;
                else
                    threads.stop = true;
            }
            else
                threads.increaseDepth = mainThread->ponder || elapsedTime <= totalTime * 0.5138;
        }

        mainThread->iterValue[iterIdx] = bestValue;
        iterIdx                        = (iterIdx + 1) & 3;
    }

    if (!mainThread)
        return;

    mainThread->previousTimeReduction = timeReduction;

    // Apply skill level
    if (skill.enabled())
        std::swap(rootMoves[0],
                  *std::find(rootMoves.begin(), rootMoves.end(),
                             skill.best ? skill.best : skill.pick_best(rootMoves, multiPV)));
}

// Main search function with policy-based design
template<NodeType NT,
         typename RazoringPolicy,
         typename FutilityPolicy,
         typename NMPolicy,
         typename LMRPolicy,
         typename SEEPolicy,
         typename ExtensionPolicy>
Value Worker::search(
  Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode) {

    constexpr bool PvNode   = NT != NonPV;
    constexpr bool rootNode = NT == Root;
    const bool     allNode  = !(PvNode || cutNode);

    // Dive into quiescence search when the depth reaches zero
    if (depth <= 0)
    {
        constexpr auto nextNodeType = PvNode ? PV : NonPV;
        return qsearch<nextNodeType, DefaultSEEPolicy>(pos, ss, alpha, beta);
    }

    // Limit the depth
    depth = std::min(depth, MAX_PLY - 1);

    // Check for upcoming repetition
    if (!rootNode && alpha < VALUE_DRAW && pos.upcoming_repetition(ss->ply))
    {
        alpha = value_draw(nodes);
        if (alpha >= beta)
            return alpha;
    }

    assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));
    assert(0 < depth && depth < MAX_PLY);
    assert(!(PvNode && cutNode));

    Move      pv[MAX_PLY + 1];
    StateInfo st;

    Key   posKey;
    Move  move, excludedMove, bestMove;
    Depth extension, newDepth;
    Value bestValue, value, eval, maxValue, probCutBeta;
    bool  givesCheck, improving, priorCapture, opponentWorsening;
    bool  capture, ttCapture;
    int   priorReduction;
    Piece movedPiece;

    SearchedList capturesSearched;
    SearchedList quietsSearched;

    // Step 1. Initialize node
    ss->inCheck   = pos.checkers();
    priorCapture  = pos.captured_piece();
    Color us      = pos.side_to_move();
    ss->moveCount = 0;
    bestValue     = -VALUE_INFINITE;
    maxValue      = VALUE_INFINITE;

    // Check for available time
    if (is_mainthread())
        main_manager()->check_time(*this);

    // Update selDepth
    if (PvNode && selDepth < ss->ply + 1)
        selDepth = ss->ply + 1;

    if (!rootNode)
    {
        // Step 2. Check for aborted search and immediate draw
        if (threads.stop.load(std::memory_order_relaxed) || pos.is_draw(ss->ply)
            || ss->ply >= MAX_PLY)
            return (ss->ply >= MAX_PLY && !ss->inCheck) ? evaluate(pos) : value_draw(nodes);

        // Step 3. Mate distance pruning
        alpha = std::max(mated_in(ss->ply), alpha);
        beta  = std::min(mate_in(ss->ply + 1), beta);
        if (alpha >= beta)
            return alpha;
    }

    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    Square prevSq  = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;
    bestMove       = Move::none();
    priorReduction = (ss - 1)->reduction;
    (ss - 1)->reduction = 0;
    ss->statScore       = 0;
    (ss + 2)->cutoffCnt = 0;

    // Step 4. Transposition table lookup
    excludedMove = ss->excludedMove;
    posKey       = pos.key();
    
    // TT probe with NUMA support
    TTData ttData(Move::none(), VALUE_NONE, VALUE_NONE, DEPTH_NONE, BOUND_NONE, false);
    TTWriter ttWriter(nullptr);
    bool ttHit;

#ifdef USE_NUMA_TT
    if (l1_tt && l2_tt)
    {
        // Hierarchical probe: L1 first, then L2 on miss
        prefetch(l1_tt->first_entry(posKey));
        prefetch(l2_tt->first_entry(posKey));

        auto l1res = l1_tt->probe(posKey);
        ttHit = l1res.hit;
        ttData = l1res.data;
        ttWriter = l1res.writer;

        if (!ttHit)
        {
            auto l2res = l2_tt->probe(posKey);
            if (l2res.hit)
            {
                ttHit = true;
                ttData = l2res.data;
                // Promote from L2 to L1
                l1res.writer.write(posKey, ttData.value, ttData.is_pv, ttData.bound, 
                                   ttData.depth, ttData.move, ttData.eval, generation8);
            }
        }
    }
    else
    {
        ttHit = false;
    }
#else
    auto res = tt.probe(posKey);
    ttHit = res.hit;
    ttData = res.data;
    ttWriter = res.writer;
#endif

    ss->ttHit    = ttHit;
    ttData.move  = rootNode ? rootMoves[pvIdx].pv[0] : ttHit ? ttData.move : Move::none();
    ttData.value = ttHit ? value_from_tt(ttData.value, ss->ply, pos.rule50_count()) : VALUE_NONE;
    ss->ttPv     = excludedMove ? ss->ttPv : PvNode || (ttHit && ttData.is_pv);
    ttCapture    = ttData.move && pos.capture_stage(ttData.move);

    // TT cutoff check
    if (!PvNode && !excludedMove && ttData.depth > depth - (ttData.value <= beta)
        && is_valid(ttData.value)
        && (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER))
        && (cutNode == (ttData.value >= beta) || depth > 5))
    {
        // Update move sorting heuristics
        if (ttData.move && ttData.value >= beta)
        {
            if (!ttCapture)
                update_quiet_histories(pos, ss, *this, ttData.move,
                                       std::min(127 * depth - 74, 1063));

            if (prevSq != SQ_NONE && (ss - 1)->moveCount <= 3 && !priorCapture)
                update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -2128);
        }

        // Graph history interaction workaround
        if (pos.rule50_count() < 91)
        {
            if (depth >= 8 && ttData.move && pos.pseudo_legal(ttData.move) && pos.legal(ttData.move)
                && !is_decisive(ttData.value))
            {
                pos.do_move(ttData.move, st);
                Key nextPosKey = pos.key();
                
                TTData nextTTData(Move::none(), VALUE_NONE, VALUE_NONE, DEPTH_NONE, BOUND_NONE, false);
#ifdef USE_NUMA_TT
                if (l1_tt)
                {
                    auto l1res = l1_tt->probe(nextPosKey);
                    nextTTData = l1res.data;
                    if (!l1res.hit && l2_tt)
                    {
                        auto l2res = l2_tt->probe(nextPosKey);
                        if (l2res.hit) nextTTData = l2res.data;
                    }
                }
#else
                nextTTData = tt.probe(nextPosKey).data;
#endif

                pos.undo_move(ttData.move);

                if (!is_valid(nextTTData.value))
                    return ttData.value;
                if ((ttData.value >= beta) == (-nextTTData.value >= beta))
                    return ttData.value;
            }
            else
                return ttData.value;
        }
    }

    // Step 5. Tablebases probe
    if (!rootNode && !excludedMove && tbConfig.cardinality)
    {
        int piecesCount = pos.count<ALL_PIECES>();

        if (piecesCount <= tbConfig.cardinality
            && (piecesCount < tbConfig.cardinality || depth >= tbConfig.probeDepth)
            && pos.rule50_count() == 0 && !pos.can_castle(ANY_CASTLING))
        {
            TB::ProbeState err;
            TB::WDLScore   wdl = Tablebases::probe_wdl(pos, &err);

            if (is_mainthread())
                main_manager()->callsCnt = 0;

            if (err != TB::ProbeState::FAIL)
            {
                tbHits.fetch_add(1, std::memory_order_relaxed);

                int drawScore = tbConfig.useRule50 ? 1 : 0;
                Value tbValue = VALUE_TB - ss->ply;

                value = wdl < -drawScore ? -tbValue
                      : wdl > drawScore  ? tbValue
                                         : VALUE_DRAW + 2 * wdl * drawScore;

                Bound b = wdl < -drawScore ? BOUND_UPPER
                        : wdl > drawScore  ? BOUND_LOWER
                                           : BOUND_EXACT;

                if (b == BOUND_EXACT || (b == BOUND_LOWER ? value >= beta : value <= alpha))
                {
#ifdef USE_NUMA_TT
                    uint8_t gen = generation8.load();
#else
                    uint8_t gen = tt.generation();
#endif
                    ttWriter.write(posKey, value_to_tt(value, ss->ply), ss->ttPv, b,
                                   std::min(MAX_PLY - 1, depth + 6), Move::none(), VALUE_NONE,
                                   gen);
                    return value;
                }

                if (PvNode)
                {
                    if (b == BOUND_LOWER)
                        bestValue = value, alpha = std::max(alpha, bestValue);
                    else
                        maxValue = value;
                }
            }
        }
    }

    // Step 6. Static evaluation
    Value      unadjustedStaticEval = VALUE_NONE;
    const auto correctionValue      = correction_value(*this, pos, ss);
    
    if (ss->inCheck)
    {
        ss->staticEval = eval = (ss - 2)->staticEval;
        improving             = false;
        goto moves_loop;
    }
    else if (excludedMove)
        unadjustedStaticEval = eval = ss->staticEval;
    else if (ss->ttHit)
    {
        // Never assume anything about values stored in TT
        unadjustedStaticEval = ttData.eval;
        if (!is_valid(unadjustedStaticEval))
            unadjustedStaticEval = evaluate(pos);

        ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue);

        // ttValue can be used as a better position evaluation
        if (is_valid(ttData.value)
            && (ttData.bound & (ttData.value > eval ? BOUND_LOWER : BOUND_UPPER)))
            eval = ttData.value;
    }
    else
    {
        unadjustedStaticEval = evaluate(pos);
        ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue);

        // Save static evaluation
#ifdef USE_NUMA_TT
        uint8_t gen = generation8.load();
#else
        uint8_t gen = tt.generation();
#endif
        ttWriter.write(posKey, VALUE_NONE, ss->ttPv, BOUND_NONE, DEPTH_UNSEARCHED, Move::none(),
                       unadjustedStaticEval, gen);
    }

    // Use static evaluation difference for quiet move ordering
    if (((ss - 1)->currentMove).is_ok() && !(ss - 1)->inCheck && !priorCapture)
    {
        int bonus = std::clamp(-10 * int((ss - 1)->staticEval + ss->staticEval), -1979, 1561) + 630;
        mainHistory[~us][((ss - 1)->currentMove).from_to()] << bonus * 935 / 1024;
        if (!ttHit && type_of(pos.piece_on(prevSq)) != PAWN
            && ((ss - 1)->currentMove).type_of() != PROMOTION)
            pawnHistory[pawn_history_index(pos)][pos.piece_on(prevSq)][prevSq]
              << bonus * 1428 / 1024;
    }

    // Set up improving flag
    improving         = ss->staticEval > (ss - 2)->staticEval;
    opponentWorsening = ss->staticEval > -(ss - 1)->staticEval;

    if (priorReduction >= (depth < 10 ? 1 : 3) && !opponentWorsening)
        depth++;
    if (priorReduction >= 2 && depth >= 2 && ss->staticEval + (ss - 1)->staticEval > 177)
        depth--;
    
    // Create the context object for policy calls
    const NodeContext ctx(pos, ss, alpha, beta, depth, eval, cutNode, PvNode, improving,
                          opponentWorsening, correctionValue, nmpMinPly, (bool)excludedMove,
                          completedDepth, ttData, rootDepth);

    // Step 7. Razoring (Policy-based)
    if (RazoringPolicy::should_prune(ctx))
        return qsearch<NonPV, SEEPolicy>(pos, ss, alpha, beta);

    // Step 8. Futility pruning: child node (Policy-based)
    Value futilityMargin = FutilityPolicy::get_margin(ctx);
    if (FutilityPolicy::should_prune(ctx, futilityMargin))
        return beta + (eval - beta) / 3;

    // Step 9. Null move search (Policy-based)
    if (NMPolicy::should_prune(ctx))
    {
        assert((ss - 1)->currentMove != Move::null());

        Depth R = NMPolicy::get_reduction(depth);

        ss->currentMove                   = Move::null();
        ss->continuationHistory           = &continuationHistory[0][0][NO_PIECE][0];
        ss->continuationCorrectionHistory = &continuationCorrectionHistory[NO_PIECE][0];

        do_null_move(pos, st);

        Value nullValue = -search<NonPV, RazoringPolicy, FutilityPolicy, NMPolicy, 
                                  LMRPolicy, SEEPolicy, ExtensionPolicy>(
            pos, ss + 1, -beta, -beta + 1, depth - R, false);

        undo_null_move(pos);

        // Do not return unproven mate or TB scores
        if (nullValue >= beta && !is_win(nullValue))
        {
            if (nmpMinPly || depth < 16)
                return nullValue;

            assert(!nmpMinPly);  // Recursive verification is not allowed

            // Do verification search at high depths
            nmpMinPly = ss->ply + 3 * (depth - R) / 4;

            Value v = search<NonPV, RazoringPolicy, FutilityPolicy, NMPolicy, 
                             LMRPolicy, SEEPolicy, ExtensionPolicy>(
                pos, ss, beta - 1, beta, depth - R, false);

            nmpMinPly = 0;

            if (v >= beta)
                return nullValue;
        }
    }

    improving |= ss->staticEval >= beta;

    // Step 10. Internal iterative reductions
    if (!allNode && depth >= 6 && !ttData.move && priorReduction <= 3)
        depth--;

    // Step 11. ProbCut
    probCutBeta = beta + 215 - 60 * improving;
    if (depth >= 3
        && !is_decisive(beta)
        && !(is_valid(ttData.value) && ttData.value < probCutBeta))
    {
        assert(probCutBeta < VALUE_INFINITE && probCutBeta > beta);

        MovePicker mp(pos, ttData.move, probCutBeta - ss->staticEval, &captureHistory);
        Depth      probCutDepth = std::max(depth - 5, 0);

        while ((move = mp.next_move()) != Move::none())
        {
            assert(move.is_ok());

            if (move == excludedMove || !pos.legal(move))
                continue;

            assert(pos.capture_stage(move));

            movedPiece = pos.moved_piece(move);

            do_move(pos, move, st, ss);

            value = -qsearch<NonPV, SEEPolicy>(pos, ss + 1, -probCutBeta, -probCutBeta + 1);

            if (value >= probCutBeta && probCutDepth > 0)
                value = -search<NonPV, RazoringPolicy, FutilityPolicy, NMPolicy, 
                                LMRPolicy, SEEPolicy, ExtensionPolicy>(
                    pos, ss + 1, -probCutBeta, -probCutBeta + 1, probCutDepth, !cutNode);

            undo_move(pos, move);

            if (value >= probCutBeta)
            {
#ifdef USE_NUMA_TT
                uint8_t gen = generation8.load();
#else
                uint8_t gen = tt.generation();
#endif
                ttWriter.write(posKey, value_to_tt(value, ss->ply), ss->ttPv, BOUND_LOWER,
                               probCutDepth + 1, move, unadjustedStaticEval, gen);

                if (!is_decisive(value))
                    return value - (probCutBeta - beta);
            }
        }
    }

moves_loop:  // When in check, search starts here

    // Step 12. A small Probcut idea
    probCutBeta = beta + 417;
    if ((ttData.bound & BOUND_LOWER) && ttData.depth >= depth - 4 && ttData.value >= probCutBeta
        && !is_decisive(beta) && is_valid(ttData.value) && !is_decisive(ttData.value))
        return probCutBeta;

    const PieceToHistory* contHist[] = {
      (ss - 1)->continuationHistory, (ss - 2)->continuationHistory, (ss - 3)->continuationHistory,
      (ss - 4)->continuationHistory, (ss - 5)->continuationHistory, (ss - 6)->continuationHistory};

    MovePicker mp(pos, ttData.move, depth, &mainHistory, &lowPlyHistory, &captureHistory, contHist,
                  &pawnHistory, ss->ply);

    value = bestValue;

    int moveCount = 0;

    // Step 13. Loop through all pseudo-legal moves
    while ((move = mp.next_move()) != Move::none())
    {
        assert(move.is_ok());

        if (move == excludedMove)
            continue;

        // Check for legality
        if (!pos.legal(move))
            continue;

        // At root obey the "searchmoves" option
        if (rootNode && !std::count(rootMoves.begin() + pvIdx, rootMoves.begin() + pvLast, move))
            continue;

        ss->moveCount = ++moveCount;

        if (rootNode && is_mainthread() && nodes > 10000000)
        {
            main_manager()->updates.onIter(
              {depth, UCIEngine::move(move, pos.is_chess960()), moveCount + pvIdx});
        }
        if (PvNode)
            (ss + 1)->pv = nullptr;

        extension  = 0;
        capture    = pos.capture_stage(move);
        movedPiece = pos.moved_piece(move);
        givesCheck = pos.gives_check(move);

        (ss + 1)->quietMoveStreak = (!capture && !givesCheck) ? (ss->quietMoveStreak + 1) : 0;

        // Calculate new depth for this move
        newDepth = depth - 1;

        int delta = beta - alpha;

        Depth r = reduction(improving, depth, moveCount, delta);

        if (ss->ttPv)
            r += 931;

        // Step 14. Pruning at shallow depth
        if (!rootNode && pos.non_pawn_material(us) && !is_loss(bestValue))
        {
            // Skip quiet moves if movecount exceeds threshold
            if (moveCount >= (3 + depth * depth) / (2 - improving))
                mp.skip_quiet_moves();

            int lmrDepth = newDepth - r / 1024;

            if (capture || givesCheck)
            {
                Piece capturedPiece = pos.piece_on(move.to_sq());
                int   captHist = captureHistory[movedPiece][move.to_sq()][type_of(capturedPiece)];

                // Futility pruning for captures
                if (!givesCheck && lmrDepth < 7 && !ss->inCheck)
                {
                    Value futilityValue = ss->staticEval + 225 + 220 * lmrDepth
                                        + 275 * (move.to_sq() == prevSq) + PieceValue[capturedPiece]
                                        + 131 * captHist / 1024;
                    if (futilityValue <= alpha)
                        continue;
                }

                // SEE based pruning for captures (Policy-based)
                if (SEEPolicy::should_prune_capture(ctx, move, captHist))
                {
                    bool mayStalemateTrap =
                      depth > 2 && alpha < 0 && pos.non_pawn_material(us) == PieceValue[movedPiece]
                      && PieceValue[movedPiece] >= RookValue
                      && !(attacks_bb<KING>(pos.square<KING>(us)) & move.from_sq())
                      && !mp.can_move_king_or_pawn();

                    if (!mayStalemateTrap)
                        continue;
                }
            }
            else
            {
                int history = (*contHist[0])[movedPiece][move.to_sq()]
                            + (*contHist[1])[movedPiece][move.to_sq()]
                            + pawnHistory[pawn_history_index(pos)][movedPiece][move.to_sq()];

                // Continuation history based pruning
                if (history < -4361 * depth)
                    continue;

                history += 71 * mainHistory[us][move.from_to()] / 32;

                lmrDepth += history / 3233;

                Value baseFutility = (bestMove ? 46 : 230);
                Value futilityValue =
                  ss->staticEval + baseFutility + 131 * lmrDepth + 91 * (ss->staticEval > alpha);

                // Futility pruning: parent node
                if (!ss->inCheck && lmrDepth < 11 && futilityValue <= alpha)
                {
                    if (bestValue <= futilityValue && !is_decisive(bestValue)
                        && !is_win(futilityValue))
                        bestValue = futilityValue;
                    continue;
                }

                lmrDepth = std::max(lmrDepth, 0);

                // Prune moves with negative SEE (Policy-based)
                if (SEEPolicy::should_prune_quiet(ctx, move, lmrDepth))
                    continue;
            }
        }

        // Step 15. Extensions (Policy-based)
        if (ExtensionPolicy::should_apply_singular(ctx, move, ttData))
        {
            Value singularBeta  = ExtensionPolicy::get_singular_beta(ctx, ttData);
            Depth singularDepth = newDepth / 2;

            ss->excludedMove = move;
            value = search<NonPV, RazoringPolicy, FutilityPolicy, NMPolicy, 
                           LMRPolicy, SEEPolicy, ExtensionPolicy>(
                pos, ss, singularBeta - 1, singularBeta, singularDepth, cutNode);
            ss->excludedMove = Move::none();

            extension = ExtensionPolicy::get_singular_extension(ctx, value, singularBeta, ttData);
            
            if (value >= beta && !is_decisive(value))
                return value;
        }

        // Step 16. Make the move
        do_move(pos, move, st, givesCheck, ss);

        newDepth += extension;
        uint64_t nodeCount = rootNode ? uint64_t(nodes) : 0;

        // All reduction logic is now in the LMR Policy
        r = LMRPolicy::get_reduction(ctx, improving, depth, moveCount, delta, 
                                     PvNode, cutNode, ttData, ss, move, capture, ttCapture);

        // Step 17. Late moves reduction / extension (LMR)
        if (LMRPolicy::should_apply(ctx, moveCount))
        {
            Depth d = std::max(1, std::min(newDepth - r / 1024, newDepth + 1 + PvNode)) + PvNode;

            ss->reduction = newDepth - d;
            value         = -search<NonPV, RazoringPolicy, FutilityPolicy, NMPolicy, 
                                    LMRPolicy, SEEPolicy, ExtensionPolicy>(
                pos, ss + 1, -(alpha + 1), -alpha, d, true);
            ss->reduction = 0;

            // Do a full-depth search when reduced LMR search fails high
            if (value > alpha)
            {
                const bool doDeeperSearch = d < newDepth && value > (bestValue + 43 + 2 * newDepth);
                const bool doShallowerSearch = value < bestValue + 9;

                newDepth += doDeeperSearch - doShallowerSearch;

                if (newDepth > d)
                    value = -search<NonPV, RazoringPolicy, FutilityPolicy, NMPolicy, 
                                    LMRPolicy, SEEPolicy, ExtensionPolicy>(
                        pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);

                // Post LMR continuation history updates
                update_continuation_histories(ss, movedPiece, move.to_sq(), 1412);
            }
        }

        // Step 18. Full-depth search when LMR is skipped
        else if (!PvNode || moveCount > 1)
        {
            value = -search<NonPV, RazoringPolicy, FutilityPolicy, NMPolicy, 
                            LMRPolicy, SEEPolicy, ExtensionPolicy>(
                pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);
        }

        // For PV nodes only, do a full PV search
        if (PvNode && (moveCount == 1 || value > alpha))
        {
            (ss + 1)->pv    = pv;
            (ss + 1)->pv[0] = Move::none();

            if (move == ttData.move && rootDepth > 8)
                newDepth = std::max(newDepth, 1);

            value = -search<PV, RazoringPolicy, FutilityPolicy, NMPolicy, 
                            LMRPolicy, SEEPolicy, ExtensionPolicy>(
                pos, ss + 1, -beta, -alpha, newDepth, false);
        }

        // Step 19. Undo move
        undo_move(pos, move);

        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

        // Step 20. Check for a new best move
        if (threads.stop.load(std::memory_order_relaxed))
            return VALUE_ZERO;

        if (rootNode)
        {
            RootMove& rm = *std::find(rootMoves.begin(), rootMoves.end(), move);

            rm.effort += nodes - nodeCount;

            rm.averageScore =
              rm.averageScore != -VALUE_INFINITE ? (value + rm.averageScore) / 2 : value;

            rm.meanSquaredScore = rm.meanSquaredScore != -VALUE_INFINITE * VALUE_INFINITE
                                  ? (value * std::abs(value) + rm.meanSquaredScore) / 2
                                  : value * std::abs(value);

            // PV move or new best move?
            if (moveCount == 1 || value > alpha)
            {
                rm.score = rm.uciScore = value;
                rm.selDepth            = selDepth;
                rm.scoreLowerbound = rm.scoreUpperbound = false;

                if (value >= beta)
                {
                    rm.scoreLowerbound = true;
                    rm.uciScore        = beta;
                }
                else if (value <= alpha)
                {
                    rm.scoreUpperbound = true;
                    rm.uciScore        = alpha;
                }

                rm.pv.resize(1);

                assert((ss + 1)->pv);

                for (Move* m = (ss + 1)->pv; *m != Move::none(); ++m)
                    rm.pv.push_back(*m);

                // Count best move changes
                if (moveCount > 1 && !pvIdx)
                    ++bestMoveChanges;
            }
            else
                // All other moves but the PV, are set to the lowest value
                rm.score = -VALUE_INFINITE;
        }

        // Check for alternative best move
        int inc = (value == bestValue && ss->ply + 2 >= rootDepth && (int(nodes) & 14) == 0
                   && !is_win(std::abs(value) + 1));

        if (value + inc > bestValue)
        {
            bestValue = value;

            if (value + inc > alpha)
            {
                bestMove = move;

                if (PvNode && !rootNode)  // Update pv even in fail-high case
                    update_pv(ss->pv, move, (ss + 1)->pv);

                if (value >= beta)
                {
                    ss->cutoffCnt += (extension < 2) || PvNode;
                    assert(value >= beta);  // Fail high
                    break;
                }

                // Reduce other moves if we have found at least one score improvement
                if (depth > 2 && depth < 16 && !is_decisive(value))
                    depth -= 2;

                assert(depth > 0);
                alpha = value;  // Update alpha! Always alpha < beta
            }
        }

        // Remember moves for stats update
        if (move != bestMove && moveCount <= SEARCHEDLIST_CAPACITY)
        {
            if (capture)
                capturesSearched.push_back(move);
            else
                quietsSearched.push_back(move);
        }
    }

    // Step 21. Check for mate and stalemate
    assert(moveCount || !ss->inCheck || excludedMove || !MoveList<LEGAL>(pos).size());

    if (!moveCount)
        bestValue = excludedMove ? alpha : ss->inCheck ? mated_in(ss->ply) : VALUE_DRAW;

    // Adjust best value for fail high cases
    if (bestValue >= beta && !is_decisive(bestValue) && !is_decisive(alpha))
        bestValue = (bestValue * depth + beta) / (depth + 1);

    if (bestMove)
    {
        update_all_stats(pos, ss, *this, bestMove, prevSq, quietsSearched, capturesSearched, depth,
                         ttData.move, moveCount);
        if (!PvNode)
            ttMoveHistory << (bestMove == ttData.move ? 811 : -848);
    }
    // Bonus for prior quiet countermove that caused the fail low
    else if (!priorCapture && prevSq != SQ_NONE)
    {
        int bonusScale = -215;
        bonusScale += std::min(-(ss - 1)->statScore / 103, 337);
        bonusScale += std::min(64 * depth, 552);
        bonusScale += 177 * ((ss - 1)->moveCount > 8);
        bonusScale += 141 * (!ss->inCheck && bestValue <= ss->staticEval - 94);
        bonusScale += 141 * (!(ss - 1)->inCheck && bestValue <= -(ss - 1)->staticEval - 76);

        bonusScale = std::max(bonusScale, 0);

        const int scaledBonus = std::min(155 * depth - 88, 1416) * bonusScale;

        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq,
                                      scaledBonus * 397 / 32768);

        mainHistory[~us][((ss - 1)->currentMove).from_to()] << scaledBonus * 224 / 32768;

        if (type_of(pos.piece_on(prevSq)) != PAWN && ((ss - 1)->currentMove).type_of() != PROMOTION)
            pawnHistory[pawn_history_index(pos)][pos.piece_on(prevSq)][prevSq]
              << scaledBonus * 1127 / 32768;
    }
    // Bonus for prior capture countermove that caused the fail low
    else if (priorCapture && prevSq != SQ_NONE)
    {
        Piece capturedPiece = pos.captured_piece();
        assert(capturedPiece != NO_PIECE);
        captureHistory[pos.piece_on(prevSq)][prevSq][type_of(capturedPiece)] << 1042;
    }

    if (PvNode)
        bestValue = std::min(bestValue, maxValue);

    if (bestValue <= alpha)
        ss->ttPv = ss->ttPv || (ss - 1)->ttPv;

    // Write gathered information in transposition table
    if (!excludedMove && !(rootNode && pvIdx))
    {
        Bound bound = bestValue >= beta    ? BOUND_LOWER
                    : PvNode && bestMove ? BOUND_EXACT
                                         : BOUND_UPPER;
        Depth ttDepth = moveCount != 0 ? depth : std::min(MAX_PLY - 1, depth + 6);
        
#ifdef USE_NUMA_TT
        uint8_t gen = generation8.load(std::memory_order_relaxed);
#else
        uint8_t gen = tt.generation();
#endif
        ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv, bound,
                       ttDepth, bestMove, unadjustedStaticEval, gen);
    }

    // Adjust correction history
    if (!ss->inCheck && !(bestMove && pos.capture(bestMove))
        && ((bestValue < ss->staticEval && bestValue < beta)
            || (bestValue > ss->staticEval && bestMove)))
    {
        auto bonus = std::clamp(int(bestValue - ss->staticEval) * depth / 8,
                                -CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
        update_correction_history(pos, ss, *this, bonus);
    }

    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

    return bestValue;
}

// Quiescence search function
template<NodeType NT, typename SEEPolicy>
Value Worker::qsearch(Position& pos, Stack* ss, Value alpha, Value beta) {

    static_assert(NT != Root);
    constexpr bool PvNode = NT == PV;

    assert(alpha >= -VALUE_INFINITE && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));

    if (alpha < VALUE_DRAW && pos.upcoming_repetition(ss->ply))
    {
        alpha = value_draw(nodes);
        if (alpha >= beta)
            return alpha;
    }

    Move      pv[MAX_PLY + 1];
    StateInfo st;

    Key   posKey;
    Move  move, bestMove;
    Value bestValue, value, futilityBase;
    bool  pvHit, givesCheck, capture;
    int   moveCount;

    if (PvNode)
    {
        (ss + 1)->pv = pv;
        ss->pv[0]    = Move::none();
    }

    bestMove    = Move::none();
    ss->inCheck = pos.checkers();
    moveCount   = 0;

    if (PvNode && selDepth < ss->ply + 1)
        selDepth = ss->ply + 1;

    if (pos.is_draw(ss->ply) || ss->ply >= MAX_PLY)
        return (ss->ply >= MAX_PLY && !ss->inCheck) ? evaluate(pos) : VALUE_DRAW;

    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    // TT lookup with NUMA support
    posKey = pos.key();
    TTData ttData(Move::none(), VALUE_NONE, VALUE_NONE, DEPTH_NONE, BOUND_NONE, false);
    TTWriter ttWriter(nullptr);
    bool ttHit;

#ifdef USE_NUMA_TT
    if (l1_tt && l2_tt)
    {
        auto l1res = l1_tt->probe(posKey);
        ttHit = l1res.hit;
        ttData = l1res.data;
        ttWriter = l1res.writer;
        
        if (!ttHit)
        {
            auto l2res = l2_tt->probe(posKey);
            if (l2res.hit)
            {
                ttHit = true;
                ttData = l2res.data;
                l1res.writer.write(posKey, ttData.value, ttData.is_pv, ttData.bound,
                                   ttData.depth, ttData.move, ttData.eval, generation8);
            }
        }
    }
    else
    {
        ttHit = false;
    }
#else
    auto res = tt.probe(posKey);
    ttHit = res.hit;
    ttData = res.data;
    ttWriter = res.writer;
#endif

    ss->ttHit    = ttHit;
    ttData.move  = ttHit ? ttData.move : Move::none();
    ttData.value = ttHit ? value_from_tt(ttData.value, ss->ply, pos.rule50_count()) : VALUE_NONE;
    pvHit        = ttHit && ttData.is_pv;

    if (!PvNode && ttData.depth >= DEPTH_QS
        && is_valid(ttData.value)
        && (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER)))
        return ttData.value;

    Value unadjustedStaticEval = VALUE_NONE;
    if (ss->inCheck)
        bestValue = futilityBase = -VALUE_INFINITE;
    else
    {
        const auto correctionValue = correction_value(*this, pos, ss);

        if (ss->ttHit)
        {
            unadjustedStaticEval = ttData.eval;
            if (!is_valid(unadjustedStaticEval))
                unadjustedStaticEval = evaluate(pos);
            ss->staticEval = bestValue =
              to_corrected_static_eval(unadjustedStaticEval, correctionValue);

            if (is_valid(ttData.value) && !is_decisive(ttData.value)
                && (ttData.bound & (ttData.value > bestValue ? BOUND_LOWER : BOUND_UPPER)))
                bestValue = ttData.value;
        }
        else
        {
            unadjustedStaticEval = evaluate(pos);
            ss->staticEval = bestValue =
              to_corrected_static_eval(unadjustedStaticEval, correctionValue);
        }

        // Stand pat
        if (bestValue >= beta)
        {
            if (!is_decisive(bestValue))
                bestValue = (bestValue + beta) / 2;
            if (!ss->ttHit)
            {
#ifdef USE_NUMA_TT
                uint8_t gen = generation8.load();
#else
                uint8_t gen = tt.generation();
#endif
                ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), false, BOUND_LOWER,
                               DEPTH_UNSEARCHED, Move::none(), unadjustedStaticEval, gen);
            }
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

    // Create minimal context for SEE policy
    NodeContext ctx(pos, ss, alpha, beta, 0, bestValue, false, PvNode, false, 
                    false, 0, 0, false, 0, ttData, 0);

    // Loop through moves
    while ((move = mp.next_move()) != Move::none())
    {
        assert(move.is_ok());

        if (!pos.legal(move))
            continue;

        givesCheck = pos.gives_check(move);
        capture    = pos.capture_stage(move);

        moveCount++;

        // Pruning
        if (!is_loss(bestValue))
        {
            // Futility pruning and moveCount pruning
            if (!givesCheck && move.to_sq() != prevSq && !is_loss(futilityBase)
                && move.type_of() != PROMOTION)
            {
                if (moveCount > 2)
                    continue;

                Value futilityValue = futilityBase + PieceValue[pos.piece_on(move.to_sq())];

                if (futilityValue <= alpha)
                {
                    bestValue = std::max(bestValue, futilityValue);
                    continue;
                }

                if (!pos.see_ge(move, alpha - futilityBase))
                {
                    bestValue = std::min(alpha, futilityBase);
                    continue;
                }
            }

            // Continuation history based pruning
            if (!capture
                && (*contHist[0])[pos.moved_piece(move)][move.to_sq()]
                       + pawnHistory[pawn_history_index(pos)][pos.moved_piece(move)][move.to_sq()]
                     <= 5868)
                continue;

            // Do not search moves with bad enough SEE values
            if (!pos.see_ge(move, -74))
                continue;
        }

        // Make and search the move
        do_move(pos, move, st, givesCheck, ss);

        value = -qsearch<NT, SEEPolicy>(pos, ss + 1, -beta, -alpha);
        
        undo_move(pos, move);

        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

        // Check for a new best move
        if (value > bestValue)
        {
            bestValue = value;

            if (value > alpha)
            {
                bestMove = move;

                if (PvNode)  // Update pv even in fail-high case
                    update_pv(ss->pv, move, (ss + 1)->pv);

                if (value < beta)  // Update alpha here!
                    alpha = value;
                else
                    break;  // Fail high
            }
        }
    }

    // Check for mate
    if (ss->inCheck && bestValue == -VALUE_INFINITE)
    {
        assert(!MoveList<LEGAL>(pos).size());
        return mated_in(ss->ply);
    }

    if (!is_decisive(bestValue) && bestValue > beta)
        bestValue = (bestValue + beta) / 2;

    // Detect stalemate
    Color us = pos.side_to_move();
    if (!ss->inCheck && !moveCount && !pos.non_pawn_material(us)
        && type_of(pos.captured_piece()) >= ROOK)
    {
        if (!((us == WHITE ? shift<NORTH>(pos.pieces(us, PAWN))
                           : shift<SOUTH>(pos.pieces(us, PAWN)))
              & ~pos.pieces()))
        {
            pos.state()->checkersBB = Rank1BB;
            if (!MoveList<LEGAL>(pos).size())
                bestValue = VALUE_DRAW;
            pos.state()->checkersBB = 0;
        }
    }

    // Save gathered info in transposition table
#ifdef USE_NUMA_TT
    uint8_t gen = generation8.load();
#else
    uint8_t gen = tt.generation();
#endif
    ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), pvHit,
                   bestValue >= beta ? BOUND_LOWER : BOUND_UPPER, DEPTH_QS, bestMove,
                   unadjustedStaticEval, gen);

    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

    return bestValue;
}

Depth Worker::reduction(bool i, Depth d, int mn, int delta) const {
    int reductionScale = reductions[d] * reductions[mn];
    return reductionScale - delta * 731 / rootDelta + !i * reductionScale * 216 / 512 + 1089;
}

TimePoint Worker::elapsed() const {
    return main_manager()->tm.elapsed([this]() { return threads.nodes_searched(); });
}

TimePoint Worker::elapsed_time() const { return main_manager()->tm.elapsed_time(); }

Value Worker::evaluate(const Position& pos) {
    return Eval::evaluate(networks[numaAccessToken], pos, accumulatorStack, refreshTable,
                          optimism[pos.side_to_move()]);
}

namespace {

// value_to_tt() adjusts a mate or TB score
Value value_to_tt(Value v, int ply) {
    return is_win(v) ? v + ply : is_loss(v) ? v - ply : v;
}

// value_from_tt() adjusts a mate or TB score from the transposition table
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

// update_pv() adds current move and appends child pv[]
void update_pv(Move* pv, Move move, const Move* childPv) {

    for (*pv++ = move; childPv && *childPv != Move::none();)
        *pv++ = *childPv++;
    *pv = Move::none();
}

// update_all_stats() updates killers, history, countermove and countermove
// history when a new best move is found
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

    if (!pos.capture_stage(bestMove))
    {
        update_quiet_histories(pos, ss, workerThread, bestMove, bonus * 1054 / 1024);

        // Decrease stats for all non-best quiet moves
        for (Move move : quietsSearched)
            update_quiet_histories(pos, ss, workerThread, move, -malus * 1388 / 1024);
    }
    else
    {
        // Increase stats for the best move in case it was a capture move
        capturedPiece = type_of(pos.piece_on(bestMove.to_sq()));
        captureHistory[movedPiece][bestMove.to_sq()][capturedPiece] << bonus * 1235 / 1024;
    }

    // Extra penalty for a quiet early move that was not a TT move in
    // previous ply when it gets refuted.
    if (prevSq != SQ_NONE && ((ss - 1)->moveCount == 1 + (ss - 1)->ttHit) && !pos.captured_piece())
        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -malus * 595 / 1024);

    // Decrease stats for all non-best capture moves
    for (Move move : capturesSearched)
    {
        movedPiece    = pos.moved_piece(move);
        capturedPiece = type_of(pos.piece_on(move.to_sq()));
        captureHistory[movedPiece][move.to_sq()][capturedPiece] << -malus * 1354 / 1024;
    }
}

// update_continuation_histories() updates histories of the move pairs formed
// by moves at ply -1, -2, -3, -4, and -6 with current move
void update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus) {
    static constexpr std::array<ConthistBonus, 6> conthist_bonuses = {
      {{1, 1108}, {2, 652}, {3, 273}, {4, 572}, {5, 126}, {6, 449}}};

    for (const auto [i, weight] : conthist_bonuses)
    {
        // Only update the first 2 continuation histories if we are in check
        if (ss->inCheck && i > 2)
            break;
        if (((ss - i)->currentMove).is_ok())
            (*(ss - i)->continuationHistory)[pc][to] << (bonus * weight / 1024) + 80 * (i < 2);
    }
}

// update_quiet_histories() updates move sorting heuristics
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

}  // namespace

// Move class methods for Worker
void Worker::do_move(Position& pos, const Move move, StateInfo& st, Stack* const ss) {
    do_move(pos, move, st, pos.gives_check(move), ss);
}

void Worker::do_move(
  Position& pos, const Move move, StateInfo& st, const bool givesCheck, Stack* const ss) {
    bool       capture = pos.capture_stage(move);
    DirtyPiece dp      = pos.do_move(move, st, givesCheck,
#ifdef USE_NUMA_TT
                                      nullptr // No TT in NUMA mode for do_move
#else
                                      &tt
#endif
                                      );
    nodes.fetch_add(1, std::memory_order_relaxed);
    accumulatorStack.push(dp);
    if (ss != nullptr)
    {
        ss->currentMove         = move;
        ss->continuationHistory = &continuationHistory[ss->inCheck][capture][dp.pc][move.to_sq()];
        ss->continuationCorrectionHistory = &continuationCorrectionHistory[dp.pc][move.to_sq()];
    }
}

void Worker::do_null_move(Position& pos, StateInfo& st) {
    pos.do_null_move(st,
#ifdef USE_NUMA_TT
                     nullptr  // No TT in NUMA mode for do_null_move
#else
                     tt
#endif
                     );
}

void Worker::undo_move(Position& pos, const Move move) {
    pos.undo_move(move);
    accumulatorStack.pop();
}

void Worker::undo_null_move(Position& pos) { pos.undo_null_move(); }

// Reset histories, usually before a new game
void Worker::clear() {
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

// Skill struct methods
Move Skill::pick_best(const RootMoves& rootMoves, size_t multiPV) {
    static PRNG rng(now());  // PRNG sequence should be non-deterministic

    // RootMoves are already sorted by score in descending order
    Value  topScore = rootMoves[0].score;
    int    delta    = std::min(topScore - rootMoves[multiPV - 1].score, int(PawnValue));
    int    maxScore = -VALUE_INFINITE;
    double weakness = 120 - 2 * level;

    // Choose best move. For each move score we add two terms
    for (size_t i = 0; i < multiPV; ++i)
    {
        // This is our magic formula
        int push = int(weakness * int(topScore - rootMoves[i].score)
                       + delta * (rng.rand<unsigned>() % int(weakness)))
                 / 128;

        if (rootMoves[i].score + push >= maxScore)
        {
            maxScore = rootMoves[i].score + push;
            best     = rootMoves[i].pv[0];
        }
    }

    return best;
}

// SearchManager methods
void SearchManager::check_time(Search::Worker& worker) {
    if (--callsCnt > 0)
        return;

    // When using nodes, ensure checking rate is not lower than 0.1% of nodes
    callsCnt = worker.limits.nodes ? std::min(512, int(worker.limits.nodes / 1024)) : 512;

    static TimePoint lastInfoTime = now();

    TimePoint elapsed = tm.elapsed([&worker]() { return worker.threads.nodes_searched(); });
    TimePoint tick    = worker.limits.startTime + elapsed;

    if (tick - lastInfoTime >= 1000)
    {
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

#ifdef USE_NUMA_TT
void SearchManager::pv(Search::Worker&   worker,
                       const ThreadPool& threads,
                       Depth             depth) {
#else
void SearchManager::pv(Search::Worker&           worker,
                       const ThreadPool&         threads,
                       const TranspositionTable& tt,
                       Depth                     depth) {
#endif

    const auto nodes     = threads.nodes_searched();
    auto&      rootMoves = worker.rootMoves;
    auto&      pos       = worker.rootPos;
    size_t     pvIdx     = worker.pvIdx;
    size_t     multiPV   = std::min(size_t(worker.options["MultiPV"]), rootMoves.size());
    uint64_t   tbHits    = threads.tb_hits() + (worker.tbConfig.rootInTB ? rootMoves.size() : 0);

    for (size_t i = 0; i < multiPV; ++i)
    {
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

        // Potentially correct and extend the PV
        if (is_decisive(v) && std::abs(v) < VALUE_MATE_IN_MAX_PLY
            && ((!rootMoves[i].scoreLowerbound && !rootMoves[i].scoreUpperbound) || isExact))
            syzygy_extend_pv(worker.options, worker.limits, pos, rootMoves[i], v);

        std::string pv_str;
        for (Move m : rootMoves[i].pv)
            pv_str += UCIEngine::move(m, pos.is_chess960()) + " ";

        // Remove last whitespace
        if (!pv_str.empty())
            pv_str.pop_back();

        auto wdl   = worker.options["UCI_ShowWDL"] ? UCIEngine::wdl(v, pos) : "";
        auto bound = rootMoves[i].scoreLowerbound ? "lowerbound"
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
        info.pv        = pv_str;
        
#ifdef USE_NUMA_TT
        // Calculate hashfull from L1 and L2 tables
        int hashfull = 0;
        if (worker.l1_tt) hashfull = std::max(hashfull, worker.l1_tt->hashfull());
        if (worker.l2_tt) hashfull = std::max(hashfull, worker.l2_tt->hashfull());
        info.hashfull = hashfull;
#else
        info.hashfull = tt.hashfull();
#endif

        updates.onUpdateFull(info);
    }
}

// RootMove methods
#ifdef USE_NUMA_TT
bool RootMove::extract_ponder_from_tt(const TranspositionTable& tt, Position& pos) {
#else
bool RootMove::extract_ponder_from_tt(const TranspositionTable& tt, Position& pos) {
#endif

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
    if (ttHit)
    {
        if (MoveList<LEGAL>(pos).contains(ttData.move))
            pv.push_back(ttData.move);
    }

    pos.undo_move(pv[0]);
    return pv.size() > 1;
}

// Explicit template instantiations
template Value Worker::search<NonPV,
                              DefaultRazoringPolicy,
                              DefaultFutilityPolicy,
                              DefaultNMPolicy,
                              DefaultLMRPolicy,
                              DefaultSEEPolicy,
                              DefaultExtensionPolicy>(
  Position&, Stack*, Value, Value, Depth, bool);

template Value Worker::search<PV,
                              DefaultRazoringPolicy,
                              DefaultFutilityPolicy,
                              DefaultNMPolicy,
                              DefaultLMRPolicy,
                              DefaultSEEPolicy,
                              DefaultExtensionPolicy>(
  Position&, Stack*, Value, Value, Depth, bool);

template Value Worker::search<Root,
                              DefaultRazoringPolicy,
                              DefaultFutilityPolicy,
                              DefaultNMPolicy,
                              DefaultLMRPolicy,
                              DefaultSEEPolicy,
                              DefaultExtensionPolicy>(
  Position&, Stack*, Value, Value, Depth, bool);

template Value Worker::qsearch<NonPV, DefaultSEEPolicy>(Position&, Stack*, Value, Value);
template Value Worker::qsearch<PV, DefaultSEEPolicy>(Position&, Stack*, Value, Value);

}  // namespace Stockfish
