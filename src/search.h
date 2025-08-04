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

#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "history.h"
#include "misc.h"
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"
#include "numa.h"
#include "position.h"
#include "score.h"
#include "searchpolicies.h" // New header for modular search heuristics
#include "syzygy/tbprobe.h"
#include "timeman.h"
#include "types.h"

namespace Stockfish {

// Different node types, used as a template parameter
enum NodeType {
    NonPV,
    PV,
    Root
};

class TranspositionTable;
class ThreadPool;
class OptionsMap;

namespace Search {

// Stack struct keeps track of the information we need to remember from nodes
// shallower and deeper in the tree during the search. Each search thread has
// its own array of Stack objects, indexed by the current ply.
struct Stack {
    Move* pv;
    PieceToHistory* continuationHistory;
    CorrectionHistory<PieceTo>* continuationCorrectionHistory;
    int                         ply;
    Move                        currentMove;
    Move                        excludedMove;
    Value                       staticEval;
    int                         statScore;
    int                         moveCount;
    bool                        inCheck;
    bool                        ttPv;
    bool                        ttHit;
    int                         cutoffCnt;
    int                         reduction;
    int                         quietMoveStreak;
};


// RootMove struct is used for moves at the root of the tree.
struct RootMove {

    explicit RootMove(Move m) :
        pv(1, m) {}
    bool extract_ponder_from_tt(const TranspositionTable& tt, Position& pos);
    bool operator==(const Move& m) const { return pv[0] == m; }
    // Sort in descending order
    bool operator<(const RootMove& m) const {
        return m.score != score ? m.score < score : m.previousScore < previousScore;
    }

    uint64_t            effort           = 0;
    Value               score            = -VALUE_INFINITE;
    Value               previousScore    = -VALUE_INFINITE;
    Value               averageScore     = -VALUE_INFINITE;
    Value               meanSquaredScore = -VALUE_INFINITE * VALUE_INFINITE;
    Value               uciScore         = -VALUE_INFINITE;
    bool                scoreLowerbound  = false;
    bool                scoreUpperbound  = false;
    int                 selDepth         = 0;
    int                 tbRank           = 0;
    Value               tbScore;
    std::vector<Move> pv;
};

using RootMoves = std::vector<RootMove>;


// LimitsType struct stores information sent by the caller about the analysis required.
struct LimitsType {
    // Init explicitly due to broken value-initialization of non POD in MSVC
    LimitsType() {
        time[WHITE] = time[BLACK] = inc[WHITE] = inc[BLACK] = npmsec = movetime = TimePoint(0);
        movestogo = depth = mate = perft = infinite = 0;
        nodes                                       = 0;
        ponderMode                                  = false;
    }

    bool use_time_management() const { return time[WHITE] || time[BLACK]; }

    std::vector<std::string> searchmoves;
    TimePoint                time[COLOR_NB], inc[COLOR_NB], npmsec, movetime, startTime;
    int                      movestogo, depth, mate, perft, infinite;
    uint64_t                 nodes;
    bool                     ponderMode;
};


// The SharedState struct is used to easily forward shared data to the Search::Worker class.
// Its layout is conditionally compiled to support both UMA and NUMA architectures.
struct SharedState {
    SharedState(const OptionsMap&                                 optionsMap,
                ThreadPool&                                     threadPool,
#ifndef USE_NUMA_TT
                TranspositionTable&                             transpositionTable,
#endif
                const LazyNumaReplicated<Eval::NNUE::Networks>& nets) :
        options(optionsMap),
        threads(threadPool),
#ifdef USE_NUMA_TT
        networks(nets),
        generation8(0)
#else
        tt(transpositionTable),
        networks(nets)
#endif
    {}

    const OptionsMap&                                 options;
    ThreadPool&                                       threads;
    const LazyNumaReplicated<Eval::NNUE::Networks>& networks;

#ifdef USE_NUMA_TT
    // The new TT hierarchy for NUMA systems
    std::vector<std::unique_ptr<TranspositionTable>> l1_tts;
    std::unique_ptr<TranspositionTable>              l2_tt;
    // The generation counter is now shared at this level for all TTs
    std::atomic<uint8_t>                             generation8;
#else
    // The original single TT for UMA systems
    TranspositionTable& tt;
#endif
};

class Worker;

// Null Object Pattern, implement a common interface for the SearchManagers.
class ISearchManager {
   public:
    virtual ~ISearchManager() {}
    virtual void check_time(Search::Worker&) = 0;
};

struct InfoShort {
    int   depth;
    Score score;
};

struct InfoFull: InfoShort {
    int              selDepth;
    size_t           multiPV;
    std::string_view wdl;
    std::string_view bound;
    size_t           timeMs;
    size_t           nodes;
    size_t           nps;
    size_t           tbHits;
    std::string_view pv;
    int              hashfull;
};

struct InfoIteration {
    int              depth;
    std::string_view currmove;
    size_t           currmovenumber;
};

// Skill structure is used to implement strength limit.
struct Skill {
    // Lowest and highest Elo ratings used in the skill level calculation
    constexpr static int LowestElo  = 1320;
    constexpr static int HighestElo = 3190;

    Skill(int skill_level, int uci_elo) {
        if (uci_elo)
        {
            double e = double(uci_elo - LowestElo) / (HighestElo - LowestElo);
            level = std::clamp((((37.2473 * e - 40.8525) * e + 22.2943) * e - 0.311438), 0.0, 19.0);
        }
        else
            level = double(skill_level);
    }
    bool enabled() const { return level < 20.0; }
    bool time_to_pick(Depth depth) const { return depth == 1 + int(level); }
    Move pick_best(const RootMoves&, size_t multiPV);

    double level;
    Move   best = Move::none();
};

// SearchManager manages the search from the main thread.
class SearchManager: public ISearchManager {
   public:
    using UpdateShort    = std::function<void(const InfoShort&)>;
    using UpdateFull     = std::function<void(const InfoFull&)>;
    using UpdateIter     = std::function<void(const InfoIteration&)>;
    using UpdateBestmove = std::function<void(std::string_view, std::string_view)>;

    struct UpdateContext {
        UpdateShort    onUpdateNoMoves;
        UpdateFull     onUpdateFull;
        UpdateIter     onIter;
        UpdateBestmove onBestmove;
    };

    SearchManager(const UpdateContext& updateContext) :
        updates(updateContext) {}

    void check_time(Search::Worker& worker) override;

    void pv(Search::Worker&           worker,
            const ThreadPool&         threads,
#ifndef USE_NUMA_TT
            const TranspositionTable& tt,
#endif
            Depth                     depth);

    Stockfish::TimeManagement tm;
    double                    originalTimeAdjust;
    int                       callsCnt;
    std::atomic_bool          ponder;
    std::array<Value, 4> iterValue;
    double               previousTimeReduction;
    Value                bestPreviousScore;
    Value                bestPreviousAverageScore;
    bool                 stopOnPonderhit;
    size_t id;
    const UpdateContext& updates;
};

class NullSearchManager: public ISearchManager {
   public:
    void check_time(Search::Worker&) override {}
};


// Search::Worker is the class that does the actual search.
class Worker {
   public:
    Worker(SharedState&, std::unique_ptr<ISearchManager>, size_t, NumaReplicatedAccessToken);

    void clear();
    void start_searching();
    bool is_mainthread() const { return threadIdx == 0; }
    void ensure_network_replicated();

    // Public move ordering and history tables
    ButterflyHistory mainHistory;
    LowPlyHistory    lowPlyHistory;
    CapturePieceToHistory captureHistory;
    ContinuationHistory   continuationHistory[2][2];
    PawnHistory           pawnHistory;
    CorrectionHistory<Pawn>         pawnCorrectionHistory;
    CorrectionHistory<Minor>        minorPieceCorrectionHistory;
    CorrectionHistory<NonPawn>      nonPawnCorrectionHistory;
    CorrectionHistory<Continuation> continuationCorrectionHistory;
    TTMoveHistory ttMoveHistory;

   private:
    void iterative_deepening();
    void do_move(Position& pos, const Move move, StateInfo& st, Stack* const ss);
    void do_move(Position& pos, const Move move, StateInfo& st, const bool givesCheck, Stack* const ss);
    void do_null_move(Position& pos, StateInfo& st);
    void undo_move(Position& pos, const Move move);
    void undo_null_move(Position& pos);

    // The main search function, now templated on node type and search policies
    template<NodeType NT,
             typename RazoringPolicy,
             typename FutilityPolicy,
             typename NMPolicy,
             typename LMRPolicy,
             typename SEEPolicy,
             typename ExtensionPolicy>
    Value search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode);

    // Quiescence search function, now templated on its policies
    template<NodeType NT,
             typename SEEPolicy>
    Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta);

    Depth reduction(bool i, Depth d, int mn, int delta) const;

    SearchManager* main_manager() const {
        assert(threadIdx == 0);
        return static_cast<SearchManager*>(manager.get());
    }

    TimePoint elapsed() const;
    TimePoint elapsed_time() const;
    Value evaluate(const Position&);

    // Member Variables
    LimitsType limits;
    size_t                pvIdx, pvLast;
    std::atomic<uint64_t> nodes, tbHits, bestMoveChanges;
    int                   selDepth, nmpMinPly;
    Value optimism[COLOR_NB];
    Position  rootPos;
    StateInfo rootState;
    RootMoves rootMoves;
    Depth     rootDepth, completedDepth;
    Value     rootDelta;
    size_t                    threadIdx;
    NumaReplicatedAccessToken numaAccessToken;
    std::array<int, MAX_MOVES> reductions;
    std::unique_ptr<ISearchManager> manager;
    Tablebases::Config tbConfig;
    const OptionsMap&                                 options;
    ThreadPool&                                       threads;
    const LazyNumaReplicated<Eval::NNUE::Networks>& networks;

#ifdef USE_NUMA_TT
    // Pointers to this worker's assigned TTs for NUMA builds
    TranspositionTable* l1_tt;
    TranspositionTable* l2_tt;
    // Reference to the shared generation counter
    std::atomic<uint8_t>& generation8;
#else
    // Original single TT reference for standard UMA builds
    TranspositionTable& tt;
#endif

    Eval::NNUE::AccumulatorStack  accumulatorStack;
    Eval::NNUE::AccumulatorCaches refreshTable;

    friend class Stockfish::ThreadPool;
    friend class SearchManager;
};

struct ConthistBonus {
    int index;
    int weight;
};

}  // namespace Search
}  // namespace Stockfish

#endif  // #ifndef SEARCH_H_INCLUDED
