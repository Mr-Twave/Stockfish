
# searchpolicies.h

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

#ifndef SEARCHPOLICIES_H_INCLUDED
#define SEARCHPOLICIES_H_INCLUDED

#include "types.h"

namespace Stockfish {

class Position;

namespace Search {

struct Stack;

// This header contains the policy-based design for the search function.
// Each heuristic is encapsulated in its own class as a static function.
// This allows for compile-time polymorphism with zero runtime overhead,
// improving modularity and enabling easier experimentation.

// A single, non-owning struct to pass all necessary read-only context
// to the policy functions. This is a zero-cost abstraction.
struct NodeContext {
    const Position& pos;
    const Stack* ss;
    const Value     alpha;
    const Value     beta;
    const Depth     depth;
    const Value     eval;
    const bool      cutNode;
    const bool      pvNode;
    const bool      improving;
    const bool      opponentWorsening;
    const Value     correctionValue;
    const int       nmpMinPly;
    const bool      excludedMove;
    
    // Constructor for easy initialization
    NodeContext(const Position& p, const Stack* s, Value a, Value b, Depth d, 
                Value e, bool cn, bool pv, bool imp, bool ow, Value cv, int nmp, bool em) :
        pos(p), ss(s), alpha(a), beta(b), depth(d), eval(e), cutNode(cn), 
        pvNode(pv), improving(imp), opponentWorsening(ow), correctionValue(cv), 
        nmpMinPly(nmp), excludedMove(em) {}
};

// Policy for Razoring
struct DefaultRazoringPolicy {
    static constexpr bool Enabled = true;
    static bool should_prune(const NodeContext& ctx) {
        return !ctx.pvNode && ctx.eval < ctx.alpha - 495 - 290 * ctx.depth * ctx.depth;
    }
};

// Policy for Futility Pruning
struct DefaultFutilityPolicy {
    static constexpr bool Enabled = true;
    static Value get_margin(const NodeContext& ctx) {
        Value futilityMult = 90 - 20 * (ctx.cutNode && !ctx.ss->ttHit);
        return futilityMult * ctx.depth
             - ctx.improving * futilityMult * 2
             - ctx.opponentWorsening * futilityMult / 3
             + (ctx.ss - 1)->statScore / 356
             + std::abs(ctx.correctionValue) / 171290;
    }
    
    static bool should_prune(const NodeContext& ctx, Value margin) {
        return !ctx.ss->ttPv && ctx.depth < 14 && ctx.eval - margin >= ctx.beta 
               && ctx.eval >= ctx.beta && !is_loss(ctx.beta) && !is_win(ctx.eval);
    }
};

// Policy for Null Move Pruning (NMP)
struct DefaultNMPolicy {
    static constexpr bool Enabled = true;
    static bool should_prune(const NodeContext& ctx) {
        return ctx.cutNode && ctx.ss->staticEval >= ctx.beta - 19 * ctx.depth + 403
            && !ctx.excludedMove
            && ctx.pos.non_pawn_material(ctx.pos.side_to_move())
            && ctx.ss->ply >= ctx.nmpMinPly && !is_loss(ctx.beta);
    }
    
    static Depth get_reduction(Depth depth) {
        return 7 + depth / 3;
    }
};

// Policy for Late Move Reductions (LMR)
struct DefaultLMRPolicy {
    static constexpr bool Enabled = true;
    
    static bool should_apply(const NodeContext& ctx, Depth newDepth, int moveCount) {
        return newDepth >= 2 && moveCount > 1;
    }
    
    static Depth get_reduction(const NodeContext& ctx, bool improving, Depth depth, 
                               int moveCount, int delta, bool pvNode, bool cutNode) {
        // This would contain the complex LMR reduction logic from the original code
        // For now, providing a simplified version
        int r = 650; // Base reduction offset
        r -= moveCount * 69;
        r -= std::abs(ctx.correctionValue) / 27160;
        
        if (cutNode)
            r += 3000 + 1024 * !ctx.ss->ttHit;
            
        if (pvNode)
            r -= 2510;
            
        return r;
    }
};

// Policy for SEE (Static Exchange Evaluation) pruning
struct DefaultSEEPolicy {
    static constexpr bool Enabled = true;
    
    static bool should_prune_capture(const NodeContext& ctx, Move move, int captHist) {
        int margin = std::clamp(158 * ctx.depth + captHist / 31, 0, 283 * ctx.depth);
        return !ctx.pos.see_ge(move, -margin);
    }
    
    static bool should_prune_quiet(const NodeContext& ctx, Move move) {
        return !ctx.pos.see_ge(move, -26 * ctx.depth * ctx.depth);
    }
};

// Policy for Extensions (Singular, etc.)
struct DefaultExtensionPolicy {
    static constexpr bool Enabled = true;
    
    static bool should_extend_singular(const NodeContext& ctx, Move ttMove, Value ttValue) {
        return !ctx.excludedMove && move == ttMove && ctx.depth >= 6 - (ctx.completedDepth > 26) + ctx.ss->ttPv
               && is_valid(ttValue) && !is_decisive(ttValue) && (ttData.bound & BOUND_LOWER)
               && ttData.depth >= ctx.depth - 3;
    }
    
    static Value get_singular_beta(const NodeContext& ctx, Value ttValue) {
        return ttValue - (56 + 79 * (ctx.ss->ttPv && !ctx.pvNode)) * ctx.depth / 58;
    }
};

} // namespace Search
} // namespace Stockfish

#endif // SEARCHPOLICIES_H_INCLUDED
