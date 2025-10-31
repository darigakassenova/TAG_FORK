package players.groupG.heuristics;

import core.AbstractGameState;
import core.interfaces.IStateHeuristic;
import core.components.Counter;
import games.sushigo.SGGameState;
import games.sushigo.cards.SGCard;

import java.util.Map;

public class SushiGoHeuristic implements IStateHeuristic {

    @Override
    public double evaluateState(AbstractGameState gs, int playerId) {
        if (!(gs instanceof SGGameState)) return 0.0;
        SGGameState state = (SGGameState) gs;

        double score = state.getGameScore(playerId);

        int tempura   = state.getPlayedCardTypes(SGCard.SGCardType.Tempura, playerId).getValue();
        int sashimi   = state.getPlayedCardTypes(SGCard.SGCardType.Sashimi, playerId).getValue();
        int dumpling  = state.getPlayedCardTypes(SGCard.SGCardType.Dumpling, playerId).getValue();
        int maki      = state.getPlayedCardTypes(SGCard.SGCardType.Maki, playerId).getValue();
        int puddings  = state.getPlayedCardTypesAllGame()[playerId].get(SGCard.SGCardType.Pudding).getValue();
        int wasabi    = state.getPlayedCardTypes(SGCard.SGCardType.Wasabi, playerId).getValue();
        int chopsticks= state.getPlayedCardTypes(SGCard.SGCardType.Chopsticks, playerId).getValue();
        int eggNigiri = state.getPlayedCardTypes(SGCard.SGCardType.EggNigiri, playerId).getValue();
        int salmonNigiri = state.getPlayedCardTypes(SGCard.SGCardType.SalmonNigiri, playerId).getValue();
        int squidNigiri  = state.getPlayedCardTypes(SGCard.SGCardType.SquidNigiri, playerId).getValue();

        // ===== progress in game (0..1)
        int round = gs.getRoundCounter();
        int turn = gs.getTurnCounter();
        double progress = Math.min(1.0, (round * 0.33) + (turn * 0.02));

        // ===== sets & combos (конкретные правила)
        score += (tempura / 2) * 5 + (tempura % 2) * 1.0;
        score += (sashimi / 3) * 10 + (sashimi % 3) * 2.0;

        int[] dumV = {0,1,3,6,10,15};
        score += dumV[Math.min(dumpling, 5)];

        int totalNigiri = eggNigiri + salmonNigiri + squidNigiri;
        double nigiriValue = eggNigiri*1 + salmonNigiri*2 + squidNigiri*3;
        if (wasabi > 0 && totalNigiri > 0) score += Math.min(wasabi, totalNigiri) * 3.0;
        else if (wasabi > 0) score -= 1.5;
        score += nigiriValue;

        double makiWeight = 0.5 + 1.0 * progress;
        score += maki * makiWeight;

        double puddingWeight = (progress > 0.7 ? 2.0 : 0.8);
        score += puddings * puddingWeight;

        score += chopsticks * 0.5;

        if (sashimi % 3 == 2) score += 1.5;
        if (tempura % 2 == 1) score += 1.0;

        int bestMakiOther = bestOther(state, SGCard.SGCardType.Maki, playerId, false);
        if (bestMakiOther >= maki + 3) score -= 1.0;

        int bestPuddingOther = bestOther(state, SGCard.SGCardType.Pudding, playerId, true);
        if (bestPuddingOther - puddings >= 2) score += 1.0;

        int diversity = 0;
        if (tempura > 0) diversity++;
        if (sashimi > 0) diversity++;
        if (dumpling > 0) diversity++;
        if (totalNigiri > 0) diversity++;
        if (puddings > 0) diversity++;
        score += diversity * 0.3;

        return score;
    }

    private int get(Map<SGCard.SGCardType, Counter>[] mapArr, SGCard.SGCardType t, int player) {
        if (mapArr == null || mapArr[player] == null) return 0;
        Counter c = mapArr[player].get(t);
        return c == null ? 0 : c.getValue();
    }

    private int bestOther(SGGameState s, SGCard.SGCardType type, int me, boolean allGame) {
        int best = 0;
        for (int p = 0; p < s.getNPlayers(); p++) {
            if (p == me) continue;
            int v = allGame
                    ? get(s.getPlayedCardTypesAllGame(), type, p)
                    : get(s.getPlayedCardTypes(), type, p);
            if (v > best) best = v;
        }
        return best;
    }

    @Override public double minValue() { return Double.NEGATIVE_INFINITY; }
    @Override public double maxValue() { return Double.POSITIVE_INFINITY; }
    @Override public String toString() { return "SushiGoHeuristic"; }
}
