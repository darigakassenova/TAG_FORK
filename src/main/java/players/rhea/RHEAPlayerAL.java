package players.rhea;

import core.AbstractGameState;
import core.AbstractPlayer;
import core.actions.AbstractAction;
import core.interfaces.IStateHeuristic;
import games.GameType;
import games.sushigo.actions.ChooseCard;
import games.sushigo.SGGameState;
import games.sushigo.cards.SGCard;
import players.IAnyTimePlayer;
import players.PlayerConstants;
import players.mcts.MASTPlayer;
import players.simple.RandomPlayer;
import utilities.ElapsedCpuTimer;
import utilities.Pair;
import utilities.Utils;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class   RHEAPlayer extends AbstractPlayer implements IAnyTimePlayer {
    private static final AbstractPlayer randomPlayer = new RandomPlayer();
    List<Map<Object, Pair<Integer, Double>>> MASTStatistics; // a list of one Map per player. Action -> (visits, totValue)
    protected List<RHEAIndividual> population = new ArrayList<>();
    // Budgets Complications
    protected double timePerIteration = 0, timeTaken = 0, initTime = 0;
    protected int numIters = 0;
    protected int fmCalls = 0;
    protected int copyCalls = 0;
    protected int repairCount, nonRepairCount;
    private MASTPlayer mastPlayer;
    private GameType gameType;  // Track game type for domain-specific optimizations
    private IStateHeuristic enhancedHeuristic;  // Wrapped heuristic with Sushi Go enhancements

    public RHEAPlayer(RHEAParams params) {
        super(params, "RHEAPlayer");
    }

    public RHEAPlayer(RHEAParams params, String name) {
        super(params, name);
    }

    @Override
    public RHEAParams getParameters() {
        return (RHEAParams) parameters;
    }
    @Override
    public void initializePlayer(AbstractGameState state) {
        MASTStatistics = new ArrayList<>();
        for (int i = 0; i < state.getNPlayers(); i++)
            MASTStatistics.add(new HashMap<>());
        population = new ArrayList<>();

        // Track game type for domain-specific optimizations
        this.gameType = state.getGameType();

        // Initialize MAST with domain knowledge for Sushi Go
        if (gameType == GameType.SushiGo) {
            initializeSushiGoMAST(state.getNPlayers());
            // Wrap the heuristic with Sushi Go enhancements
            RHEAParams params = getParameters();
            if (params.heuristic != null) {
                enhancedHeuristic = new SushiGoEnhancedHeuristic(params.heuristic);
            }
        }
    }

    /**
     * Initialize MAST statistics with domain knowledge for Sushi Go
     * High-value cards get better initial scores
     */
    private void initializeSushiGoMAST(int nPlayers) {
        // Very high value cards (sashimi, high nigiri)
        String[] highValueCards = {"Sashimi", "SquidNigiri", "Tempura"};
        // Medium value cards (good with wasabi or for sets)
        String[] mediumValueCards = {"SalmonNigiri", "Wasabi", "EggNigiri", "Dumpling"};
        // Defensive/situational cards (pudding for tiebreak, chopsticks for flexibility)
        String[] situationalCards = {"Pudding", "Chopsticks"};

        for (int p = 0; p < nPlayers; p++) {
            Map<Object, Pair<Integer, Double>> stats = MASTStatistics.get(p);

            // Initialize high-value cards with good prior
            for (String card : highValueCards) {
                stats.put("CardType_" + card, new Pair<>(20, 100.0));  // ~5.0 value per visit
            }

            // Initialize medium-value cards
            for (String card : mediumValueCards) {
                stats.put("CardType_" + card, new Pair<>(20, 60.0));  // ~3.0 value per visit
            }

            // Initialize situational cards (lower priority but available)
            for (String card : situationalCards) {
                stats.put("CardType_" + card, new Pair<>(20, 30.0));  // ~1.5 value per visit
            }
        }
    }

    @Override
    public AbstractAction _getAction(AbstractGameState stateObs, List<AbstractAction> possibleActions) {
        ElapsedCpuTimer timer = new ElapsedCpuTimer();  // New timer for this game tick
        timer.setMaxTimeMillis(parameters.budget);
        numIters = 0;
        fmCalls = 0;
        copyCalls = 0;
        repairCount = 0;
        nonRepairCount = 0;
        RHEAParams params = getParameters();

        if (params.useMAST) {
            if (MASTStatistics == null) {
                MASTStatistics = new ArrayList<>();
                for (int i = 0; i < stateObs.getNPlayers(); i++)
                    MASTStatistics.add(new HashMap<>());
            } else {
                MASTStatistics = MASTStatistics.stream()
                        .map(m -> Utils.decay(m, params.discountFactor))
                        .collect(Collectors.toList());
            }
            mastPlayer = new MASTPlayer(null, 1.0, 0.0, System.currentTimeMillis(), 0.0);
            mastPlayer.setMASTStats(MASTStatistics);
        }
        // Use enhanced heuristic for Sushi Go, otherwise use base heuristic
        IStateHeuristic heuristicToUse = (gameType == GameType.SushiGo && enhancedHeuristic != null)
                ? enhancedHeuristic : params.heuristic;

        // Initialise individuals
        if (params.shiftLeft && !population.isEmpty()) {
            // Update heuristic for existing individuals if needed
            if (heuristicToUse != params.heuristic) {
                for (RHEAIndividual genome : population) {
                    genome.heuristic = heuristicToUse;
                }
            }
            population.forEach(i -> i.value = Double.NEGATIVE_INFINITY);  // so that any we don't have time to shift are ignored when picking an action
            for (RHEAIndividual genome : population) {
                if (!budgetLeft(timer)) break;
                System.arraycopy(genome.actions, 1, genome.actions, 0, genome.actions.length - 1);
                // we shift all actions along, and then rollout with repair
                genome.gameStates[0] = stateObs.copy();
                Pair<Integer, Integer> calls = genome.rollout(getForwardModel(), 0, getPlayerID(), true);
                fmCalls += calls.a;
                copyCalls += calls.b;
            }
        } else {
            population = new ArrayList<>();
            // Smart initialization: mix random and MAST-informed individuals
            int randomCount = Math.max(1, params.populationSize / 2);
            int mastCount = params.populationSize - randomCount;

            // Random initialization for exploration
            for (int i = 0; i < randomCount; ++i) {
                if (!budgetLeft(timer)) break;
                population.add(new RHEAIndividual(params.horizon, params.discountFactor, getForwardModel(), stateObs,
                        getPlayerID(), rnd, heuristicToUse, randomPlayer));
                fmCalls += population.get(population.size() - 1).length;
                copyCalls += population.get(population.size() - 1).length;
            }

            // MAST-informed initialization if enabled (better starting population)
            if (params.useMAST) {
                for (int i = 0; i < mastCount; ++i) {
                    if (!budgetLeft(timer)) break;
                    population.add(new RHEAIndividual(params.horizon, params.discountFactor, getForwardModel(), stateObs,
                            getPlayerID(), rnd, heuristicToUse, mastPlayer));
                    fmCalls += population.get(population.size() - 1).length;
                    copyCalls += population.get(population.size() - 1).length;
                }
            } else {
                // Fill remaining with random if MAST not enabled
                for (int i = 0; i < mastCount; ++i) {
                    if (!budgetLeft(timer)) break;
                    population.add(new RHEAIndividual(params.horizon, params.discountFactor, getForwardModel(), stateObs,
                            getPlayerID(), rnd, heuristicToUse, randomPlayer));
                    fmCalls += population.get(population.size() - 1).length;
                    copyCalls += population.get(population.size() - 1).length;
                }
            }
        }

        population.sort(Comparator.naturalOrder());
        initTime = timer.elapsedMillis();
        // Run evolution
        while (budgetLeft(timer)) {
            runIteration();
        }

        timeTaken = timer.elapsedMillis();
        timePerIteration = numIters == 0 ? 0.0 : (timeTaken - initTime) / numIters;
        // Return first action of best individual
        AbstractAction retValue = population.get(0).actions[0];
        List<AbstractAction> actions = getForwardModel().computeAvailableActions(stateObs, params.actionSpace);
        if (!actions.contains(retValue))
            throw new AssertionError("Action chosen is not legitimate " + numIters + ", " + params.shiftLeft);
        return retValue;
    }

    private boolean budgetLeft(ElapsedCpuTimer timer) {
        RHEAParams params = getParameters();
        if (params.budgetType == PlayerConstants.BUDGET_TIME) {
            long remaining = timer.remainingTimeMillis();
            return remaining > params.breakMS;
        } else if (params.budgetType == PlayerConstants.BUDGET_FM_CALLS) {
            return fmCalls < params.budget;
        } else if (params.budgetType == PlayerConstants.BUDGET_COPY_CALLS) {
            return copyCalls < params.budget && numIters < params.budget;
        } else if (params.budgetType == PlayerConstants.BUDGET_FMANDCOPY_CALLS) {
            return (fmCalls + copyCalls) < params.budget;
        } else if (params.budgetType == PlayerConstants.BUDGET_ITERATIONS) {
            return numIters < params.budget;
        }
        throw new AssertionError("This should be unreachable : " + params.budgetType);
    }

    @Override
    public RHEAPlayer copy() {
        RHEAParams newParams = (RHEAParams) parameters.copy();
        newParams.setRandomSeed(rnd.nextInt());
        RHEAPlayer retValue = new RHEAPlayer(newParams, toString());
        retValue.setForwardModel(getForwardModel());
        return retValue;
    }

    private RHEAIndividual crossover(RHEAIndividual p1, RHEAIndividual p2) {
        switch (getParameters().crossoverType) {
            case NONE: // we just take the first parent
                return new RHEAIndividual(p1);
            case UNIFORM:
                return uniformCrossover(p1, p2);
            case ONE_POINT:
                return onePointCrossover(p1, p2);
            case TWO_POINT:
                return twoPointCrossover(p1, p2);
            default:
                throw new RuntimeException("Unexpected crossover type");
        }
    }

    private RHEAIndividual uniformCrossover(RHEAIndividual p1, RHEAIndividual p2) {
        RHEAIndividual child = new RHEAIndividual(p1);
        copyCalls += child.length;
        int min = Math.min(p1.length, p2.length);
        for (int i = 0; i < min; ++i) {
            if (rnd.nextFloat() >= 0.5f) {
                child.actions[i] = p2.actions[i];
                child.gameStates[i] = p2.gameStates[i]; //.copy();
            }
        }
        return child;
    }

    private RHEAIndividual onePointCrossover(RHEAIndividual p1, RHEAIndividual p2) {
        RHEAIndividual child = new RHEAIndividual(p1);
        copyCalls += child.length;
        int tailLength = Math.min(p1.length, p2.length) / 2;

        for (int i = 0; i < tailLength; ++i) {
            child.actions[child.length - 1 - i] = p2.actions[p2.length - 1 - i];
            child.gameStates[child.length - 1 - i] = p2.gameStates[p2.length - 1 - i]; //.copy();
        }
        return child;
    }

    private RHEAIndividual twoPointCrossover(RHEAIndividual p1, RHEAIndividual p2) {
        RHEAIndividual child = new RHEAIndividual(p1);
        copyCalls += child.length;
        int tailLength = Math.min(p1.length, p2.length) / 3;
        for (int i = 0; i < tailLength; ++i) {
            child.actions[i] = p2.actions[i];
            child.gameStates[i] = p2.gameStates[i]; //.copy();
            child.actions[child.length - 1 - i] = p2.actions[p2.length - 1 - i];
            child.gameStates[child.length - 1 - i] = p2.gameStates[p2.length - 1 - i]; //.copy();
        }
        return child;
    }

    RHEAIndividual[] selectParents() {
        RHEAIndividual[] parents = new RHEAIndividual[2];

        switch (getParameters().selectionType) {
            case TOURNAMENT:
                parents[0] = tournamentSelection();
                parents[1] = tournamentSelection();
                break;
            case RANK:
                parents[0] = rankSelection();
                parents[1] = rankSelection();
                break;
            default:
                throw new RuntimeException("Unexpected selection type");
        }

        return parents;
    }

    RHEAIndividual tournamentSelection() {
        RHEAIndividual best = null;
        for (int i = 0; i < getParameters().tournamentSize; ++i) {
            int rand = rnd.nextInt(population.size());

            RHEAIndividual current = population.get(rand);
            if (best == null || current.value > best.value)
                best = current;
        }
        return best;
    }

    RHEAIndividual rankSelection() {
        population.sort(Comparator.naturalOrder());
        int rankSum = 0;
        for (int i = 0; i < population.size(); ++i)
            rankSum += i + 1;
        int ran = rnd.nextInt(rankSum);
        int p = 0;
        for (int i = 0; i < population.size(); ++i) {
            p += population.size() - (i);
            if (p >= ran)
                return population.get(i);
        }
        throw new RuntimeException("Random Generator generated an invalid goal, goal: " + ran + " p: " + p);
    }

    /**
     * Run evolutionary process for one generation
     */
    private void runIteration() {
        //copy elites
        RHEAParams params = getParameters();
        List<RHEAIndividual> newPopulation = new ArrayList<>();
        for (int i = 0, max = Math.min(params.eliteCount, population.size()); i < max; ++i) {
            newPopulation.add(new RHEAIndividual(population.get(i)));
        }
        //crossover
        for (int i = 0; i < params.childCount; ++i) {
            RHEAIndividual[] parents = selectParents();
            RHEAIndividual child = crossover(parents[0], parents[1]);
            population.add(child);
        }

        for (RHEAIndividual individual : population) {
            Pair<Integer, Integer> calls = individual.mutate(getForwardModel(), getPlayerID(), params.mutationCount);
            fmCalls += calls.a;
            copyCalls += calls.b;
            repairCount += individual.repairCount;
            nonRepairCount += individual.nonRepairCount;
            if (params.useMAST)
                MASTBackup(individual.actions, individual.value, getPlayerID());
        }

        //sort
        population.sort(Comparator.naturalOrder());

        //best ones get moved to the new population
        for (int i = 0; i < Math.min(population.size(), params.populationSize - params.eliteCount); ++i) {
            newPopulation.add(population.get(i));
        }

        population = newPopulation;

        population.sort(Comparator.naturalOrder());
        // Update budgets
        numIters++;
    }


    protected void MASTBackup(AbstractAction[] rolloutActions, double delta, int player) {
        for (int i = 0; i < rolloutActions.length; i++) {
            AbstractAction action = rolloutActions[i];
            if (action == null)
                break;

            // For Sushi Go, track by card type rather than exact action for better generalization
            Object mastKey = action;  // default to exact action
            if (action instanceof ChooseCard && !population.isEmpty()) {
                try {
                    // Try to extract card type for Sushi Go
                    AbstractGameState gs = population.get(0).gameStates[0];
                    if (gs instanceof SGGameState) {
                        ChooseCard cc = (ChooseCard) action;
                        mastKey = "CardType_" + ((SGCard)cc.getCard(gs)).type.name();
                    }
                } catch (Exception e) {
                    // Fall back to exact action if anything goes wrong
                    mastKey = action;
                }
            }

            Pair<Integer, Double> stats = MASTStatistics.get(player).getOrDefault(mastKey, new Pair<>(0, 0.0));
            stats.a++;  // visits
            stats.b += delta;   // value
            MASTStatistics.get(player).put(mastKey, stats);
        }
    }


    @Override
    public void setBudget(int budget) {
        parameters.budget = budget;
        parameters.setParameterValue("budget", budget);
    }

    @Override
    public int getBudget() {
        return parameters.budget;
    }

    /**
     * Enhanced heuristic wrapper for Sushi Go that adds two high-impact features:
     * 1. Set Completion Bonus System - rewards near-completion states
     * 2. Competitive Position Tracking - tracks competitive standing for Maki and Pudding
     */
    private static class SushiGoEnhancedHeuristic implements IStateHeuristic {
        private final IStateHeuristic baseHeuristic;

        public SushiGoEnhancedHeuristic(IStateHeuristic baseHeuristic) {
            this.baseHeuristic = baseHeuristic;
        }

        @Override
        public double evaluateState(AbstractGameState state, int playerID) {
            double baseScore = baseHeuristic.evaluateState(state, playerID);

            if (!(state instanceof SGGameState)) {
                return baseScore;
            }

            SGGameState sggs = (SGGameState) state;
            double enhancedScore = baseScore;

            // === FEATURE 1: SET COMPLETION BONUS SYSTEM ===
            enhancedScore += evaluateSetCompletionBonuses(sggs, playerID);

            // === FEATURE 2: COMPETITIVE POSITION TRACKING ===
            enhancedScore += evaluateCompetitivePosition(sggs, playerID);

            return enhancedScore;
        }

        /**
         * Feature 1: Set Completion Bonus System
         * Rewards states where sets are close to completion (Tempura pairs, Sashimi triplets)
         */
        private double evaluateSetCompletionBonuses(SGGameState sggs, int playerID) {
            double bonus = 0.0;

            int tempuraCount = sggs.getPlayedCardTypes(SGCard.SGCardType.Tempura, playerID).getValue();
            int sashimiCount = sggs.getPlayedCardTypes(SGCard.SGCardType.Sashimi, playerID).getValue();

            // Tempura needs 1 more to complete a pair (worth 5 points)
            // Reward if we have 1, 3, 5... (odd numbers)
            if (tempuraCount % 2 == 1) {
                // Very close to completing a pair - high reward
                bonus += 2.5; // Worth ~2.5 points when next card completes a 5-point pair
            } else if (tempuraCount > 0 && tempuraCount % 2 == 0) {
                // Not close to completion - slight penalty for incomplete sets
                bonus -= 0.5;
            }

            // Sashimi needs 1 more to complete a triplet (worth 10 points)
            // Reward if we have 2, 5, 8... (mod 3 == 2)
            if (sashimiCount % 3 == 2) {
                // Very close to completing a triplet - very high reward
                bonus += 3.3; // Worth ~3.3 points when next card completes a 10-point triplet
            } else if (sashimiCount % 3 == 1) {
                // One third complete - moderate reward
                bonus += 0.8;
            } else if (sashimiCount > 0 && sashimiCount % 3 == 0) {
                // Complete sets already, but slight recognition
                bonus += 0.2;
            } else if (sashimiCount > 0) {
                // Not close to completion
                bonus -= 0.3;
            }

            // Completed sets recognition bonus
            int completedTempuraPairs = tempuraCount / 2;
            bonus += completedTempuraPairs * 0.5;

            int completedSashimiTriplets = sashimiCount / 3;
            bonus += completedSashimiTriplets * 0.8;

            return bonus;
        }

        /**
         * Feature 2: Competitive Position Tracking
         * Tracks competitive standing for Maki (round-end) and Pudding (game-end)
         */
        private double evaluateCompetitivePosition(SGGameState sggs, int playerID) {
            double bonus = 0.0;

            int myMaki = sggs.getPlayedCardTypes(SGCard.SGCardType.Maki, playerID).getValue();
            int myPudding = sggs.getPlayedCardTypes(SGCard.SGCardType.Pudding, playerID).getValue();
            int currentRound = sggs.getRoundCounter();

            // Get opponent statistics
            double[] otherMakiCounts = IntStream.range(0, sggs.getNPlayers())
                    .filter(i -> i != playerID)
                    .mapToDouble(i -> sggs.getPlayedCardTypes(SGCard.SGCardType.Maki, i).getValue())
                    .toArray();

            double[] otherPuddingCounts = IntStream.range(0, sggs.getNPlayers())
                    .filter(i -> i != playerID)
                    .mapToDouble(i -> sggs.getPlayedCardTypes(SGCard.SGCardType.Pudding, i).getValue())
                    .toArray();

            // === Maki Competitive Position ===
            if (otherMakiCounts.length > 0) {
                double maxOtherMaki = Arrays.stream(otherMakiCounts).max().orElse(0.0);

                if (myMaki > maxOtherMaki) {
                    // Leading - reward based on lead margin
                    double leadMargin = Math.min(0.5, (myMaki - maxOtherMaki) / 5.0);
                    bonus += 1.0 + leadMargin;
                } else if (myMaki < maxOtherMaki) {
                    // Trailing - penalize based on deficit
                    double deficit = Math.min(0.5, (maxOtherMaki - myMaki) / 5.0);
                    bonus -= 1.0 + deficit;
                }

                // Second place bonus (only valuable if there's a clear leader and we're second)
                double[] sortedMaki = Arrays.stream(otherMakiCounts).sorted().toArray();
                double secondMaxMaki = sortedMaki.length >= 2 ? sortedMaki[sortedMaki.length - 2] :
                        (sortedMaki.length == 1 ? sortedMaki[0] : 0.0);

                if (myMaki < maxOtherMaki && myMaki >= secondMaxMaki && maxOtherMaki > secondMaxMaki) {
                    bonus += 0.5; // Second place is valuable (3 points if leader is unique)
                }
            }

            // === Pudding Competitive Position ===
            if (otherPuddingCounts.length > 0) {
                double minOtherPudding = Arrays.stream(otherPuddingCounts).min().orElse(0.0);

                if (myPudding > minOtherPudding) {
                    // Safe from last place - no penalty
                } else if (myPudding < minOtherPudding) {
                    // In danger of being last - strong penalty (critical to avoid -6 points)
                    bonus -= 2.0;
                } else {
                    // Tied for last - slight penalty
                    bonus -= 1.0;
                }
            }

            // === Round-Aware Pudding Priority ===
            // In round 3, pudding becomes critical for tiebreaker
            if (currentRound >= 2) { // Round 3 (0-indexed, so round 2 is the third round)
                bonus += myPudding * 1.5; // Strongly prioritize pudding in final round
            }

            return bonus;
        }
    }
}

