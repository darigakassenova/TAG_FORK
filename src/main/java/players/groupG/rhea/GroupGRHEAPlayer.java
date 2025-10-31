package players.groupG.rhea;

import core.AbstractGameState;
import core.AbstractPlayer;
import core.actions.AbstractAction;
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

public class GroupGRHEAPlayer extends AbstractPlayer implements IAnyTimePlayer {
    private static final AbstractPlayer randomPlayer = new RandomPlayer();
    List<Map<Object, Pair<Integer, Double>>> MASTStatistics; // a list of one Map per player. Action -> (visits, totValue)
    protected List<RHEAIndividual> population = new ArrayList<>();

    // Budgets and diagnostics
    protected double timePerIteration = 0, timeTaken = 0, initTime = 0;
    protected int numIters = 0;
    protected int fmCalls = 0;
    protected int copyCalls = 0;
    protected int repairCount, nonRepairCount;

    private MASTPlayer mastPlayer;
    private GameType gameType;  // to enable SushiGo-specific logic

    public GroupGRHEAPlayer(RHEAParams params) {
        super(params, "GroupGRHEAPlayer");
    }

    public GroupGRHEAPlayer(RHEAParams params, String name) {
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

        // Track game type for domain-specific tweaks
        this.gameType = state.getGameType();

        // Initialise MAST priors if SushiGo is detected
        if (gameType == GameType.SushiGo) {
            initializeSushiGoMAST(state.getNPlayers());
        }
    }

    /**
     * SushiGo-specific MAST initialization: assign priors based on known card values.
     */
    private void initializeSushiGoMAST(int nPlayers) {
        String[] highValue = {"Sashimi", "SquidNigiri", "Tempura"};
        String[] mediumValue = {"SalmonNigiri", "Wasabi", "EggNigiri", "Dumpling"};
        String[] situational = {"Pudding", "Chopsticks"};

        for (int p = 0; p < nPlayers; p++) {
            Map<Object, Pair<Integer, Double>> stats = MASTStatistics.get(p);

            for (String card : highValue)
                stats.put("CardType_" + card, new Pair<>(20, 100.0)); // ≈5.0 avg
            for (String card : mediumValue)
                stats.put("CardType_" + card, new Pair<>(20, 60.0)); // ≈3.0 avg
            for (String card : situational)
                stats.put("CardType_" + card, new Pair<>(20, 30.0)); // ≈1.5 avg
        }
    }

    @Override
    public AbstractAction _getAction(AbstractGameState stateObs, List<AbstractAction> possibleActions) {
        ElapsedCpuTimer timer = new ElapsedCpuTimer();
        timer.setMaxTimeMillis(parameters.budget);

        numIters = 0;
        fmCalls = 0;
        copyCalls = 0;
        repairCount = 0;
        nonRepairCount = 0;
        RHEAParams params = getParameters();

        // === MAST Handling ===
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

        // === Population Initialization ===
        if (params.shiftLeft && !population.isEmpty()) {
            population.forEach(i -> i.value = Double.NEGATIVE_INFINITY);
            for (RHEAIndividual genome : population) {
                if (!budgetLeft(timer)) break;
                System.arraycopy(genome.actions, 1, genome.actions, 0, genome.actions.length - 1);
                genome.gameStates[0] = stateObs.copy();
                Pair<Integer, Integer> calls = genome.rollout(getForwardModel(), 0, getPlayerID(), true);
                fmCalls += calls.a;
                copyCalls += calls.b;
            }
        } else {
            population = new ArrayList<>();

            int randomCount = Math.max(1, params.populationSize / 2);
            int mastCount = params.populationSize - randomCount;

            // Random initialization
            for (int i = 0; i < randomCount; i++) {
                if (!budgetLeft(timer)) break;
                population.add(new RHEAIndividual(params.horizon, params.discountFactor, getForwardModel(),
                        stateObs, getPlayerID(), rnd, params.heuristic, randomPlayer));
                fmCalls += population.get(i).length;
                copyCalls += population.get(i).length;
            }

            // MAST-based initialization for informed exploration
            for (int i = 0; i < mastCount; i++) {
                if (!budgetLeft(timer)) break;
                AbstractPlayer initPlayer = params.useMAST ? mastPlayer : randomPlayer;
                population.add(new RHEAIndividual(params.horizon, params.discountFactor, getForwardModel(),
                        stateObs, getPlayerID(), rnd, params.heuristic, initPlayer));
                fmCalls += population.get(i).length;
                copyCalls += population.get(i).length;
            }
        }

        population.sort(Comparator.naturalOrder());
        initTime = timer.elapsedMillis();

        // === Evolution Loop ===
        while (budgetLeft(timer)) runIteration();

        timeTaken = timer.elapsedMillis();
        timePerIteration = numIters == 0 ? 0.0 : (timeTaken - initTime) / numIters;

        // === Select Action ===
        AbstractAction retValue = population.get(0).actions[0];
        List<AbstractAction> actions = getForwardModel().computeAvailableActions(stateObs, params.actionSpace);
        if (!actions.contains(retValue))
            throw new AssertionError("Invalid action: " + retValue + " after " + numIters + " iterations");
        return retValue;
    }

    private boolean budgetLeft(ElapsedCpuTimer timer) {
        RHEAParams params = getParameters();
        if (params.budgetType == PlayerConstants.BUDGET_TIME)
            return timer.remainingTimeMillis() > params.breakMS;
        if (params.budgetType == PlayerConstants.BUDGET_FM_CALLS)
            return fmCalls < params.budget;
        if (params.budgetType == PlayerConstants.BUDGET_COPY_CALLS)
            return copyCalls < params.budget && numIters < params.budget;
        if (params.budgetType == PlayerConstants.BUDGET_FMANDCOPY_CALLS)
            return (fmCalls + copyCalls) < params.budget;
        if (params.budgetType == PlayerConstants.BUDGET_ITERATIONS)
            return numIters < params.budget;
        throw new AssertionError("Unexpected budget type: " + params.budgetType);
    }

    @Override
    public GroupGRHEAPlayer copy() {
        RHEAParams newParams = (RHEAParams) parameters.copy();
        newParams.setRandomSeed(rnd.nextInt());
        GroupGRHEAPlayer retValue = new GroupGRHEAPlayer(newParams, toString());
        retValue.setForwardModel(getForwardModel());
        return retValue;
    }

    // === Evolution operators ===
    private void runIteration() {
        RHEAParams params = getParameters();
        List<RHEAIndividual> newPopulation = new ArrayList<>();

        for (int i = 0, max = Math.min(params.eliteCount, population.size()); i < max; i++)
            newPopulation.add(new RHEAIndividual(population.get(i)));

        for (int i = 0; i < params.childCount; i++) {
            RHEAIndividual[] parents = selectParents();
            RHEAIndividual child = crossover(parents[0], parents[1]);
            population.add(child);
        }

        for (RHEAIndividual ind : population) {
            Pair<Integer, Integer> calls = ind.mutate(getForwardModel(), getPlayerID(), params.mutationCount);
            fmCalls += calls.a;
            copyCalls += calls.b;
            repairCount += ind.repairCount;
            nonRepairCount += ind.nonRepairCount;
            if (params.useMAST) MASTBackup(ind.actions, ind.value, getPlayerID());
        }

        population.sort(Comparator.naturalOrder());
        for (int i = 0; i < Math.min(population.size(), params.populationSize - params.eliteCount); i++)
            newPopulation.add(population.get(i));

        population = newPopulation;
        population.sort(Comparator.naturalOrder());
        numIters++;
    }

    private RHEAIndividual[] selectParents() {
        RHEAIndividual[] parents = new RHEAIndividual[2];
        switch (getParameters().selectionType) {
            case TOURNAMENT -> {
                parents[0] = tournamentSelection();
                parents[1] = tournamentSelection();
            }
            case RANK -> {
                parents[0] = rankSelection();
                parents[1] = rankSelection();
            }
            default -> throw new RuntimeException("Unexpected selection type");
        }
        return parents;
    }

    private RHEAIndividual tournamentSelection() {
        RHEAIndividual best = null;
        for (int i = 0; i < getParameters().tournamentSize; i++) {
            RHEAIndividual candidate = population.get(rnd.nextInt(population.size()));
            if (best == null || candidate.value > best.value)
                best = candidate;
        }
        return best;
    }

    private RHEAIndividual rankSelection() {
        population.sort(Comparator.naturalOrder());
        int rankSum = (population.size() * (population.size() + 1)) / 2;
        int randomRank = rnd.nextInt(rankSum);
        int cumulative = 0;
        for (int i = 0; i < population.size(); i++) {
            cumulative += population.size() - i;
            if (cumulative >= randomRank) return population.get(i);
        }
        return population.get(0);
    }

    private RHEAIndividual crossover(RHEAIndividual p1, RHEAIndividual p2) {
        return switch (getParameters().crossoverType) {
            case NONE -> new RHEAIndividual(p1);
            case UNIFORM -> uniformCrossover(p1, p2);
            case ONE_POINT -> onePointCrossover(p1, p2);
            case TWO_POINT -> twoPointCrossover(p1, p2);
        };
    }

    private RHEAIndividual uniformCrossover(RHEAIndividual p1, RHEAIndividual p2) {
        RHEAIndividual child = new RHEAIndividual(p1);
        copyCalls += child.length;
        for (int i = 0; i < Math.min(p1.length, p2.length); i++) {
            if (rnd.nextBoolean()) {
                child.actions[i] = p2.actions[i];
                child.gameStates[i] = p2.gameStates[i];
            }
        }
        return child;
    }

    private RHEAIndividual onePointCrossover(RHEAIndividual p1, RHEAIndividual p2) {
        RHEAIndividual child = new RHEAIndividual(p1);
        int crossoverPoint = child.length / 2;
        for (int i = crossoverPoint; i < child.length; i++) {
            child.actions[i] = p2.actions[i];
            child.gameStates[i] = p2.gameStates[i];
        }
        return child;
    }

    private RHEAIndividual twoPointCrossover(RHEAIndividual p1, RHEAIndividual p2) {
        RHEAIndividual child = new RHEAIndividual(p1);
        int segment = child.length / 3;
        for (int i = 0; i < segment; i++) {
            child.actions[i] = p2.actions[i];
            child.actions[child.length - 1 - i] = p2.actions[p2.length - 1 - i];
            child.gameStates[i] = p2.gameStates[i];
        }
        return child;
    }

    // === MAST backup ===
    protected void MASTBackup(AbstractAction[] rolloutActions, double delta, int player) {
        for (AbstractAction action : rolloutActions) {
            if (action == null) break;
            Object mastKey = action;

            // SushiGo-specific generalization: group by card type instead of raw action
            if (action instanceof ChooseCard && gameType == GameType.SushiGo) {
                try {
                    AbstractGameState gs = population.get(0).gameStates[0];
                    SGCard card = (SGCard) ((ChooseCard) action).getCard(gs);
                    mastKey = "CardType_" + card.type.name();
                } catch (Exception ignored) {}
            }

            Pair<Integer, Double> stats = MASTStatistics.get(player).getOrDefault(mastKey, new Pair<>(0, 0.0));
            stats.a++;
            stats.b += delta;
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
}
