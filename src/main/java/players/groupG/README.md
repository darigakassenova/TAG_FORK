# SushiGo AI Tournament – GroupG RHEA Agent

## Overview

This project runs AI agents in the **SushiGo** environment to compare the performance of different algorithms.
The configuration files define parameters such as budget time, population size, and selection methods for each agent.

---

## Folder Structure

```
src/main/java/players/groupG/                           # Group6 RHEA implementation
src/main/java/players/group6/config/                    # JSON configuration files of agents
src/main/java/players/groupG/heuristics                 # sushigo heuristic
src/main/java/players/group6/results/                   # Output tournament results
src/main/java/players/groupG/rhea                       # GroupG RHEA agent 
src/main/java/players/groupG/sushi_tournament.json      # tournament set up
```

---

## How to Run


1.**Run a tournament**with with tournament configuration

   ```
     RunGames with program argument = config=src/main/java/players/groupG/sushi_tournament.json
   ```

    * The `sushi_tournament.json` file specifies:

        * Number of players (`nPlayers`)
        * Number of matchups
        * Mode (e.g., `"exhaustive"`)
        * Paths to player parameter files and results directory

2.**View results**
   After the run, results will be saved in:

   ```
   src/main/java/players/groupG/results/
   ```

   Open the generated text file to see scores, win rates, and rankings.

---

## Key Configuration Files

* `groupG_rhea.json` – Parameters for GroupG_RHEA (e.g., population size, mutation rate, budget time)
* `rhea.json` – Default RHEA parameters
* `basicmcts.json` – Parameters for Basic MCTS agent
* `tournament.json` – Tournament setup and agent list

---

## Notes

* To change computation budget, edit the `"budget"` value in `groupG_rhea.json`.


---


