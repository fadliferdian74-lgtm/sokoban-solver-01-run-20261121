[![make.fun bounty](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/fadliferdian74-lgtm/sokoban-solver-01-run-20261121/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/fadliferdian74-lgtm/sokoban-solver-01-run-20261121/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

# sokoban-solver

A push-optimal Sokoban solver, written for the make.fun bounty
[sokoban-solver-01-run-20261121](https://make.fun/bounty/sokoban-solver-01-run-20261121).
`main.ts` (Node 22, no dependencies) reads every level under `/input/<id>/level.txt` and
writes the player's moves to `/output/<id>/solution.txt`, in as few pushes as the level
allows.

How it works:

- **Decomposition.** These levels are rooms off a hallway, each joined to the board through
  one door square, with as many goals as boxes inside; a box can never leave its room, so a
  level's push optimum is the sum of its rooms' optima. The solver finds squares whose
  removal seals off such a part, solves each part on its own (player starting on the
  hallway square outside the door, and required to be able to walk back to it), and walks
  the player between parts. Interior splits that will not solve are struck out and the
  next candidate is tried; whatever remains is solved whole.
- **Push search.** Each (sub)level is solved by A* over pushes. A state is the box squares
  plus the player's reachable region; the estimate is a minimum-cost matching between
  boxes and goals over lone-box push distances, which lower-bounds the pushes remaining.
  Dead squares and frozen boxes (a greatest-fixed-point test) prune states before they
  are queued. The first goal state settled is push-optimal.
- **Determinism.** All budgets are counted in search expansions, never in time, so two
  runs on the same input write the same bytes. A greedy best-first fallback still tries
  to answer a level whose optimal search exhausts its budget.
