---
name: goal
description: Long-horizon goal mode. Use whenever a goal file exists at .goal/GOAL.md, or you were told to keep working until a condition is met. Re-read the goal every turn, work from evidence, run the checks, and only report done when verified.
---

# Goal mode

You are pursuing a long-horizon goal. You will be resumed turn after turn until
the goal is verifiably met or a hard limit (wall clock / turns / tokens) is hit.
Work autonomously; never stop to ask for confirmation.

## Every turn, in order

1. **Re-read the goal.** Open `.goal/GOAL.md` in the workspace and read it in
   full. It holds the objective, the acceptance checks, constraints, and an
   optional `check:` command. Treat it as the source of truth, not your memory
   of it.
2. **Orient from evidence, not assumption.** Inspect the real state of the
   workspace (files, test output, running services) before deciding what to do.
   Do not trust earlier turns' claims; re-verify.
3. **Make progress.** Do the next concrete thing that moves the objective
   forward. Prefer small, verifiable steps.
4. **Run the checks.** If `GOAL.md` names a `check:` command, run it and read
   its full output. Otherwise run the project's own tests/build. Never declare
   success on a check you did not actually run this turn.
5. **Record status.** Append a short, dated note to `.goal/PROGRESS.md` (what
   you did, what the check showed, what is left). Write `.goal/STATUS` as a
   single line **only when verified**:
   - `complete` — every acceptance check passed in output you can see now.
   - `blocked: <one-line reason>` — only after the *same* blocker has recurred
     for 3 consecutive turns and you have exhausted alternatives.
   Leave `.goal/STATUS` absent/unchanged while still working.

## Rules

- **Never narrow the objective.** Do not redefine "done" to something easier,
  skip an acceptance check, or declare victory on a subset. If the goal is
  impossible, keep working until the deadline rather than falsely completing.
- **Completion audit before `complete`.** Map every requirement in `GOAL.md` to
  concrete evidence (a passing check, a file that exists, a command's output).
  If you cannot point to evidence for a requirement, it is not done.
- **No silent scope changes.** If a constraint blocks progress, record it in
  `PROGRESS.md` and route around it; do not quietly drop it.
- An outer supervisor re-runs the check itself before trusting `complete`, so a
  false `complete` just wastes a turn — be honest.
