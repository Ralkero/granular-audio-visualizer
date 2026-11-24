# Granular Music Visualizer development instructions

These instructions apply to every agent working on Granular Music Visualizer (GMV): Claude implementing on Windows or Linux, and ChatGPT or Codex reviewing on GitHub.

The canonical repository is https://github.com/Ralkero/granular-audio-visualizer. `main` is the stable baseline. Changes reach it only through a pull request that the owner has explicitly authorized to merge.

A phase goes through these steps: planning, implementation on a feature branch, a pull request against `main`, independent review, the owner's testing, refinement, the owner's authorization to merge. The next phase starts from the updated `main`.

## Roles

- **The owner** is the product decision-maker. They set direction and priorities, approve plans, test the real application, and decide whether a phase is accepted and merged. They do not read or review code.
- **Claude** implements and validates each phase, and is responsible for its technical quality.
- **ChatGPT** reviews independently. It inspects the actual changes on GitHub, not Claude's summary of them, and gives the owner a plain-language verdict. A review, however positive, does not merge anything and is not the owner's authorization.

Never hand the owner a technical question to resolve. Explain decisions, failures, limits and risks in plain language, as consequences for GMV: what it will and will not do, and what could go wrong.

## Each phase

1. **Branch.** Start from the current `main`, on an isolated feature branch in its own worktree. Preserve existing behaviour and make no changes unrelated to the phase.
2. **Test.** Write automated tests for the new behaviour. Run the test suite and the available integration tests, and validate on real hardware when possible. Investigate failures instead of dismissing or skipping them.
3. **Document.** Update the phase report and the validation report with implementation details, test results, measured performance, known problems and the recommended next phase. Documentation describes what is actually implemented and tested, not what is planned or expected.
4. **Review for privacy.** Check tracked files for private runtime data and credentials. Commit source, tests, schemas, setup scripts and privacy-safe documentation. Keep session logs, audio, screenshots, local evidence, environments and build outputs local and ignored.
5. **Publish.** Commit focused changes, push the feature branch and open a pull request against `main` (see below). Never push directly to `main`. Preserve remote changes and never force-push.
6. **Verify.** Confirm that the remote branch contains the intended commit, then report the pull request link and what remains unvalidated. If pushing or opening the pull request fails, say so plainly instead of claiming the phase was published.

Pushing the finished feature branch and opening its pull request are part of every phase, not a follow-up that needs renewed approval. Merging is not: it needs the owner's explicit authorization.

## Pull requests

- One phase per pull request, against `main`, with focused commits.
- Open it as a **Draft** while the owner's manual testing or other validation is outstanding, or while the work is still being refined. Open it as a regular pull request, or mark the draft ready for review, when the work is ready for formal review.
- The description says:
  - what the phase accomplishes;
  - the important changes;
  - which automated tests, integration tests and hardware validation were done;
  - what is untested;
  - known problems;
  - the manual tests the owner should perform, as concrete steps: what to launch, what to play, what to watch for, and what to report back.

## Merging

Never merge without the owner's explicit authorization. Opening a pull request, passing CI, marking it ready for review or a positive review is not permission to merge.

Commit, push, opening a pull request, marking it ready for review and merging are separate operations. When reporting, say which of them were done. After a merge, the updated `main` is the stable baseline and the next phase starts from it.

## Evidence

Keep three kinds of evidence apart, in reports, pull requests and messages to the owner:

- **Automated tests:** what the test suite checks.
- **Simulated or stand-in integration:** GMV or its parts run against substitutes, such as generated audio, replayed sessions, or virtual or null devices.
- **Real-hardware validation:** GMV running on a real machine, with its real audio devices and real playback.

Never present simulated or stand-in results as hardware validation. Be plain about uncertainty and unfinished work.

## Long jobs

Run tests, builds and evaluations visibly, in a progress window or a terminal the owner can see, never silently in the background.

## Standing rules

Keep Phase One capture and analysis independent of the future rendering engine. Do not change system volume, playback routing or other applications' behavior during development or validation. Raw audio recording remains off by default.
