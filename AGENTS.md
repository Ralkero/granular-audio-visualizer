# Granular Music Visualizer development instructions

The canonical repository is https://github.com/Ralkero/granular-audio-visualizer.

The owner explicitly requested that all development milestones be uploaded here as the project progresses. Treat publishing completed milestones as part of the work, not as an optional follow-up requiring renewed approval.

For each milestone:

1. Implement and run appropriate automated checks and available integration tests.
2. Update the milestone and validation reports with implementation details, test results, known issues, measured performance and the recommended next milestone. Distinguish simulated tests from actual hardware validation.
3. Review tracked files for private runtime data and credentials. Commit source, tests, schemas, setup scripts and privacy-safe documentation. Keep session logs, audio, screenshots, local evidence, environments and build outputs local and ignored.
4. Commit coherent changes and push to the canonical repository. Use the existing branch workflow; preserve remote changes and never force-push. If branch protection requires a pull request, publish the milestone on a branch and open a pull request.
5. Verify the remote contains the intended commit, then report its link and any remaining validation limitations. If publication fails, report that failure explicitly rather than claiming the milestone was uploaded.

Keep Phase One capture and analysis independent of the future rendering engine. Do not change system volume, playback routing or other applications' behavior during development or validation. Raw audio recording remains off by default.
