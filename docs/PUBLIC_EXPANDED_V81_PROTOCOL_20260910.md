# Expanded public validation v8.1: logging-only retry

2026-09-10. This protocol incorporates the complete scientific design, cohort,
metrics, progression gates, eight-thread CPU configuration and resource limits
of [v8](PUBLIC_EXPANDED_V8_PROTOCOL_20260910.md), unchanged. It overrides only the
generic runtime output-symlink guard for one W&B SDK debug-log link. Freeze this
new runner/protocol and register fresh roles/execution before retrying.

## Preserved failed attempt

The original execution contract is
`e4a21b250f780b9acb02a3ccab62d5152975b89930d4e6093966c6d3ecb54bb9`.
The first control-reference model/prediction shards were written, then the
runtime guard rejected a W&B debug-log symlink before the first tracking event.
No learned model, inner selection or complete arm/suite was produced.
No scientific outcomes were inspected to choose this operational fix.

The failed control tracking receipt is
`8be1cebef0d8dcdd32156a63a446fa3bbae566173caf723c5cf776f3fe3a8052`:
failed, exit code 1, zero events, no cloud synchronization. Preserve the original
directory, files, code and protocol; new execution authenticates this lineage.

## Root cause and precise exception

The installed W&B SDK 0.19.11 generated:

`tracking/control/sdk/wandb/offline-run-20260910_114702-2712eb0a481f/logs/debug-core.log`

as a link to its user-cache `core-debug-20260910_114702.log`. The SDK's
`symlink=False` setting does not prevent this core-generated link. This is a
logging-layout problem, not a GPU, dataset, numerical or model failure.

The new runtime guard permits only a file matching:

`tracking/<registered-arm>/sdk/wandb/offline-run-<YYYYMMDD_HHMMSS>-<12hex-run-id>/logs/debug-core.log`

The folder run ID must match the regular local tracking receipt. Require the
same offline validation group and no cloud synchronization. The link's lexical
target must be exactly under the current user's `.cache/wandb/logs` with the
basename `core-debug-YYYYMMDD_HHMMSS.log`. Do not resolve, stat, open or read the
target. Count only the link inode's byte length with `lstat`; external debug-log
target bytes are explicitly outside the project-output accounting. The target
is not scientific evidence and is never uploaded by this workflow.

Continue to reject directory symlinks, arbitrary links, wrong tracking IDs,
wrong targets, sockets/FIFOs and all symlinks in model/prediction/receipt inputs.
Do not weaken the scientific shard reader or saved-head audit. Other output
files must be regular. Dangling permitted debug links are acceptable precisely
because no target is followed.

The new execution schema is `public-expanded-regularization-execution-v8.1`.
The frozen v8 model, registration logic, feature tables, scalar tracker and
W&B group remain unchanged; distinct run IDs and new contract hashes distinguish
this retry. It restarts all seven arms in a fresh directory; failed partial
shards are not adopted as trained checkpoints. No promoter/Feng/submission
progression follows from fixing logging.
