# CHANGE_REQUEST FB-X1: Codex read-only sandbox cannot run commands on host A

## Evidence (no Astra call made)
- The design requires `-s read-only` ("commands run but nothing writes"). The agent needs commands, even in A0, just to read its files.
- On host A, the Codex Linux sandbox (bubblewrap) cannot start any command:
  ```
  $ codex sandbox -P :read-only -C ~/langres_state/fb/sbtest echo hello
  bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
  ```
- The built-in `:workspace` profile fails the same way. Only `:danger-full-access` (no sandbox) runs commands.
- Cause: `kernel.apparmor_restrict_unprivileged_userns = 1` (Ubuntu AppArmor policy). It blocks the network namespace that bubblewrap creates. Node A2 has the same setting and has no Codex installed.
- The pilot never saw this because Astra ran no commands there.
- If we run A1 and A0 as designed, every command fails. Astra could then only answer from the prompt text, and the calls would be spent for nothing.

## Options
1. **Allow bubblewrap user namespaces (recommended; no change to the design).** You, as root, add an AppArmor profile for bwrap only:
   ```
   sudo tee /etc/apparmor.d/bwrap >/dev/null <<'P'
   abi <abi/4.0>,
   include <tunables/global>
   profile bwrap /usr/bin/bwrap flags=(unconfined) {
     userns,
     include if exists <local/bwrap>
   }
   P
   sudo apparmor_parser -r /etc/apparmor.d/bwrap
   ```
   - This is the standard Ubuntu 24.04 fix for bubblewrap. It affects bwrap for every user on host A.
   - I then re-test with `codex sandbox -P :read-only` (no model call): the venv imports, writes, and network access must all behave as the design expects. Only then do I run A1 and A0 under the frozen settings.
2. **Run without the sandbox** (`--dangerously-bypass-approvals-and-sandbox`), keeping the audit. Nothing would stop Astra writing files or using the network; the audit would only detect it afterwards. Writes outside the workspace and network use would void the call. Weaker protection; not recommended.
3. **Drop method 2.** Report method 1 (neighbors, Phi-4) only, and use no Astra calls.

## Side by side
Running v1 unchanged would make both agent calls command-less, so the result could not be compared fairly. I therefore propose no side-by-side v1 run.

Waiting for approval. Until then, FB-1 and FB-2 (method 1, no Astra) proceed.

## Decision
- 2026-11-18: USER APPROVED option 1. The user ran `sudo bash ~/langres_state/fb/bwrap_fix.sh` (the profile above) and reported "loaded".
- Re-test, with no model call: `codex sandbox -P :read-only` runs echo; the venv Python imports numpy 2.3.5 and sklearn 1.9.1. Writes in the workspace and under ~/langres_state are blocked (OSError), and a socket connect is blocked (PermissionError). Reads outside the workspace remain possible, as read-only mode intends; the audit covers them.
- The frozen call settings are unchanged.
