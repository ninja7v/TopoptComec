# Updating TopoptComec

TopoptComec checks the official GitHub releases page:
<https://github.com/ninja7v/TopoptComec/releases>. Only stable releases are
offered; prereleases are ignored.

## Check for updates

```bash
topoptcomec update --check
```

This prints the installed version, the latest stable version, and the release
notes. It does not download or install anything.

## Update from the terminal

```bash
topoptcomec update
```

The command shows the new version and release notes, then asks for
confirmation. To install without a prompt (for scripts):

```bash
topoptcomec update --yes
```

Exit codes:

| Code | Meaning |
| ---- | ------- |
| `0` | Up to date, update installed, check-only run, or update cancelled |
| `1` | Network, release metadata, or installation error |
| `2` | Confirmation was required but the session is not interactive (`--yes` missing) |
| `3` | An update exists but must be installed manually |

`topoptcomec --version` prints the installed version.

## Update from the GUI

Click the update button at the top-left of the header (it replaces the former
info button). Clicking the **TopoptComec** title opens the project's GitHub page.

- If you already run the latest version, a short message says so.
- If an update is available, a Yes/No question asks whether to update.
- Progress is shown in the status bar; success asks you to restart TopoptComec.

The window stays responsive while the check or update runs. Only one update
check or installation can run at a time.

## Which installations can update automatically?

| Installation | Automatic update |
| ------------ | ---------------- |
| Git clone (the documented `git clone` + `uv run topoptcomec` workflow) | Yes, when the checkout is clean |
| Standalone PyInstaller executable | No; the app links to the release page |
| Package installed into `site-packages` | No; the project is not published on PyPI |

### Git checkout

The updater fetches tags from `origin` and fast-forwards the current branch to
the release tag with `git merge --ff-only`. It refuses to do anything when:

- there are uncommitted or untracked changes,
- local commits make a fast-forward impossible,
- the release tag is missing or does not match the release version.

Nothing is overwritten in those cases. Commit or stash your work, then run the
update again. Because the previous revision remains in the git history, you
can return to it with `git reset --hard ORIG_HEAD` if needed (this discards any
changes made after the update).

Running TopoptComec does not make the checkout dirty: `results/` is ignored by
git and the GUI saves your presets to `~/.topoptcomec/presets.json` instead of
the repository. Presets of a new release show up alongside your own ones.

### Standalone executable or installed package

No platform binaries are published yet, so the updater does not replace a
running executable or files inside `site-packages`. The CLI and GUI explain how
to update manually:

1. Open the release page printed by the update command/dialog.
2. Download the release.
3. Quit TopoptComec, replace the old executable (or reinstall the package),
   and start the new version.

For a source installation:

```bash
git fetch --tags origin
git checkout v4.0
uv sync
```

## Safety and failures

- Downloads require HTTPS and trusted GitHub hosts, are written to a temporary
  `.part` file, and are moved into place only after the SHA-256 digest
  published with the release matches. Artifacts without a published digest are
  not treated as verified.
- A checksum mismatch or interrupted download discards the partial file and
  leaves the existing installation untouched.
- A digest published alongside the release detects corruption. It is not a
  cryptographic signature: the updater does not claim that a release is
  authenticated.
- The CLI prints recovery instructions after a failed installation. The GUI
  keeps working and lets you retry or open the release page.
- Update checks are only performed when you run `topoptcomec update` or open
  the GUI update dialog. Application startup never requires the network.

## Release checklist for maintainers

The version has a single source: `topoptcomec/version.py`. `pyproject.toml`
reads it dynamically.

1. Bump `__version__` in `topoptcomec/version.py`.
2. Merge to `main`, then create and push a matching tag, e.g. `v3.1`
   (a leading `v` is optional when comparing versions).
3. Publish a GitHub release for that tag with the release notes.
4. Do not mark stable releases as prereleases.

The updater uses GitHub's `/releases/latest` endpoint, which already excludes
prereleases. Publishing optional downloadable binaries would allow future
automatic installation for standalone builds; the current updater deliberately
does not invent one.
