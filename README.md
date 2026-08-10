# VolleyDash

A recruiting analytics dashboard for volleyball coaches: browse a knowledge-base
tree of stats, ask plain-language questions ("What is Sloan's Kills Per Set in
the Vegas Aces game?"), and author new derived metrics on the fly. LLM-backed
(Groq's free API) with a rule-based fallback when the LLM is unreachable.

## Architecture: two repos

This repo (**VolleyDash**, public) is code only -- `app.py`,
`recruiting_tree.py`, `recruiting_llm.py`, `recruiting_operations.py`,
`recruiting_data_store.py`. It contains no player data and never will
(`.gitignore` excludes `*.csv` and `recruiting_kb_data.json` specifically so
that can't happen by accident). This is the repo Streamlit Community Cloud
deploys from.

The real match CSVs (real player names, cannot be public) and
`recruiting_kb_data.json` (the committed knowledge tree) live in a **separate
private repo, VolleyData**. `VolleyDash` never touches that data locally --
`recruiting_data_store.py` is the ONE module that reaches it, entirely over
GitHub's Contents API, authenticated with a token scoped only to that repo.
That boundary is deliberate: if VolleyData is ever swapped for a real database
(e.g. Supabase), only `recruiting_data_store.py` changes -- `app.py` just calls
`save_committed_tree()`/`load_committed_tree()`/`get_games()`/`load_game_df()`
and never knows or cares where the bytes actually live.

## Local setup

```bash
pip install -r requirements.txt
```

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (gitignored,
never committed) and fill in:

- `GROQ_API_KEY` -- free key from https://console.groq.com/keys
- `GITHUB_DATA_REPO` -- `"owner/VolleyData"`
- `GITHUB_DATA_TOKEN` -- a fine-grained GitHub PAT
  (https://github.com/settings/personal-access-tokens) scoped to **only** the
  VolleyData repo, with **Contents: Read and write** permission

Or export the same three as environment variables instead.

```bash
streamlit run app.py
```

Without `GROQ_API_KEY`, LLM-backed features fall back to a rule-based parser
with a clear "LLM unreachable" message. Without `GITHUB_DATA_REPO`/
`GITHUB_DATA_TOKEN`, the app can't load or save anything at all (there's no
local fallback store) -- those two are required, not optional.

## Setting up VolleyData (one-time, by hand)

1. Create a **private** GitHub repo named `VolleyData`.
2. Upload `recruiting_kb_data.json` to its root, and every CSV into a
   `recruiting_csvs/` folder -- matching exactly what `recruiting_data_store.py`
   expects (`TREE_JSON_PATH`/`CSV_DIR_PATH`).
3. Create the fine-grained PAT described above and put it in
   `GITHUB_DATA_TOKEN`.

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub as `VolleyDash`.
2. Go to https://share.streamlit.io, "New app", point it at this repo/branch
   and `app.py`.
3. In the app's **Settings -> Secrets**, paste `GROQ_API_KEY`,
   `GITHUB_DATA_REPO`, and `GITHUB_DATA_TOKEN` (TOML format, same as
   `.streamlit/secrets.toml.example`).
4. Deploy. No GPU, no self-hosted server, no cost for the LLM calls at this
   workload's volume (Groq's free tier), and no player data anywhere in this
   public repo's history.
5. Under the app's **Settings -> Sharing**, restrict viewers to specific
   emails if you don't want the running dashboard's URL publicly loadable --
   repo privacy alone doesn't control that.

## Running tests

```bash
pytest test_integration.py test_data_store.py -v
```

Tests that need a live LLM call (`test_live_*`) skip gracefully if
`GROQ_API_KEY` isn't set or Groq is unreachable. `test_data_store.py` mocks
all GitHub HTTP calls, so it runs without real credentials.
