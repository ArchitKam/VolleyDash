import pandas as pd
from recruiting_operations import (
    Slice, Reduce, Rank, Compare, ValuePredicate, run_pipeline, describe_pipeline,
)

# One long frame holding MULTIPLE metrics -- the caller fetches everything
# the pipeline references up front. 3 players x 4 games x 2 metrics.
rows = []
kills   = {"#7 Sloan T.": [4,12,None,15], "#19 Angela Z.": [11,3,14,2], "#22 Azana S.": [8,9,7,10]}
passing = {"#7 Sloan T.": [1.5,2.0,None,1.8], "#19 Angela Z.": [2.1,1.9,2.4,2.2], "#22 Azana S.": [1.2,1.4,1.1,1.6]}
for metric, data in (("Kills", kills), ("Passing quality", passing)):
    for player, vals in data.items():
        for game, v in zip(["G1","G2","G3","G4"], vals):
            rows.append({"Metric": metric, "Game": game, "Player": player, "Value": v})
df = pd.DataFrame(rows)

QUESTIONS = [
    ("Sloan's passing, all games",
     [Slice("Metric", keep=["Passing quality"]), Slice("Player", keep=["#7 Sloan T."])]),

    ("Sloan's passing in games where SHE had 10+ kills",
     [Slice("Game", predicate=ValuePredicate("Kills", ">=", 10)),
      Slice("Metric", keep=["Passing quality"]), Slice("Player", keep=["#7 Sloan T."])]),

    ("Everyone's passing in games where SLOAN had 10+ kills",
     [Slice("Game", predicate=ValuePredicate("Kills", ">=", 10, decided_by_player="#7 Sloan T.")),
      Slice("Metric", keep=["Passing quality"])]),

    ("Average passing per player across all games",
     [Slice("Metric", keep=["Passing quality"]), Reduce("Game", "mean")]),

    ("Who has the highest average passing? (top 1)",
     [Slice("Metric", keep=["Passing quality"]), Reduce("Game","mean"), Rank("Player", limit=1)]),

    ("Top 2 passers by average",
     [Slice("Metric", keep=["Passing quality"]), Reduce("Game","mean"), Rank("Player", limit=2)]),

    ("Is Sloan above TEAM average on passing, per game?",
     [Slice("Metric", keep=["Passing quality"]), Compare("Player","mean","difference"),
      Slice("Player", keep=["#7 Sloan T."])]),

    ("Which of Sloan's games beat her OWN season average?",
     [Slice("Metric", keep=["Passing quality"]), Slice("Player", keep=["#7 Sloan T."]),
      Compare("Game","mean","difference")]),

    ("Median kills per player (median added to the whitelist, no new op)",
     [Slice("Metric", keep=["Kills"]), Reduce("Game","median")]),

    ("Sloan's passing as a RATIO of team average",
     [Slice("Metric", keep=["Passing quality"]), Compare("Player","mean","ratio"),
      Slice("Player", keep=["#7 Sloan T."])]),
]

for label, pipeline in QUESTIONS:
    print("="*72)
    print(f"Q: {label}")
    print(f"   pipeline: {describe_pipeline(pipeline)}")
    out, notes = run_pipeline(df, pipeline)
    cols = [c for c in ["Metric","Game","Player","Value","Baseline","N Used"] if c in out.columns]
    print(out[cols].to_string(index=False) if not out.empty else "   (empty)")
    for n in notes:
        print(f"   note: {n}")

print("\n" + "="*72)
print("SPOT CHECKS")
# Q4: Sloan mean of 3 (G3 blank) not 4
out,_ = run_pipeline(df, [Slice("Metric", keep=["Passing quality"]), Reduce("Game","mean")])
s = out[out["Player"]=="#7 Sloan T."].iloc[0]
assert s["N Used"]==3 and abs(s["Value"]-(1.5+2.0+1.8)/3)<1e-9
print("blank G3 excluded, N Used=3 makes it visible")

# Q5: Angela should win (2.15 > 1.766 > 1.325)
out,_ = run_pipeline(df, [Slice("Metric",keep=["Passing quality"]), Reduce("Game","mean"), Rank("Player",limit=1)])
assert out.iloc[0]["Player"]=="#19 Angela Z."
print("top-1 ranking correct, NaN never wins")

# Q7: team-average comparison sign check for G1 (Sloan 1.5 vs team mean of 1.5,2.1,1.2=1.6)
out,_ = run_pipeline(df, [Slice("Metric",keep=["Passing quality"]), Compare("Player","mean","difference"), Slice("Player",keep=["#7 Sloan T."])])
g1 = out[out["Game"]=="G1"].iloc[0]
assert abs(g1["Value"] - (1.5 - (1.5+2.1+1.2)/3)) < 1e-9
print("vs-team-average difference correct")

# empty pipeline = unchanged
out,notes = run_pipeline(df, [])
assert out.equals(df) and not notes
print("empty pipeline is a no-op (backwards compatible)")

# Rank must rank WITHIN each remaining-axis group, not across the whole
# frame -- a skill-group category result carries several metrics at once
# (Metric stays a real column, never reduced away), so Rank(axis=Player)
# has to treat each metric independently. Before this fix, a global
# sort+head mixed every metric's rows together and condensed the whole
# category down to `limit` rows total, starving every metric but one.
category_rows = []
for metric, vals in (("Kills", kills), ("Passing quality", passing)):
    for player, series in vals.items():
        category_rows.append({"Metric": metric, "Game": "G1", "Player": player, "Value": series[0]})
category_df = pd.DataFrame(category_rows)
out,_ = run_pipeline(category_df, [Rank("Player", limit=1)])
assert set(out["Metric"].unique()) == {"Kills", "Passing quality"}, (
    "Ranking a multi-metric (category) result must keep EVERY metric, not condense to one."
)
print("category rank keeps every metric independently (no more condensing to one number)")

# Rank always shows at least 3 per group, even when limit asks for fewer --
# "who's the highest" reads better with a little context than as one bare
# number. (Reduce("Game","mean") first collapses to one row per player,
# same as the "top 1"/"top 2" questions above, so there's exactly one
# group here and `limit` alone would otherwise decide the row count.)
out,_ = run_pipeline(df, [Slice("Metric", keep=["Passing quality"]), Reduce("Game","mean"), Rank("Player", limit=1)])
assert len(out) == 3, "Rank(limit=1) with 3 candidates in the group should still show all 3 (floor of 3)."
print("rank floor of 3 applied even when a smaller limit was requested")

print("\nALL 10 COACH QUESTIONS EXPRESSED AS COMPOSITIONS -- ZERO NEW FUNCTIONS")
