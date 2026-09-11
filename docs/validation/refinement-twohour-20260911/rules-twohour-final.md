# Rules

Participant-facing and served by the site. Changing a rule needs the organising team's sign-off.

## Dates

| What | When |
|---|---|
| Registration | open now, closes 11 September 11:00 |
| Qualifier ladder | 4 to 11 September |
| Rated rounds | every hour, 08:00 to 22:00, from 4 September 08:00 |
| Daily Five | 6 to 10 September |
| Uploads close | 11 September 11:00 |
| Team changes close | 11 September 11:00. Creating, joining and leaving a team all stop |
| Final qualification | 11 September afternoon, a 13-round Swiss over locked builds |
| Finalist invites | 11 September, once the final Swiss has run |
| Live final | 12 September at Encode Club, London |

## Teams and eligibility

- Teams are 1 to 3 people, and a person is on one team.
- Entry is open worldwide.
- A team enters the final Swiss if at least one member is a UK university student.
- Only a team's UK university students can take a London seat, and that is verified before invites go out.

## Match conditions

| Condition | Rule |
|---|---|
| Time control | 120 s plus 0.5 s per move, per side, on wall time |
| Init budget | 90 s to import your agent, before the clock starts |
| Hardware | one core of an AMD EPYC 9V74, measured at 2.60 GHz. 2 GB RAM. No network. No GPU. Identical for every game |
| Environment | Python 3.12 with torch, numpy, python-chess, onnxruntime and numba preinstalled at the fixed versions the docs page lists. Nothing else installs and a `requirements.txt` in the zip is ignored |
| Packages | ask hello@aichessathon.com for a package the stack lacks. Any addition is announced to every team |
| Pondering | your process is suspended while your opponent thinks, so work you leave running between your own moves does not run. Each side has the core to itself while it thinks |

## How a game ends

- An illegal move, malformed output, a crash, running out of memory or missing the init budget loses the game. A move payload over 4 KB counts as an illegal move.
- Losing on time loses unless the other side has no way to mate, and then the game is a draw.
- If both sides fail the game is void.
- Draws follow FIDE rules. A game still running at 600 plies is a draw, and the opening position counts toward the 600.
- Every game starts from a curated opening position that is close to level. The set is not published.
- Knockout ties play each position once with each colour.

## Ranking and qualification

- The ladder ranks by a standard rating fitted to your current build's results.
- The ladder only seeds the final Swiss.
- House bots and house engines play the ladder. The engines keep a fixed rating, which holds the scale. None can qualify.
- Only the locked-build final Swiss counts for qualification, by points. An odd field gives one team a 1-point bye.
- Tie-breaks are points, then Buchholz, then head-to-head, then the earlier final submission.
- A level knockout tie goes to the better final Swiss standing.

## London seats

- The final Swiss decides the London field.
- The room holds 50 people and a person takes one seat.
- Seats fill in seed order, one per UK university student on a team.
- Invites go out in that order and are confirmed by reply, first come, until the room is full.

## Prizes

£1,000 to the winner, £500 to the runner-up and £250 to third, paid at the London final. Third is the losing semi-finalist with the better final Swiss standing. A prize goes to the team, split as the team decides.

## Daily Five

- 6 to 10 September. One attempt a day, five positions, 20 minutes.
- One wrong move ends a position.
- Your five are drawn for you.
- Start by 23:40 London.
- Anyone signed in may play.

### Ranking

- Positions solved, then the time of your last solve, then who finished first.
- A position withdrawn as broken counts as solved and adds no time.

### Wildcards

- The top 3 eligible participants each day earn a Finals Day Wildcard. That is a seat at the London final, not a place in the bracket.
- One per person across the five days, so places roll down.
- A wildcard won by someone who also takes a seat through the final Swiss rolls down the same way.
- Eligible means a UK university student who is not an organiser and not on a disqualified team.

### Fair play

- No engines, no other people, no other accounts, no looking positions up.
- Every move is timed.
- Invitations follow review and may include solving a position in person at the final.

## Submissions

| Limit | Value |
|---|---|
| Size | at most 50 MB unzipped |
| Uploads | 10 per team per day |
| Which plays | the latest upload that passed validation |

## What you may ship

### Engines

- Third party engines are prohibited. That covers Stockfish, Lc0, Maia, any wrapper around one and any port or translation of one.
- Your moves come from code you wrote. An engine you wrote yourself before the event is your own code.
- Use any AI support you like to write it, as long as the submission keeps to these rules and you can explain it when asked.
- A model is not required. A classical search is a full entry.

### Models and training data

- Any network you ship is one you trained yourself. Starting from a published chess network is not allowed, so fine-tuning or re-exporting one counts as shipping it.
- Training data is unrestricted, including positions annotated by an existing engine. What ships inside the zip is what the ban covers.

### Books and tablebases

- A table you ship and read during a game may answer the opening or the endgame. The middlegame you search yourself.
- The opening is a position whose move number is 20 or lower. The endgame is a position of at most 7 pieces, counting both kings.
- Both are read from the position you are given, and neither depends on what produced the table.
- A table that answers a middlegame position is a stored search and counts as an engine.
- Books and tablebases count against the 50 MB cap.
- `chess.polyglot` and `chess.syzygy` are in the base image.

### Code

- What you ship must be source a judge can read. Obfuscated agents are disqualified.
- Everything that runs is Python from your zip plus the preinstalled stack.

### Verification

- Every submission faces automated and human checks.
- Each finalist team walks through how its agent was built, and a team that ships a network shows how it was trained.
- Disqualification can be retroactive.
