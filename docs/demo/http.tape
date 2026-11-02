# Records docs/demo/http.gif from a real run: `vhs docs/demo/http.tape` from the repo root.
Output docs/demo/http.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 480
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate` Enter
Sleep 2s
Type `python -m rlhf_reward_model serve --port 18325 >/dev/null 2>&1 &` Enter
Sleep 2s
Sleep 18s
Type `clear` Enter
Sleep 1s
Show
Type `curl -s localhost:18325/score -d '{"prompt": 3}' | jq -c '.responses[] | {response, true_reward, learned_reward, policy_prob}'` Enter
Sleep 3.5s
Type `curl -s localhost:18325/report | jq -c` Enter
Sleep 3.5s
Hide
Type `kill %1` Enter
Sleep 1s
