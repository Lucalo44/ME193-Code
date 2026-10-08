# UNO Q ↔ VS Code setup (Luca)

This folder holds Chris's `unoq-vscode-kit`, set up following Kat's instructions on the
"UNO Q tips" Notion page and Chris's `DevelopingUNOQAppsInVSCode.md`. In those docs this
folder is called `ArduinoApps`.

## Already done on the laptop

- [x] Kit unzipped here: `.vscode/tasks.json`, `tools/`, `CLAUDE.md`
- [x] `tools/board_secrets.py` created from the template. **It still has placeholder values.**
- [x] `.gitignore` ignores `.cache/`, `__pycache__/`, `.venv/` and `tools/board_secrets.py`
- [x] `zeroconf` installed in `.venv/`. The "Advertise UNO Q on Tufts WiFi" task uses
      `.venv/bin/python`, because Homebrew's Python refuses a global `pip install`.

## Still to do (needs you and the board)

Replace `<IP>` with the board's IP and `<BOARD>` with its name in the commands below.

1. **Find the board's IP and name.** In Arduino App Lab, go to Settings → Network
   Connections. While you're in Settings → System Info, set a password and turn on
   Remote access (SSH).
2. **Make a laptop SSH key** (press Enter at every question), then copy it to the board:
   ```bash
   ssh-keygen -t ed25519
   ssh-copy-id arduino@<IP>
   ssh arduino@<IP> hostname          # should print the name with no password prompt
   ```
3. **Edit `tools/board_secrets.py`:**
   ```python
   HOSTS = ["<IP>", "<BOARD>.local"]
   USER = "arduino"
   BOARD_NAME = "<BOARD>"
   BOARD_IP = "<IP>"
   ```
4. **On github.com, create an empty private repo named `ArduinoApps`.** Don't add a
   README, .gitignore or license.
5. **On the board** (`ssh arduino@<IP>`):
   ```bash
   git --version || sudo apt install -y git
   git config --global user.name "<BOARD> UNO Q"
   git config --global user.email "luca.lodewick@tufts.edu"
   ssh-keygen -t ed25519 -C "<BOARD>"
   cat ~/.ssh/id_ed25519.pub
   ```
   Copy the printed line into the repo's **Settings → Deploy keys → Add deploy key**, and
   check **Allow write access**. Then, still on the board:
   ```bash
   ssh -T git@github.com
   cd ~/ArduinoApps
   printf '.cache/\n__pycache__/\n*.pyc\n.DS_Store\n' > .gitignore
   git init -b main
   git add .
   git status                    # no .cache folders should be listed
   git commit -m "Initial commit from <BOARD>"
   git remote add origin git@github.com:Lucalo44/ArduinoApps.git
   git push -u origin main
   exit
   ```
6. **Bring the board's apps into this folder** (on the laptop):
   ```bash
   cd ~/Documents/GitHub/ME193-Code/ArduinoUNOq
   git init -b main
   git remote add origin https://github.com/Lucalo44/ArduinoApps.git
   git pull origin main
   git add . && git commit -m "Add deploy kit" && git push -u origin main
   ```
   This makes the folder its own repo inside ME193-Code. If you do this, add
   `ArduinoUNOq/` to ME193-Code's `.gitignore` so the two repos don't overlap.
7. **Test it.** In VS Code, use **File → Open Folder… → ArduinoUNOq**. The tasks only work
   when this folder is the open folder. Open a file inside one of your apps and press
   **Cmd+Shift+B**, then pick **Run on UNO Q**. A quick connection test:
   ```bash
   python3 tools/deploy.py --cmd "hostname"
   ```

## Cmd+Shift+B menu

| Task | What it does |
|---|---|
| Run on UNO Q | Copies the open file's app to the board and starts it |
| Python in UNO Q container | Opens a Python prompt inside the running app (or on the board) |
| Terminal on UNO Q | Opens a shell on the board |
| Advertise UNO Q on Tufts WiFi | Lets App Lab find the board on Tufts_Robotics. Press Ctrl+C to stop |

Daily use: edit here, deploy with Cmd+Shift+B, and commit/sync when it works. To make the
board match GitHub again:
`python3 tools/deploy.py --cmd "cd ~/ArduinoApps && git fetch && git reset --hard origin/main"`

Troubleshooting is at the end of Chris's `DevelopingUNOQAppsInVSCode.md` (ME193-Robotics repo,
`Public stuff/UNOQ code/`).
