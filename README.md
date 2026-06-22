# 📁 Internship Task Submissions — [Crowd Surveillance]

Welcome to the official repository for all internship task submissions.
This repo is maintained by the Team Leads. Every intern is expected to follow the guidelines below **strictly** before pushing any code or output.

---

## 🗂️ Folder Naming Rules

Each group alloted with a task must create their own single folder(Only one folder per taskID, made by any of the team's member) in the following format:

```
TaskID_TaskName/
```

**Examples:**
```
Task115_ZoneViolationDetection/
Task02_ModelTraining/
Task03_Visualization/
```

> ❌ Do NOT use random names like `task`, `mywork`, `test123`, or leave spaces in folder names.

---

## 📝 What Goes Inside Your Folder

Each task folder must contain:
- Your code files (`.py`, `.ipynb`, etc.)
- Output files (screenshots, CSVs, PDFs, model(pth) files — whatever applies)
- A short `notes.md` or `README.md` explaining what the task was and what you did

---
## 📂 Folder Structure(To be maintained strictly as per the rules)

Task115_ZoneViolationDetection/
│
├── Code/                   ← Your scripts and notebooks (.py, .ipynb)
│   ├── train.py
│   ├── detect.py
│   └── utils.py
│
├── Outputs/                ← Results: images, videos, CSVs your model produced
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── demo_clip.mp4
│
└── Models/                 ← Saved model weight files (.pt, .pth, .onnx)
    ├── best.pt
    ├── last.pt
    └── checkpoint.pth

---    

## 💬 Commit Message Rules

A commit message must clearly describe **what you did**. Be specific.

**✅ Good examples:**
```
Add Task 01: Data cleaning script with null value handling
Update Task 02: Fixed model accuracy issue in training loop
Add Task 03: Final output CSV and visualization screenshots
```

**❌ Bad examples:**
```
done
updated
fix
asdfgh
final final v2
```

> Think of commit messages as notes to your TL and teammates. If it doesn't make sense to someone reading it cold, rewrite it.

---

## 🔀 Pull Request (PR) Process — MANDATORY

**Do NOT push directly to `main`.** All code must go through a Pull Request so it can be reviewed before merging.

### Steps to follow:

1. **Create your own branch** before making any changes:
   ```bash
   git checkout -b YourName-TaskNumber
   ```
   Example: `git checkout -b Rahul-Task01`

2. **Add your files and commit:**
   ```bash
   git add .
   git commit -m "Add Task 01: Brief description of what you did"
   ```

3. **Push your branch to GitHub:**
   ```bash
   git push origin YourName-TaskNumber
   ```

4. **Open a Pull Request on GitHub:**
   - Go to the repo on GitHub
   - Click **"Compare & pull request"**
   - Add a clear title and description of your task
   - Request review from your TL
   - Wait for approval — **do not merge yourself**

---

## ⚠️ Important Rules

- Never push directly to `main` — PRs only
- Never overwrite or modify another intern's folder even by mistake
- Always double-check your folder name and commit message before pushing
- If you're unsure about anything, ask your TL before pushing

---

## 👤 Team Lead
For any doubts, reach out before pushing anything wrong. It's always better to ask.
