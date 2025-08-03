
"""
tk_review_danbooru.py  –  tag‑by‑tag reviewer for Danbooru‑style datasets
=========================================================================
• ONE proposed tag at a time – A=accept · R=reject · B=back · E=edit · N=next
• Side panel shows filename, Added / Removed / Original, **and live “Final” output**
• Edit (E) opens a scrollable one‑tag‑per‑line dialog that edits ONLY the final output
• “Final” preview updates instantly after every Accept / Reject / Back / Edit
• Saves a clean two‑column CSV (file_name, original_tags) to a new file

INSTALL  →  pip install pillow pandas
-----------------------------------------------------------------
"""

import pathlib, re, pandas as pd, tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ───────────── EDIT THESE PATHS ─────────────
CSV_PATH         = r"D:\TrainingCandidates\tag_review2.csv"            # source CSV
IMAGE_DIR        = pathlib.Path(r"D:\TrainingCandidates\images")
EDITED_CSV_PATH  = r"D:\TrainingCandidates\reviewzed.csv"     # output CSV
# ────────────────────────────────────────────

MAX_IMG        = (700, 700)
BACKUP_EVERY   = 1                      # write CSV every N images
_SPLIT_RX      = re.compile(r"[,\s]+")  # split on commas *or* whitespace


# ---------------- helper utils ----------------
def split_tags(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    return [t for t in _SPLIT_RX.split(str(val).strip()) if t]


def join_space(tags):
    return " ".join(sorted(tags))


def join_lines(tags):
    return "\n".join(sorted(tags))


def clean_fname(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s
# ------------------------------------------------------


class EditDialog(tk.Toplevel):
    """Large scrollable editor – one tag per line."""
    def __init__(self, parent, current_tags):
        super().__init__(parent)
        self.title("Edit final tag list – one per line")
        self.transient(parent)
        self.grab_set()

        tk.Label(self, text="Edit tags below (one per line):").pack(pady=(6, 0))

        frame = tk.Frame(self); frame.pack(padx=10, pady=6)
        self.text = tk.Text(frame, width=40, height=15, wrap="none")
        self.text.pack(side="left", fill="both", expand=True)
        self.text.insert("1.0", join_lines(current_tags))

        scrollbar = tk.Scrollbar(frame, command=self.text.yview)
        scrollbar.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=scrollbar.set)

        btns = tk.Frame(self); btns.pack(pady=6)
        tk.Button(btns, text="Save (Ctrl+Enter)", command=self._save).pack(side="left", padx=4)
        tk.Button(btns, text="Cancel (Esc)",      command=self._cancel).pack(side="left", padx=4)

        self.bind("<Escape>", lambda e: self._cancel())
        self.text.bind("<Control-Return>", lambda e: self._save())

        self.result = None  # will hold a set of tags or None

    def _save(self):
        raw = self.text.get("1.0", "end").strip()
        self.result = {t for t in _SPLIT_RX.split(raw) if t}
        self.destroy()

    def _cancel(self):
        self.destroy()


class TagReviewer:
    # ---------- Tkinter callbacks ----------
    def accept(self, *_):
        if self.ptr >= len(self.queue): return
        mode, tag = self.queue[self.ptr]
        (self.tags.add if mode == "ADD" else self.tags.discard)(tag)
        self.log.append((mode, tag, "ACC"))
        self.ptr += 1
        self._after_tag_change()

    def reject(self, *_):
        if self.ptr >= len(self.queue): return
        mode, tag = self.queue[self.ptr]
        self.log.append((mode, tag, "REJ"))
        self.ptr += 1
        self._after_tag_change()

    def back(self, *_):
        if not self.log:
            messagebox.showinfo("Info", "Already at first tag."); return
        self.ptr -= 1
        mode, tag, decision = self.log.pop()
        if decision == "ACC":
            (self.tags.discard if mode == "ADD" else self.tags.add)(tag)
        self._after_tag_change()

    def manual_edit(self, *_):
        dlg = EditDialog(self.root, self.tags)
        self.root.wait_window(dlg)
        if dlg.result is not None:
            self.tags = dlg.result
            self._after_tag_change()

    def next_image(self, *_):
        if self.ptr < len(self.queue): return  # still evaluating tags
        self.df.at[self.idx, "original_tags"] = join_space(self.tags)
        self.reviewed += 1
        if self.reviewed % BACKUP_EVERY == 0: self._persist_edits()
        self.idx += 1
        if self.idx >= len(self.df):
            self._persist_edits()
            messagebox.showinfo("Done", "All images reviewed!")
            self.root.quit()
        else:
            self._load_row()

    # ---------- internal helpers ----------
    def _persist_edits(self):
        self.df[["file_name", "original_tags"]].to_csv(EDITED_CSV_PATH, index=False)

    def _show_image(self, path):
        img = Image.open(path); img.thumbnail(MAX_IMG)
        self.tk_img = ImageTk.PhotoImage(img)  # keep reference
        self.canvas.config(width=self.tk_img.width(), height=self.tk_img.height())
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def _update_static_panel(self):
        self.fname_var.set(self.row.file_name)
        self.added_var.set(join_space(self.adds)   or "—")
        self.removed_var.set(join_space(self.rms)  or "—")
        self.orig_var.set(join_space(self.orig)    or "—")

    def _update_final_preview(self):
        self.final_var.set(join_space(self.tags) or "—")

    def _after_tag_change(self):
        self._update_final_preview()
        self._update_prompt()

    def _update_prompt(self):
        if self.ptr < len(self.queue):
            mode, tag = self.queue[self.ptr]
            txt = f"{'Remove' if mode == 'REM' else 'Add'} tag “{tag}” ?"
        else:
            txt = f"Finished {self.row.file_name} – press N to continue or E to edit."
        self.prompt.config(text=txt)

    def _load_row(self):
        while True:
            self.row = self.df.iloc[self.idx]
            fname = clean_fname(self.row.file_name)
            if fname: self.img_path = IMAGE_DIR / fname; break
            self.idx += 1
            if self.idx >= len(self.df):
                self._persist_edits(); messagebox.showinfo("Done", "All images reviewed!"); self.root.quit(); return

        if not self.img_path.exists():
            messagebox.showwarning("Missing image", f"{self.img_path} not found – skipped.")
            self.idx += 1; self._load_row(); return

        self.adds  = split_tags(self.row.added)
        self.rms   = split_tags(self.row.removed)
        self.orig  = split_tags(self.row.original_tags)
        self.queue = [("ADD", t) for t in self.adds] + [("REM", t) for t in self.rms]
        self.tags  = set(self.orig)
        self.ptr   = 0
        self.log   = []

        self._show_image(self.img_path)
        self._update_static_panel()
        self._update_final_preview()
        self._update_prompt()

    # ---------- constructor ----------
    def __init__(self, master, df):
        self.root, self.df = master, df
        self.idx = self.reviewed = 0

        # layout
        outer = tk.Frame(master); outer.pack(padx=5, pady=5)
        self.canvas = tk.Canvas(outer, bd=0, highlightthickness=0); self.canvas.pack(side="left")

        panel = tk.Frame(outer, relief="groove", bd=2); panel.pack(side="left", padx=6)
        tk.Label(panel, text="File:", anchor="w").grid(row=0, column=0, sticky="w")
        self.fname_var = tk.StringVar(); tk.Label(panel, textvariable=self.fname_var,
                                                  wraplength=250, justify="left"
                                                 ).grid(row=0, column=1, sticky="w")

        def row(lbl, var, r):
            tk.Label(panel, text=lbl, anchor="nw").grid(row=r, column=0, sticky="nw", pady=2)
            tk.Label(panel, textvariable=var, wraplength=250, justify="left",
                     bg="#f0f0f0").grid(row=r, column=1, sticky="w", pady=2)

        self.added_var, self.removed_var, self.orig_var, self.final_var = (
            tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar())
        row("Added:",    self.added_var,   1)
        row("Removed:",  self.removed_var, 2)
        row("Original:", self.orig_var,    3)
        row("Final:",    self.final_var,   4)   # live preview

        self.prompt = tk.Label(master, font=("Arial", 14)); self.prompt.pack(pady=4)
        btns = tk.Frame(master); btns.pack(pady=4)
        for txt, fn in (
            ("Accept (A)", self.accept),
            ("Reject (R)", self.reject),
            ("Back (B)",   self.back),
            ("Edit (E)",   self.manual_edit),
            ("Next (N)",   self.next_image)):
            tk.Button(btns, text=txt, width=10, command=fn).pack(side="left", padx=2)

        for key, fn in {"a": self.accept, "r": self.reject, "b": self.back,
                        "e": self.manual_edit, "n": self.next_image}.items():
            master.bind(key, fn); master.bind(key.upper(), fn)

        self._load_row()


# --------------- main() -----------------
def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

    df = pd.read_csv(CSV_PATH, dtype={'file_name': str})
    required = {"file_name", "added", "removed", "original_tags"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {', '.join(missing)}")

    root = tk.Tk(); root.title("Danbooru Tag Review")
    TagReviewer(root, df)
    root.mainloop()
    df[["file_name", "original_tags"]].to_csv(EDITED_CSV_PATH, index=False)
    print("Edits saved to", EDITED_CSV_PATH)


if __name__ == "__main__":
    main()
