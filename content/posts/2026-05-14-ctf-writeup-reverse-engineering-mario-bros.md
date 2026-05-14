---
title: "CTF Writeup — Reverse Engineering Mario Bros"
date: 2026-05-14
categories: ["writeups"]
tags: ["reverse-engineering", "ctf", "ida", "cyberchef", "elf", "xor"]
summary: "A Mario-themed RE challenge wants you to grind 6007 coins to unlock a vault. We never launched the game."
ShowToc: true
TocOpen: true
---

> **Author:** Radu-Petruț Vătase — [TCSI](https://tcsi.ro/) | May 2026

This one had a fun premise: play through a Mario clone, hoard **6007 coins**, unlock a vault, get the flag. The challenge is rated *easy* on [cyber-edu.co](https://app.cyber-edu.co/), and it's a nice excuse to walk through a clean static-analysis workflow on a Linux ELF — without ever launching the binary.

If you're new to RE, this is exactly the kind of warm-up that teaches the right reflex: **before you grind, ask whether the grind is real**.

---

## 1. The Challenge

A single Linux executable and a one-liner:

> *Mario has a secret in his vault and you need to recover it.*

![Challenge description](/images/ctf-writeup-reverse-engineering-mario-bros/01-challenge.png)
*Figure 1: The challenge page on cyber-edu.co.*

The binary is a fan-made C++/SDL2 recreation of the 1985 classic:

![Game title screen](/images/ctf-writeup-reverse-engineering-mario-bros/02-game.png)
*Figure 2: Title screen of the recreated game. The line at the bottom — "some values are just for show" — is the author practically winking at you.*

Two immediate thoughts:

1. Collecting 6007 coins by hand sounds awful.
2. The flag almost certainly lives in the binary already, gated by some `if (coins >= 6007)` check. If it does, that gate is bypassable by analysis.

So instead of playing, let's open it up.

---

## 2. What Are We Dealing With?

First thing into **Detect It Easy** to confirm the file type:

![DIE output](/images/ctf-writeup-reverse-engineering-mario-bros/03-die.png)
*Figure 3: ELF64, AMD64, GCC, dynamically linked, with SDL — and crucially, **not stripped**.*

Two details matter here:

- It's an **ELF64**, but we're on Windows. That's fine — for static analysis the host OS is irrelevant.
- The binary is **not stripped**, meaning all the C++ class and function names are still in the symbol table. In an unstripped C++ game, that's a goldmine.

---

## 3. Strings, Targeted

A blind `strings` dump on a 1.2 MB binary is noise. Let's grep for what we actually care about — anything mentioning the vault, or anything obvious like an encrypted flag:

```bash
strings super-mario-bros | grep "vault\|EncryptedFlag"
```

![Strings grep output](/images/ctf-writeup-reverse-engineering-mario-bros/04-strings.png)
*Figure 4: Two hits. One is bait, the other is the entire solution.*

We get exactly two hits:

```
DEV NOTE: Collect at least 6007 visible coins to open the castle vault.
_ZN12_GLOBAL__N_1L14kEncryptedFlagE
```

The first is the "intended" path — collect coins, unlock vault. The second is a C++ mangled symbol named **`kEncryptedFlag`**, sitting somewhere in the binary's data. Whoever wrote this challenge named their secret constant `kEncryptedFlag`. We're done thinking; time to look at it directly.

---

## 4. Following the Symbol in IDA

Loaded the binary into **IDA Free**, let auto-analysis finish, then opened the Names window with `Shift+F4` and filtered for `EncryptedFlag`:

![IDA Names window](/images/ctf-writeup-reverse-engineering-mario-bros/05-ida-names.png)
*Figure 5: Single match at `.rodata:0xD3800`.*

One result. Double-clicking jumps to the data — 69 bytes of mostly-hex characters with a few outliers (`` ` ``, `g`, `l`, `m`) sprinkled in. Definitely encoded, not plaintext.

Now the question is **who reads these bytes**. Press `X` on the symbol to list cross-references, and there's exactly one: a method called `ScoreSystem::getFlag()`. That's our decryption routine.

---

## 5. Reading the Decryption

### 5.1 The Loop in Assembly

Jumping into `ScoreSystem::getFlag` and switching to graph view (`Space`) shows a function with a single loop and almost no branching. The prologue allocates a 69-byte `std::string` buffer (`mov esi, 45h`) and loads the address of `kEncryptedFlag` into `r12`:

![IDA graph view](/images/ctf-writeup-reverse-engineering-mario-bros/06-graph.png)
*Figure 6: Prologue of `ScoreSystem::getFlag`. Reserves 69 bytes, points at `kEncryptedFlag`, and falls into a loop body.*

You could read the whole loop in asm if you wanted. But this is what `F5` is for.

### 5.2 The Loop in Pseudocode

Hitting `F5` collapses the function to something a human can read at a glance:

![IDA pseudocode](/images/ctf-writeup-reverse-engineering-mario-bros/07-pseudo.png)
*Figure 7: The decompiled function. Ignore the `std::string` plumbing — the payload is the highlighted line.*

The decompiler's output looks busy because of all the `std::string` reallocation logic, but everything important fits on one line:

```c
*(_BYTE *)(*a1 + v3) = v4 ^ 85;
```

Strip the allocator boilerplate and the function is:

```c
for (p = kEncryptedFlag; p < kEncryptedFlag + 0x45; p++)
    result.push_back(*p ^ 85);
```

That's it. The "encryption" is:

| Parameter | Value |
|---|---|
| Length | `69` bytes (`0x45`) |
| Key | `85` decimal (`0x55`) |
| Operation | Single-byte XOR |

### 5.3 About the 6007-Coin Gate

Worth pausing here. There's a separate method, `ScoreSystem::challengeSolved()`, that does check `visibleCoins > 0x1776` and a vault-state flag — that's the in-game gate the dev note advertises. But `getFlag()` reads the bytes directly from `.rodata`. There's no key derivation, no salt from game state, nothing dynamic. The coin count gates **when** the game calls `getFlag()`, not **what** it returns.

The dev note isn't a red herring exactly — it's a real path. It's just one of two paths, and the static one is faster by several orders of magnitude.

---

## 6. Grabbing the Bytes

Back to the data view at `0xD3800`. Selected 69 bytes (start, then jump to `0xD3844` and shift-click), hit `Shift+E`, and picked **hex string (unspaced)**:

![IDA Shift+E export](/images/ctf-writeup-reverse-engineering-mario-bros/08-export.png)
*Figure 8: The exported blob — 138 hex chars = our 69 encoded bytes.*

---

## 7. Decrypting in CyberChef

You can decrypt in three lines of Python, but if I'm writing this up I might as well use **CyberChef** so anyone reading can drop the recipe into a browser tab and follow along:

1. `From Hex`
2. `XOR` — Key `85`, Key format **Decimal**, Scheme `Standard`

![CyberChef recipe](/images/ctf-writeup-reverse-engineering-mario-bros/09-cyberchef.png)
*Figure 9: Two operations. Output panel populates instantly.*

The flag comes out in the format:

```
HTA{ef44**********************************************************fa7}
```

*(Middle deliberately redacted so the writeup doesn't become a copy-paste shortcut — reproduce the recipe and you'll get the full value in a few seconds.)*

---

## 8. Why I Like This One

It's a beginner challenge, but the lesson generalizes. CTF authors lean on "do the hard thing in the game" as a stand-in for actual reverse-engineering difficulty. The trick is recognizing the difference between:

- **A gate enforced by cryptography** — where the flag is genuinely derived from runtime state, and you can't get it without that state.
- **A gate enforced by control flow** — where the flag is just sitting in the binary, and the gate is one `if`-statement away from being irrelevant.

Almost every "play the game" gate in a CTF is the second kind. The five-minute test is: find where the flag lives, check whether the function that reads it depends on game state. If it doesn't, you're done.

A few habits this challenge reinforced:

- Run **DIE** before anything else. Knowing the binary type and whether it's stripped tells you which workflow to use.
- `strings | grep` with **specific keywords from the prompt** is dramatically more useful than scrolling through everything.
- In unstripped C++ binaries, **constant names are part of the attack surface**. `kEncryptedFlag` was the whole challenge.
- The IDA decompiler (`F5`) turns "reading a 30-instruction loop" into "reading one line of C." Always start there; drop to asm only when the pseudocode looks wrong.
- One xref + one decompile + one CyberChef recipe is a complete static-analysis pipeline for a huge chunk of beginner/easy RE.

---

**Tools used:** [Detect It Easy](https://github.com/horsicq/Detect-It-Easy), [IDA Free](https://hex-rays.com/ida-free), [CyberChef](https://gchq.github.io/CyberChef/), `strings` from binutils.

**Challenge:** *I am Mario* on [cyber-edu.co](https://app.cyber-edu.co/).
