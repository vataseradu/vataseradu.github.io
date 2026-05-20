---
title: "CTF Writeup — Crackme: BobGambling"
date: 2026-05-21
categories: ["writeups"]
tags: ["reverse-engineering", "ctf", "crackme", "ida", "die", "windows", "debugger"]
summary: "Bypassing a negative-input validation in a Windows crackme by flipping the SF flag at a jns instruction in the IDA debugger."
ShowToc: true
TocOpen: true
---

> **Author:** Radu Vătase | May 2026

**Platform:** [crackmes.one](https://crackmes.one/crackme/69b9accff2d49d8512f64a8f)
**Challenge:** *BobGambling*

---

## Description

A small Windows crackme that exposes a gambling-themed menu. One option leads to an admin terminal gated by an input check that rejects negative values. The goal is to reach the admin terminal and retrieve the flag. The strings of the binary already leak the flag, but the intended path is to defeat the input check at runtime.

![Challenge page on crackmes.one](/images/ctf-writeup-crackme-bobgambling/poza1_ctfcrackmebobgam.png)
*Figure 1: Challenge page on crackmes.one.*

---

## 1. Initial Recon

Running the binary brings up the menu. Picking the option that should lead to the admin path and entering a negative value yields the guard message `negative values are not allowed`.

![Application menu](/images/ctf-writeup-crackme-bobgambling/poza2_ctfcrackmebobgam.png)
*Figure 2: Application menu — "negative values are not allowed".*

That message is our hook into the code path that gates the admin terminal. Two reasonable approaches from here:

- **Static**: open the binary in DIE / IDA, find the guard, patch it.
- **Dynamic**: attach a debugger, break at the guard, and flip the comparison's outcome on the fly.

We take the dynamic route — it is faster and more illustrative.

---

## 2. File Triage with DIE

Opening the binary in **Detect It Easy** confirms a standard Windows PE and a clean compiler signature. More usefully, the *Strings* view shows the flag already embedded in the binary.

![DIE output showing flag in strings](/images/ctf-writeup-crackme-bobgambling/poza3_ctfcrackmebobgam.png)
*Figure 3: DIE detects the binary and exposes the flag in strings.*

The flag is right there — we could stop now. But the cleaner solution (and the one worth writing up) is to actually unlock the admin terminal so the binary itself prints the flag. That means defeating the *negative values are not allowed* check.

---

## 3. Locating the Validation Check

Load the binary in **IDA Free** and search for the cross-reference to the *"negative values are not allowed"* string. The handler around it follows the textbook signed-compare-and-branch pattern:

```nasm
; pseudo-shape of the guard
cmp     <input>, 0
jns     short ok            ; jump if SF == 0  (input >= 0)
; error path: print "negative values are not allowed"
...
ok:
; admin terminal path
...
```

`jns` ("jump if not sign") branches only when SF (the sign flag) is `0`, i.e. when the previous arithmetic on the user input did **not** set the high bit. A negative input flips SF to `1`, so `jns` falls through into the error path.

Set a breakpoint right on the `jns` instruction, start the binary under IDA's built-in debugger, choose the admin option, and submit `-1`.

![Breakpoint set at jns; debugger paused after input -1](/images/ctf-writeup-crackme-bobgambling/poza4_ctfcrackmebobgam.png)
*Figure 4: Breakpoint set at the jns guard; debugger paused after input -1.*

When execution halts at the breakpoint, the registers pane confirms `SF = 1` — exactly what the negative input produced.

---

## 4. Flipping the Sign Flag

`jns` only consults a single bit: SF. We do not need to rewrite the input, change the comparison, or patch the binary — we just toggle SF in the register pane from `1` to `0` while paused at the instruction. When we hit `F9` to continue, the CPU re-evaluates `jns` with SF=0 and takes the branch into the admin code path.

![SF flag toggled from 1 to 0 in IDA's register pane](/images/ctf-writeup-crackme-bobgambling/poza5_ctfcrackmebobgam.png)
*Figure 5: SF flag toggled from 1 to 0 in IDA's register pane, defeating the jns guard.*

---

## 5. Admin Terminal

With the guard bypassed, the binary follows the admin code path, prints the admin menu, and prints the flag.

![Admin terminal unlocked; flag printed in the menu](/images/ctf-writeup-crackme-bobgambling/poza6_ctfcrackmebobgam.png)
*Figure 6: Admin terminal unlocked; flag printed in the menu.*

Flag:

```
FLAG{dzctf(bob_is_free_1337)}
```

---

## Notes

- `jns` reads only `SF`. There are several equally valid ways to bypass it:
  - **Flip SF in the debugger** (what we did) — zero changes to the file, easy to demonstrate.
  - **Patch `jns` → `jmp`** statically — always takes the branch regardless of input.
  - **Patch `jns` → `js`** — inverts the check; negative inputs are now accepted, positive ones are rejected.
- The same idea generalises to any conditional jump driven by a single flag (`jz`, `jc`, `jo`, ...): when paused on the instruction, flipping the relevant bit lets you decide which branch the CPU takes without modifying code or data.
- Always check `strings` early. In this challenge the flag was sitting in plain sight; the debugger path was the *intended* solution, not the only one.

---

[Detect It Easy](https://github.com/horsicq/Detect-It-Easy) · [IDA Free](https://hex-rays.com/ida-free/)
