---
title: "CTF Writeup — Reverse Engineering Mario Bros"
date: 2026-05-14
categories: ["writeups"]
tags: ["reverse-engineering", "ctf", "ida", "cyberchef", "elf", "xor"]
summary: "Static recovery of an XOR-encoded flag from a Linux ELF game binary, bypassing the in-game vault gate."
ShowToc: true
TocOpen: true
---

> **Author:** Radu-Petruț Vătase — [TCSI](https://tcsi.ro/) | May 2026

**Platform:** [cyber-edu.co](https://app.cyber-edu.co/) — Reverse Engineering
**Challenge:** *I am Mario*

---

## Description

> *Mario has a secret in his vault and you need to recover it.*

![Challenge description](/images/ctf-writeup-reverse-engineering-mario-bros/01-challenge.png)
*Figure 1: Challenge page.*

The provided binary is a C++/SDL2 Mario-style platformer. An in-binary note suggests the flag is unlocked by collecting 6007 coins. This writeup skips the gameplay and recovers the flag through static analysis.

![Game title screen](/images/ctf-writeup-reverse-engineering-mario-bros/02-game.png)
*Figure 2: Title screen of the provided binary.*

---

## 1. File Triage

```
file super-mario-bros
```

ELF 64-bit LSB pie executable, x86-64, dynamically linked, **not stripped**. **Detect It Easy** confirms:

![DIE output](/images/ctf-writeup-reverse-engineering-mario-bros/03-die.png)
*Figure 3: ELF64, AMD64, GCC, with SDL.*

The fact that the binary is unstripped is the relevant detail — all C++ class and function names are preserved in the symbol table.

---

## 2. String Triage

```
strings super-mario-bros | grep "vault\|EncryptedFlag"
```

![Strings output](/images/ctf-writeup-reverse-engineering-mario-bros/04-strings.png)
*Figure 4: Filtered output.*

Two hits:

- `DEV NOTE: Collect at least 6007 visible coins to open the castle vault.`
- `_ZN12_GLOBAL__N_1L14kEncryptedFlagE`

The second is the mangled name of a constant `kEncryptedFlag` in an anonymous namespace.

---

## 3. Locating the Symbol in IDA

Loaded in **IDA Free**. After auto-analysis, `Shift+F4` opens the Names window. Filtering for `EncryptedFlag`:

![IDA Names window](/images/ctf-writeup-reverse-engineering-mario-bros/05-ida-names.png)
*Figure 5: One match at `.rodata:0xD3800`.*

The symbol is 69 bytes of encoded data. Pressing `X` on it lists cross-references — exactly one: `ScoreSystem::getFlag()`.

---

## 4. The Decryption Routine

### 4.1 Disassembly

`ScoreSystem::getFlag` prologue:

![IDA graph view](/images/ctf-writeup-reverse-engineering-mario-bros/06-graph.png)
*Figure 6: Prologue. Reserves 69 bytes (`mov esi, 45h`), loads `&kEncryptedFlag` into `r12`.*

### 4.2 Pseudocode

`F5` decompiles the function:

![IDA pseudocode](/images/ctf-writeup-reverse-engineering-mario-bros/07-pseudo.png)
*Figure 7: Decompiler output. The transformation is `v4 ^ 85`.*

Reduced to its essential semantics:

```c
for (p = kEncryptedFlag; p < kEncryptedFlag + 0x45; p++)
    result.push_back(*p ^ 85);
```

| Parameter | Value |
|---|---|
| Length | 69 bytes (`0x45`) |
| Key | `0x55` (85 decimal) |
| Operation | Single-byte XOR |

### 4.3 The Vault Gate

`ScoreSystem::challengeSolved()` enforces `visibleCoins > 0x1776 && vaultState == 1` — the in-game gate referenced by the dev note. However, `getFlag()` reads `kEncryptedFlag` directly from `.rodata` with no key derived from game state. The gate controls *when* the function is invoked during gameplay; it does not affect *what* the function returns.

---

## 5. Extracting the Encoded Bytes

In the data view at `0xD3800`, select 69 bytes (`0xD3800`–`0xD3844`), then `Shift+E` → **hex string (unspaced)**:

![IDA Shift+E export](/images/ctf-writeup-reverse-engineering-mario-bros/08-export.png)
*Figure 8: Export dialog showing the 138-character hex string.*

---

## 6. Decrypting

**CyberChef** recipe:

1. `From Hex`
2. `XOR` — Key `85`, Key format **Decimal**, Scheme `Standard`

![CyberChef recipe](/images/ctf-writeup-reverse-engineering-mario-bros/09-cyberchef.png)
*Figure 9: Output panel contains the flag.*

Flag format:

```
HTA{ef44**********************************************************fa7}
```

*(Hex body redacted.)*

---

## Notes

- Always check whether a binary is stripped before deciding on a workflow. An unstripped C++ binary with a constant named `kEncryptedFlag` is a single-xref problem.
- "Play the game" gates in CTFs are rarely cryptographic. If the gated function reads a fixed constant rather than deriving a key from runtime state, the gate is irrelevant to static analysis.

**Tools:** [Detect It Easy](https://github.com/horsicq/Detect-It-Easy) · [IDA Free](https://hex-rays.com/ida-free) · [CyberChef](https://gchq.github.io/CyberChef/) · `strings` (binutils)
