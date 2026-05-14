---
title: "CTF Writeup — Reverse Engineering Mario Bros"
date: 2026-05-14
categories: ["writeups"]
tags: ["reverse-engineering", "ctf", "ida", "cyberchef", "elf", "xor"]
summary: "Static reverse engineering of a Linux ELF challenge that hides its flag behind an unreachable in-game vault — extracted in minutes without ever launching the binary."
ShowToc: true
TocOpen: true
---

> **Author:** Radu-Petruț Vătase — [TCSI](https://tcsi.ro/) | May 2026

**Keywords:** reverse engineering, ELF, static analysis, IDA, XOR, CTF

**Platform:** [cyber-edu.co](https://app.cyber-edu.co/) — Reverse Engineering category

---

## Abstract

This writeup covers a beginner-friendly reverse engineering challenge whose intended dynamic path requires collecting an absurd number of in-game items to unlock a vault. By analyzing the binary statically, the same data can be recovered in a few minutes without playing the game at all. The post documents the workflow end-to-end: file identification, symbol triage, cross-referencing in IDA, decompilation, and final decryption.

---

## 1. Challenge Overview

The challenge ships a single Linux executable and a short prompt:

> *Mario has a secret in his vault and you need to recover it.*

![Challenge description](/images/ctf-writeup-reverse-engineering-mario-bros/01-challenge.png)
*Figure 1: Challenge prompt on the platform.*

The obvious interpretation is that the player must launch the game and trigger a specific in-game event to reveal the flag. Whenever a challenge implies "play the game," it is worth checking whether the gating logic is **enforced cryptographically** or only **enforced by control flow**. In the latter case, static analysis bypasses the gate entirely.

---

## 2. File Identification

The first step in any RE workflow is establishing what type of artifact is under analysis. **Detect It Easy (DIE)** reports the following:

![DIE output](/images/ctf-writeup-reverse-engineering-mario-bros/02-die.png)
*Figure 2: Detect It Easy classification of the binary — ELF64, AMD64, GCC, dynamically linked, with SDL.*

Two properties stand out:

- **ELF64, dynamically linked** — a Linux executable. Static analysis is platform-independent; running the binary is not required.
- **Not stripped** — symbol information is preserved. For a C++ binary this exposes class names, namespaces, and member functions, which dramatically reduces the search space.

The presence of **SDL** is consistent with the challenge being a playable game, but irrelevant to the static path.

---

## 3. String Triage

A targeted pass over the binary's printable strings narrows the focus immediately:

```bash
strings super-mario-bros | grep "vault\|EncryptedFlag"
```

![Strings grep output](/images/ctf-writeup-reverse-engineering-mario-bros/03-strings.png)
*Figure 3: Filtering the string table for vault- and flag-related identifiers.*

Two results are relevant:

- A developer-style note hinting at the in-game condition required to open the vault (the **dynamic path**).
- A C++ mangled symbol containing `kEncryptedFlag` (the **static path**). In an unstripped binary, a symbol with this name is effectively an annotation pointing at the solution.

The remainder of this writeup focuses on the static path.

---

## 4. Locating the Symbol in IDA

The binary is loaded into IDA Free, with auto-analysis allowed to complete. Opening the **Names** window (`Shift+F4`) and filtering for `EncryptedFlag` returns a single match in the `.rodata` segment at address `0xD3800`:

![IDA Names window](/images/ctf-writeup-reverse-engineering-mario-bros/04-ida-names.png)
*Figure 4: The `kEncryptedFlag` symbol located in `.rodata` at `0xD3800`.*

Double-clicking the entry jumps to the data definition. The 69-byte buffer contains a mix of hex characters and a handful of non-hex symbols, indicating an encoded — not plaintext — payload.

Pressing `X` (cross-references) on the symbol returns **one** consumer: `ScoreSystem::getFlag()`. This is the function that reads the encoded buffer at runtime.

---

## 5. Static Analysis of the Consumer

### 5.1 Graph View

Switching to graph view (`Space`) on `ScoreSystem::getFlag` reveals a single tight loop with no conditional branching beyond `std::string` capacity bookkeeping. The function reserves 69 bytes (`mov esi, 45h`) and loads the address of `kEncryptedFlag` into `r12`:

![IDA graph view](/images/ctf-writeup-reverse-engineering-mario-bros/05-graph.png)
*Figure 5: Function prologue of `ScoreSystem::getFlag` — reserves 69 bytes and points at `kEncryptedFlag`.*

The shape alone strongly suggests a stream transformation: read byte, transform, append, advance, repeat.

### 5.2 Decompiler Output

Invoking the decompiler (`F5`) collapses the function to its essential semantics:

![IDA pseudocode](/images/ctf-writeup-reverse-engineering-mario-bros/06-pseudo.png)
*Figure 6: Decompiled pseudocode. The transformation is a single XOR with the constant `85` (decimal).*

Stripped of allocator boilerplate, the loop reduces to:

```c
for (p = kEncryptedFlag; p < kEncryptedFlag + 0x45; p++)
    result.push_back(*p ^ 85);
```

Three values describe the entire encryption scheme:

- **Length:** `69` bytes (`0x45`)
- **Key:** `85` decimal (`0x55`)
- **Operation:** single-byte XOR

A separate function in the same class (`challengeSolved`) enforces the in-game gate (visible-coin threshold and a vault-state flag). Critically, the gate guards **when** the consumer is called during gameplay, but the consumer reads the constant directly from `.rodata` — there is no key derivation that depends on game state. The static path is therefore equivalent to the dynamic one.

---

## 6. Extracting the Encoded Payload

With the consumer's behavior understood, the 69 bytes at the symbol's address can be exported directly from IDA. Select the range (`0xD3800` → `0xD3844`), press `Shift+E`, and choose **hex string (unspaced)**:

![IDA Shift+E export](/images/ctf-writeup-reverse-engineering-mario-bros/07-export.png)
*Figure 7: Exporting the encoded payload as an unspaced hex string.*

The exported blob is a 138-character hex string representing the 69 encoded bytes.

---

## 7. Decryption

The final step is reproducing the loop from §5.2 on the exported bytes. **CyberChef** is the fastest option for documentation purposes:

1. `From Hex`
2. `XOR` — Key: `85`, Key format: **Decimal**, Scheme: Standard

![CyberChef recipe](/images/ctf-writeup-reverse-engineering-mario-bros/08-cyberchef.png)
*Figure 8: Two-operation CyberChef recipe yielding the plaintext flag.*

The output is a flag of the form:

```
HTA{ef44**********************************************************fa7}
```

*(Hex body redacted intentionally; reproduce the recipe to obtain the full value.)*

---

## 8. Conclusions

The challenge illustrates a common pattern in beginner RE tasks: a flashy dynamic gate (collect N items, reach a specific game state) layered on top of a trivially reversible static encoding. The gate exists to deter casual play-throughs, not to resist analysis. Recognizing this distinction early saves time.

Key observations:

- **Unstripped C++ binaries leak their structure.** Symbol names like `kEncryptedFlag` and `ScoreSystem::challengeSolved` map the entire solution before any disassembly begins.
- **Naming is half the challenge.** A constant named `kEncryptedFlag` is a directional signal; following its single cross-reference led directly to the decryption routine.
- **Dynamic gates are not cryptographic gates.** When the gated function reads a fixed constant rather than deriving a key from game state, the gate is bypassable by inspection.
- **A minimal toolchain suffices.** File classification → string triage → disassembler with decompiler → CyberChef covers the majority of entry-level reverse engineering tasks.

---

## References

1. cyber-edu.co — Reverse Engineering challenge: *I am Mario*. Available at: [https://app.cyber-edu.co/](https://app.cyber-edu.co/)
2. Hex-Rays. *IDA Free*. Available at: [https://hex-rays.com/ida-free](https://hex-rays.com/ida-free)
3. Horsicq. *Detect It Easy*. Available at: [https://github.com/horsicq/Detect-It-Easy](https://github.com/horsicq/Detect-It-Easy)
4. GCHQ. *CyberChef*. Available at: [https://gchq.github.io/CyberChef/](https://gchq.github.io/CyberChef/)
