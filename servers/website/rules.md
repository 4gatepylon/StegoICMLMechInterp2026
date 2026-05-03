---
layout: page
title: Rules
permalink: /rules/
---

## Rules

1. **Goal.** Discover the encoding scheme used to embed secret bitstrings into Python source files. Demonstrate your solution by correctly predicting the bitstring for a held-out test program.

2. **Scope.** The encoding is purely syntactic — it depends only on the source text as parsed by Python's `ast` module. Runtime behavior, comments, string contents, and whitespace are not part of the encoding.

3. **Verification.** Use the [verification API](../submit) to check whether a given program encodes a given bitstring. The API returns only `true` or `false`.

4. **Rate limits.** The verification API enforces rate limits. Brute-forcing will not work — you need to understand the scheme.

5. **Winning.** The first participant to submit a correct description of the full encoding scheme (not just a single bitstring) wins the grand prize. Partial progress prizes may be awarded at the organizers' discretion.

6. **Fair play.** Do not attack the infrastructure. Do not interfere with other participants. Social engineering the organizers does not count as a valid solution.

7. **Eligibility.** Open to everyone. You may use any tools, including AI assistants. Teams are allowed.
