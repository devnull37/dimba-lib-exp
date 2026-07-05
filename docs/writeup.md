# What a $450 diffusion language model taught me about test-time compute

*by Faris Allafi, July 2026, draft*

I'm 14, self-funded, and for the past months I've been building **DIMBA**: a language model that uses a *bidirectional Mamba* backbone with a *diffusion* objective instead of the usual left-to-right transformer. This is the story of how the first version failed for $358, how a pivot to masked diffusion made it work, and how one long day of test-time-compute experiments produced the most interesting finding of the project:

> **At 135M-class scale, every inference-time technique that asks the model to judge itself fails, and every technique that imposes an external constraint works.** Self-judgment is the first casualty of small scale.

I measured that six different ways (287.9M measured parameters, 135M-class backbone capacity). Everything below is a real, unedited model output. All checkpoints and code are public, including the failures, because negative results are results.

## Part 1: The $358 failure (and why it was worth it)

The first DIMBA used **continuous-latent diffusion**: corrupt token embeddings with Gaussian noise, train the model to denoise. I trained it on 28 billion tokens distilled from SmolLM-135M. Losses looked healthy the whole run. The model was unusable: pure word salad.

The post-mortem found a real root cause. With the loss weighting I used, there was almost no gradient pressure at high noise levels, which is exactly where every generation starts. The model had learned to output a "posterior mean blend" (roughly: the average of all plausible texts, which is mush). I verified this with a partial-noise recovery ladder: the model was a near-perfect denoiser up to 90% noise and fell off a cliff at 100%. Repair finetuning, SFT, RL, and knowledge distillation all failed to fix it. Four clean negative results convinced me the problem was *inherent to continuous-latent diffusion at this scale*, not a bug.

(This matches the literature: D3PM (Austin et al., 2021) already found that for text, **absorbing/masking corruption beats Gaussian-style noise**. I paid $358 to independently confirm it. Don't be me. Read the ablation tables first.)

## Part 2: The pivot to masked discrete diffusion

So I changed the objective, not the architecture. Following the LLaDA/MDLM recipe: corrupt text by replacing a random fraction *t* of tokens with a `[MASK]` token, train the model to predict the originals with plain cross-entropy (1/t weighted), generate MaskGIT-style: start fully masked, iteratively commit the most confident tokens. The bidirectional Mamba backbone and its 28B-token knowledge carried over unchanged (one new embedding row for `[MASK]`).

**English appeared within 30 minutes of training.** After 40k steps (~$16):

> "Once upon a time there was a great moment. The world was the first the most important thing in the whole world..."

Grammar, punctuation, sentence structure, from a backbone that produced word salad the day before. Same weights, different objective.

One thing worth stressing: as far as I can tell, **every published masked-diffusion text model is a transformer** (LLaDA, MDLM, Dream, and the rest). The objective is proven; the Mamba backbone is the novel variable. That's the same move the big labs make: take what's proven, change one thing, add your own pieces on top.

## Part 3: Instruction tuning, including the bug

I fine-tuned on Alpaca (prompt kept clean, response masked and predicted). **Version 1 collapsed to empty answers**, and the lesson is worth sharing: I trained the loss on the EOS padding tail, so "predict end-of-answer everywhere" became the cheapest solution. The loss looked great (1.38!) while the model output nothing. Restricting the loss to the response plus exactly one EOS fixed it, and the model started answering *on topic*:

> Q: "Write one sentence about the ocean." →
> "The vast sea of the ocean is surrounded in the ocean of the ocean. The ocean is in the sea of sea in the sea, and the clouds of the sea"

Loopy, but ocean questions get ocean words and France questions get country words. Conditioning works.

## Part 4: Classifier-free guidance unlocks facts

The model's remaining failure looked like *lack of commitment*: it knew the topic but hedged into generic mush. The standard diffusion cure is classifier-free guidance (CFG): train with 10% of prompts dropped (fully masked), then at sampling time compute both the prompt-conditioned and unconditioned predictions and push the output *away* from unconditional: `logits = uncond + g * (cond - uncond)`.

Same model, same question, guidance off vs on:

> **guidance OFF:** " The name of the world is the world of the world of the world of the world..."
>
> **guidance ON:** " **The European city of Paris is the city of of Paris.** It is in the city city of the Paris. It is the city of Paris and Paris."

First factually correct answer in the project's history. You can watch the fact crystallize during training. Step 1500: "European Europe... Africa". Step 2000: "European Republic of Paris". Step 2500: "the city of Paris, France". Step 3000: "The European city of Paris." The knowledge was in the weights all along, from the original 28B-token run. Guidance is what makes the model commit to it.

## Part 5: Scaling the data, hitting the wall

Next I scaled SFT: 422k pairs (Alpaca + 350k SmolTalk + 20k math), 15k steps, with CFG dropout. Real wins:

- Richer, more natural sentence shapes from SmolTalk. The sky question got its first correct answer ("bright blue").
- Math data taught the model the *format* of equations, but not arithmetic. "What is 2+3?" produces equation-shaped output with wrong numbers. Format is learnable at 135M-class; computation is not.

And two brutally clean capacity results:

**More steps stopped helping.** Checkpoints from 10k and 15k steps are indistinguishable. The model is parameter-bound, not data-bound. I'd already shown fresh pretraining data changes nothing (identical loss, identical infill accuracy on 600k never-seen documents), and now more SFT changes nothing either.

**The model memorizes facts; it does not generalize them.** My favorite demo in the whole project:

> Q: "What is the capital of Japan?" →
> "The capital of Paris is Paris, Paris, France, France..."

It learned "capital → Paris" as a *drilled association*, not "capital of X → look up X". One fact fit in the weights; the schema didn't. That single output tells you more about what a 135M-class model can hold than any benchmark number.

## Part 6: Sampling is training you get for free

Two upgrades that cost nothing but inference:

- **More diffusion steps help.** 128 unmasking steps beat 64 clearly. Fewer tokens committed per step means fewer premature commitments. This is the diffusion-native version of "thinking longer," and it's a knob autoregressive models simply don't have.
- **An anti-repetition sampler.** Repetition is the signature failure at this scale. A frequency penalty that punishes *reuse* but exempts the first occurrence (penalize `count - 1`, not `count`), plus banning a token from appearing adjacent to itself. This kills "the world of the world of the world" without banning legitimate repeats like "Paris ... Paris."

Final recipe: 128 steps, temperature 0.7, top-k 20, guidance 2.0, exempt-first frequency penalty 0.7, neighbor ban.

## Part 7: One day of test-time compute. Six ways the model can't judge itself, one way something else can

This was the big experiment day (~$6 total). The question: can we spend *inference* compute to get *training-level* quality gains? The 2025-26 literature is full of techniques for this (ReMDM, RemeDi, Token-Critic, remasking policies, and more). All of them assume the model can tell its good tokens from its bad ones. Can a 135M model?

**No. Measured six ways:**

1. **Perplexity reranking** (earlier): best-of-8 by perplexity picks "water water water", because repetition is easy to predict, so degenerate text wins.
2. **Confidence-based re-masking + argmax refill:** re-mask the least-confident committed tokens and refill greedily → output collapses toward the topic word ("ocean ocean sea ocean").
3. **Same but sampled refill:** facts drift instead ("Paris" becomes "London"). Confidence can't distinguish *wrong* from *specific*: a rare correct fact and a hallucination look identical from the inside.
4. **Repair training v2 (echo chamber, my favorite negative result):** After v1 repair training worked (below), I tried training on the model's *own sampled mistakes* instead of random corruptions, since those are more realistic errors, right? The error-detection ability **collapsed from 7.1% to 0.0%** and I killed the run mid-flight. The model can't learn to catch its own errors *because it doesn't consider them errors*: its own samples are, by definition, what it finds plausible. This is a tiny, cheap demonstration of why frontier labs filter self-generated training data so aggressively.
5. **RCR, falling-confidence remasking** (from the 2026 remasking literature): re-mask a committed token when its probability *drops* as surrounding context fills in. Implemented, instrumented, verified working. It fired **zero times on 4 of 5 test prompts**, even with thresholds so sensitive they'd trigger on a 15% dip. Once this model commits a wrong token, its confidence in it only *grows*: commit "China is located in London," and the surrounding salad makes "China" look *more* right. The model rationalizes whatever it committed.
6. **Lookahead verification** (re-check each just-committed token with one extra forward pass): same story. Zero fires except on degenerate loops, where intervening reshuffled the loop without fixing it.

**What did work: every technique with an *external* signal.**

- **CFG** (contrast against the unprompted model) unlocked facts.
- **The anti-repeat sampler** (an imposed rule) killed loops.
- **Repair training v1** (random corruptions with labels): planted-error detection went from 0% → 7.1% while keeping 99.2% of clean tokens. Small but real, for $1.20. Random corruptions taught what self-generated ones couldn't.
- **A best-of-N verifier that doesn't trust the model's self-report:** score each candidate by the *guidance gap*, meaning how much more likely the answer is with the prompt visible vs masked, evaluated with the scored tokens masked out (leave-k-out; if the tokens are visible, the model just copies them and the signal vanishes). Unlike perplexity, this never rewards degenerate repetition.
- **⭐ The token-critic head, the day's best result.** A tiny 300k-parameter MLP trained on top of the *frozen* model's hidden states to answer one question per token: "is this token wrong?" Trained for $2.50 on planted errors (half random, half the model's own samples as hard negatives, which is safe here because the critic only *judges*, it never generates, so it can't echo-chamber). Held-out results: **finds planted errors with 52.5% precision (chance: 10%) and ranks a wrong token above a correct one 78.9% of the time (chance: 50%).**

The critic head is the counterexample that completes the thesis. The error signal *exists in the model's hidden states*: a 300k-parameter probe can read it. The model just cannot act on it through its own confidence. Self-knowledge isn't missing from the weights; it's missing from the *interface*.

### So is model + critic better than the model alone?

The honest scoreboard, because I wired it in and measured rather than assumed:

- **As an error detector: yes, decisively.** The critic finds wrong tokens at 5× chance while the model's own confidence performs at roughly chance on the identical test. If your task is "flag the suspect tokens in this output," model + critic beats model, full stop.
- **As a candidate ranker (best-of-8): mildly.** The critic reliably ranks the most degenerate candidates last, something perplexity actively gets backwards. But at 135M all eight candidates are word salad, so picking the best salad moves end accuracy very little.
- **As a self-correction loop: not yet, and the failure is instructive.** Wiring the critic into decode-time correction (critic picks which tokens to re-mask, model refills) lowers the critic's wrongness score on every single prompt, but factual accuracy doesn't improve, because the *refill* is still done by the 135M model, which swaps flagged tokens for safe filler the critic tolerates. Goodhart's law, live.

So: **detection is now solved externally; correction needs a smarter generator.** That's a precise, testable prediction for 350M: same critic recipe, stronger refill model, and the loop should close.

## Part 8: A dial nobody else has, and an optimizer result

Two forward-looking pieces from the same day:

**The accuracy↔cost dial.** Diffusion gives you *two* orthogonal inference knobs: number of denoising steps, and best-of-N with the verifier. Both are per-request. That means one deployed model can serve "fast and rough" and "slow and careful" from the same weights, which is a genuine architectural advantage over autoregressive models, where thinking longer had to be trained in. (Next step: let the model set its own dial per request.)

**Muon works on Mamba diffusion.** I ran a controlled A/B (2,000 identical steps, same data, same seed) of Muon, the orthogonalized-momentum optimizer used in Kimi K2, against AdamW, with the standard hybrid split (243 weight matrices on Muon, embeddings and the rest on AdamW). Muon led at nearly every checkpoint and finished at 5.453 vs 5.470 loss. A small edge, but this was Muon's *worst-case* setting (short continuation of a converged model; its published 1.5-2× wins come from long from-scratch runs), and it was perfectly stable. As far as I can find, this is the **first Muon result on a Mamba-backbone diffusion LM.** Decision made: 350M trains with Muon.

## The thesis, and the experiment I actually want to run

Put Part 7 together and you get the sentence I'd defend in front of any researcher:

> **At small scale, inference-time quality must come from external constraints, because self-judgment is the first casualty of small scale.**

And it sets up a measurement nobody has published for diffusion LMs: **where does self-correction turn on?** I now have three cheap, repeatable probes: planted-error detection rate (7.1% at 135M), remasking fire-rate (~0 at 135M), and critic-head AUC (78.9% at 135M). Run the identical probes at 350M and 1B and you get a *calibration scaling curve*: the parameter count at which a diffusion model becomes able to fix its own mistakes. That curve is the scientific payload of the next two runs.

## What this cost

| Phase | Cost |
|---|---|
| Original 28B-token continuous run (negative result + the backbone) | $358 |
| Root-cause + repair attempts (4 clean negatives) | ~$31 |
| MDM pivot: 2 pretrains, 3 SFTs, sampler sweep, CFG | ~$37 |
| Big SFT (422k pairs, 15k steps) + repair training + test-time-compute day (repair v1+v2, RCR/lookahead, verifier, critic head, Muon A/B) | ~$25 |
| **Total** | **~$450** |

## Roadmap

- **350M from scratch, ~5B tokens, Muon, ≈ $200.** Measure the calibration curve's second point. Retrain the critic head jointly. Test whether critic-guided correction closes once the refill model is smarter.
- **Planning-latent tokens** (my idea I'm most excited about): compress a continuous "plan" vector and condition the discrete diffusion on it. Latent space where it helps (global planning), discrete tokens where diffusion is proven (the text itself). Saved for 350M/1B where there's capacity to use a plan.
- **Adaptive test-time compute:** the accuracy↔cost dial, eventually self-set per request.
- **1B, 10-20B tokens, ≈ $1,000-1,600:** where LLaDA says this recipe becomes competitive with same-size autoregressive models. MoE and step-distillation live here.
- Parked until then: Dream-style noise rescheduling, VRPO, variance-reduced masked training, jointly-trained token critic.

Release model (private for now): **huggingface.co/devnull37/hr-diffuse-1-nano**. Development archive with every checkpoint including the failures: **huggingface.co/devnull37/d1-135m-28b**.

*References: LLaDA (Nie et al.), MDLM (Sahoo et al.), MaskGIT (Chang et al.), D3PM (Austin et al.), Muon/Moonlight (Kimi), Token-Critic (Lezama et al.), ReMDM / RemeDi / remasking-policy literature (2025-26).*

*If you're a researcher and any of this looks wrong or interesting, I'd genuinely love to hear it.*
