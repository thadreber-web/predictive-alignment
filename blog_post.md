# I Taught a Neural Network 50 Tasks in a Row and It Didn't Forget Any of Them

*An experimental investigation of predictive alignment networks, from trajectory memorization to zero catastrophic forgetting*

---

I spent the last few days running experiments on a brain-inspired learning rule called Predictive Alignment, published by Asabuki & Clopath in Nature Communications earlier this year. I wanted to understand what it could and couldn't do. Along the way, I found something the original authors never tested: these networks show zero catastrophic forgetting when learning tasks sequentially. Old tasks actually *improve* while the network learns new ones.

This post covers the full journey — the failures that taught me what Predictive Alignment is (and isn't), and the discovery that followed.

## What is Predictive Alignment?

Most neural networks learn through backpropagation — computing error at the output, then sending correction signals backward through every layer. It works, but it's biologically implausible. Real neurons don't have a mechanism to send error signals backward through synapses.

Predictive Alignment is a *local* learning rule. Each synapse updates itself using only information available at that synapse — no backward error propagation required. The key idea comes from the paper's title: "Taming the chaos gently."

The architecture has two weight matrices that connect the neurons to each other:

- **G** — a fixed, random matrix that generates chaotic neural activity. This never changes.
- **M** — a plastic (learnable) matrix that sculpts the chaos into useful patterns. This is what learns.

The network starts chaotic. During training, M gradually redirects the chaotic dynamics to produce the desired output, while G provides a stable scaffold. Separate *readout* weights (w) extract the output signal, and separate *feedback* weights (Q) provide task-specific learning signals back to the network.

The learning rule is simple: M updates to align its contribution to neural activity (Mr) with the feedback signal (Qz), regularized by the fixed scaffold (αGr). Everything is local — each synapse in M only needs information from the neurons it connects.

## Phase 1: Does It Work? (Yes, With Caveats)

### Reproducing the paper

My first task was validating the implementation. After a debugging session that included discovering the paper uses dt=1ms (not the 0.1ms I initially assumed — a 10x difference in how many learning updates occur per simulated second), I got a 500-neuron network to learn autonomous sine wave generation. Error dropped monotonically across training, and the network produced a near-perfect sine wave during testing with plasticity turned off.

I then reproduced the paper's key parameter studies: the alignment strength α is optimal at 1.0, and the network learns best at the "edge of chaos" where the gain parameter g ≈ 1.0. These matched the published figures.

### The Lorenz attractor — first interesting result

Training on the 3D Lorenz chaotic attractor (the butterfly-shaped system from chaos theory) produced an interesting result. After training, the network's output diverged from the *specific trajectory* it was trained on — but the output stayed on the *attractor manifold*. It had learned the shape of the attractor, not the exact path through it. The network was memorizing the dynamical landscape, not individual trajectories.

### The pendulum — where things broke

I pushed beyond the paper's test cases by training on a damped pendulum with two state variables (angle θ and angular velocity ω). The network learned to reproduce the training trajectory well (error 0.028), but when I tested it on new initial conditions it had never seen, it failed completely.

This was the first important finding: **Predictive Alignment memorizes trajectories. It does not learn physics.**

An autonomous Predictive Alignment network that learned a pendulum starting from θ=2.0 cannot predict what happens from θ=1.0. It only knows the specific pattern it was shown.

## Phase 2: Can We Make It Learn Physics? (No, But Something Else Can)

I spent several experiments trying to make Predictive Alignment generalize to novel initial conditions. The idea was to feed the network's own output back as input — making it predict state-to-state transitions rather than just generating trajectories from memory.

### The self-feeding problem

When you train a network with "teacher forcing" (feeding it correct data as input) but then test it with self-feeding (its own predictions as input), small errors compound. The network has never seen its own mistakes during training, so it doesn't know how to recover from them.

I tried scheduled sampling — gradually replacing correct inputs with the network's own predictions during training. Standard backprop networks can learn from this because gradient information flows through the feedback loop. But Predictive Alignment's learning rule is purely local and one-step. It literally cannot see that its output from step 47 will become its input at step 48. The learning rule has no mechanism to optimize multi-step stability.

This was the second important finding: **Predictive Alignment's locality — its main advantage for biological plausibility — is exactly what prevents it from learning autoregressive prediction.**

### The right tool for the job: Neural ODEs

I then tested a completely different architecture: a Neural ODE. Instead of a recurrent network generating trajectories from internal dynamics, a small feedforward network (just 8,642 parameters vs the recurrent network's 252,000) learns the *derivative* — how the state changes at each instant. A numerical solver handles the time integration.

The results were dramatic. On five held-out initial conditions the pendulum had never seen during training:

| Method | Parameters | Mean Error | What Happened |
|--------|-----------|------------|---------------|
| Predictive Alignment (teacher forcing) | 252,000 | 5.5 | Diverges to fixed point |
| Predictive Alignment (multi-IC) | 252,000 | 2.3 | Spurious oscillation at wrong frequency |
| Predictive Alignment + scheduled sampling | 252,000 | 3.6 | Diverges to different fixed point |
| Backprop RNN + scheduled sampling | 252,000 | 1.7 | Collapses to zero output |
| **Neural ODE** | **8,642** | **0.085** | **Tracks ground truth accurately** |

The Neural ODE used 30x fewer parameters and produced 20x lower error than the best alternative. It learned the actual physics — the vector field it produced matched the true pendulum equations to three significant figures.

I extended this to Lotka-Volterra predator-prey dynamics and the chaotic double pendulum. The story was consistent: architecture match to problem structure dominates everything else. When you want to learn a differential equation, use a network that directly parameterizes a differential equation.

### The Hamiltonian lesson

For the Lotka-Volterra system (which has a conserved energy quantity), I tested Hamiltonian Neural Networks — networks that output a scalar energy function and derive dynamics from its gradients. This structurally guarantees energy conservation. The progression told a clean story:

1. **Vanilla Neural ODE**: Accurate trajectories, but energy slowly leaks (conservation drift 0.03–1.14)
2. **Hamiltonian + pointwise training**: Energy perfectly conserved (drift 0.0007), but trajectory accuracy 3.6x worse
3. **Hamiltonian + trajectory training**: Both accurate AND conserving (drift 0.005–0.041)

Right architecture + right training signal > either alone.

## Phase 3: What Is Predictive Alignment Actually Good At?

At this point I had a clear picture of what Predictive Alignment can't do (generalize physics across initial conditions) and why (local learning rule can't optimize through feedback loops). But the Lorenz result nagged at me — the network had learned a chaotic attractor autonomously, and that's not trivial.

I decided to test something the original paper never explored: **continual learning**. Can you teach a Predictive Alignment network multiple tasks in sequence without it forgetting the earlier ones?

### The setup

Train a network on a sine wave. Then, without resetting any weights, train it on a Lorenz attractor. Then test: can it still produce the sine wave?

This is the catastrophic forgetting test. Standard backprop networks fail it badly — training on task B overwrites the weights that task A needed.

### The headline result

The sine wave got *better* while the network learned the Lorenz attractor.

After sine training, the sine error was 1.24. After Lorenz training, the sine error dropped to 0.96. The forgetting ratio was 0.77 — below 1.0, meaning the old task improved. This is the opposite of catastrophic forgetting.

I then scaled up to four tasks trained sequentially: sine wave → Lorenz attractor → multi-frequency sine → sawtooth wave. The forgetting matrix after all four:

| Task | Forgetting Ratio | What Happened |
|------|-----------------|---------------|
| Sine wave | 0.88x | Improved |
| Lorenz attractor | 1.02x | Unchanged |
| Multi-frequency sine | 0.68x | Improved substantially |
| Sawtooth | 1.00x | Trained last, baseline |

Zero catastrophic forgetting. Two tasks actually improved while the network learned other things.

### Stress testing

I didn't trust this result, so I ran every ablation I could think of.

**50 sequential tasks** — I created a pool of 50 diverse waveforms (sines, sawtooths, triangles, chirps, amplitude-modulated signals, multi-frequency combinations) and trained them one after another. At N=500 neurons, the mean forgetting ratio was 0.78. No capacity wall. The network handled all 50 without systematic degradation.

**Network size scaling** — I swept the neuron count from 100 to 2,000. There's a sharp phase transition: below N≈500 (for 4 tasks), forgetting appears. Above N≈500, it vanishes completely. At N≥1,000, the maximum forgetting ratio across all tasks is exactly 1.0 — zero forgetting — with most tasks showing anti-forgetting (ratios 0.3–0.8).

**Similar tasks** — I trained on six sine waves with periods only 10ms apart (580, 590, 600, 610, 620, 630 ms). No forgetting. In fact, the very similar condition showed the *best* mean forgetting ratio (0.826), suggesting that similar tasks reinforce each other rather than competing.

**All 24 task orderings** — I tested every permutation of the 4-task sequence. 18 out of 24 orderings showed zero forgetting. The worst case (1.515) occurred with a specific ordering, but even that is mild degradation, not catastrophic collapse.

### Why it works: the ablation experiments

I ran two critical ablations to understand the mechanism.

**G ablation (remove the fixed scaffold):** Set G=0, put all connectivity in M. Result: the network couldn't learn *anything*. Not "learned but forgot" — literally no learning occurred. G provides the chaotic dynamics that M sculpts, and without it, the learning rule has nothing to work with. G isn't just protecting against forgetting — it's a prerequisite for Predictive Alignment to function at all.

**Q ablation (share feedback across tasks):** Use the same random feedback matrix Q for all tasks instead of independent Q per task. Result: still zero forgetting. Even with shared Q *and* shared readout weights w, tasks don't interfere.

This overturned my initial hypothesis. I had expected the mechanism to be about independent Q matrices projecting learning signals into orthogonal subspaces. Instead, **the forgetting resistance comes from the interaction between the fixed G scaffold and the M learning rule itself.** M naturally partitions its updates into non-interfering directions, regardless of whether the feedback pathways are shared or separate.

**BPTT comparison:** I ran the same four-task protocol using backpropagation through time instead of the Predictive Alignment learning rule. BPTT showed forgetting ratios near 1.0 — but for a trivial reason: it barely learned the tasks at all with the same training budget. Predictive Alignment's local rule processes every timestep as a learning opportunity, giving it thousands of parameter updates where truncated backprop gets only hundreds. For autonomous trajectory generation, Predictive Alignment is fundamentally more sample-efficient.

**Sequential vs simultaneous training:** I compared training tasks one at a time (sequential), cycling every step (round-robin), and cycling in 500-step blocks. Sequential training performed best. Since forgetting is zero, there's no benefit to interleaving — and some tasks (especially sawtooth) perform significantly worse when interleaved. The practical recommendation: just train one task at a time.

## What Does This Mean?

### The specific finding

Predictive Alignment networks resist catastrophic forgetting as a structural property of their architecture. This isn't a training trick or a regularization method — it emerges from how the fixed scaffold G interacts with the learnable matrix M. The mechanism works across 50 sequential tasks, holds for similar and dissimilar tasks, is largely independent of training order, and has a clean phase transition tied to network dimensionality.

Nobody has published this result for Predictive Alignment before. The original Asabuki & Clopath (2025) paper never tested continual learning.

### The broader context

Catastrophic forgetting is one of the most persistent problems in machine learning. Training modern language models costs hundreds of millions of dollars partly because you can't just add new knowledge — you typically retrain on everything from scratch. The standard workarounds (elastic weight consolidation, progressive neural networks, replay buffers) all add complexity and cost.

This result shows that a biologically-inspired architecture gets continual learning for free. The fixed scaffold + local learning rule design naturally creates non-interfering memory storage, with capacity that scales with network size.

### What this doesn't mean

This is a proof of principle on small networks (500–2,000 neurons) learning relatively simple signals (sine waves, chaotic attractors, waveforms). The gap between this and billion-parameter language models is enormous. Whether the same architectural principle — fixed scaffold + plastic overlay + task-specific readouts — would scale to practical systems is an open question.

The connection to existing techniques is suggestive. LoRA (Low-Rank Adaptation), one of the most popular methods for fine-tuning large language models, uses a conceptually similar structure: freeze the base model weights (analogous to G), train small adapter matrices on top (analogous to M). But LoRA was designed as a practical engineering shortcut, not derived from a principled learning rule. Understanding *why* fixed-scaffold architectures resist forgetting — through the kind of ablation experiments described here — could inform how these methods are designed and deployed.

### The recurring lesson

Across all three phases of this work, the same principle appeared repeatedly: **architecture match to problem structure matters more than training cleverness.**

- Predictive Alignment can't learn physics through self-feeding because its local learning rule can't see through feedback loops → use a Neural ODE that directly parameterizes derivatives
- Neural ODEs can't conserve energy because nothing structurally prevents it → use a Hamiltonian Neural Network that derives dynamics from a conserved quantity
- Predictive Alignment resists catastrophic forgetting because the fixed G scaffold structurally prevents task interference → no special training procedure needed

Encode your constraints in the architecture. Don't hope the optimizer discovers them.

---

## Details

**Hardware:** NVIDIA DGX Spark (GB10 Grace Blackwell GPU, 128GB unified memory)

**Paper:** Asabuki, T. & Clopath, C. "Taming the chaos gently: a predictive alignment learning rule in recurrent neural networks." *Nature Communications* 16, 6784 (2025). DOI: 10.1038/s41467-025-61309-9

**Code and full research log:** [github.com/thadreber-web/predictive-alignment](https://github.com/thadreber-web/predictive-alignment)

**Experiment count:** 22 experiments across three phases, including 12 continual learning experiments with systematic ablations.

**What I skipped:** Experiment 06 (Ready-Set-Go timing task) was deferred due to estimated 4–6 hour runtime. The implementation was validated against the paper's official code; three other reproduction experiments confirmed the codebase works correctly.
