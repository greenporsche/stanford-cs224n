# Concept map: *Stanford CS224N (Winter 2026)*, Chapter 3 (Lecture 3: Backpropagation and Neural Network Basics)

<!--
TEMPLATE. Copy to <work folder>/concept-map.md and fill it in. The Lead is the only writer.
Keep every "##" section, even when it is empty: the viewer (viewer/build.py) reads them by name.
It starts with ZERO entries: write an entry or row only after the Reviewer has judged it. The shapes to
follow are in comments like this one; copy the shape, never the comment.
The H1 above drives the viewer's page title, so keep its shape: "Concept map: *Work*, Chapter N (Topic)".
Table cells and ids: a vertical bar inside a table cell is a column break, so write `\|` (or the word "given").
Put a space or punctuation on both sides of every id (`ch03:C05`, `C12`); an id run into the word beside it
is not read as a pointer.
The worked example is room 65's pilot map: docs/pilots/concept-extraction/udl-ch2/concept-map.md.
-->

Lead: claude (sole writer). Reviewer: codex. An entry appears here only after the Reviewer has judged it.
The map is meant to be complete enough to rebuild the lesson without the source. Citations use section numbers
plus short quotes.

## Teaching arc

<!-- Numbered steps: how the chapter frames its ideas and in what order it teaches them, with short quotes
where the author says so. -->

1. **The plan and the goal** (slides 1–9). The title slide reads "Lecture 3: Neural Network Foundations" (Diyi Yang). The lecture runs in four timed parts: "Course logistics (3 mins) + Word2vec evaluation (7 mins)", "Introducing neural networks (10 mins)", "Matrix calculus (25 mins)" and "Backpropagation (35 mins)". Its key goal: "the mathematics and practical implementation of how neural networks are trained by backpropagation" (slide 2; the slide's header reads "Lecture 2: Neural Network Foundations", a typo). Slide 3 frames Assignment 2 as "all about making sure you really understand the math of neural networks … then we'll let the software do it!", which "also teaches us about dependency parsing". Slides 4–9 repeat Lecture 2's evaluation slides as a recap (ch02:C29, ch02:C30, ch02:C32, ch02:C33).

2. **From word vectors to a neural classifier** (slides 10–14; notes 3 §1–1.3). Named entity recognition labels word tokens (C01); a simple approach classifies each word from its window (C02). A neural classifier learns the representation too (C03), starting from an embedding layer (C04). Its unit is the neuron (C05), and slide 14's one-hidden-layer network scores "Paris" as a location (C06).

3. **Non-linearities** (slides 15–17; notes 3 §2.4). The element-wise f can be a classic function (C07), a smooth modern one (C08) or a gated unit (C09). Without such non-linearities, stacked layers collapse to one linear map (C10).

4. **What to minimize** (slides 18–19; notes 3 §1.4). The loss is restated as cross-entropy (C11); notes 3 offers a max-margin alternative (C12). Slide 19 recalls SGD (B08) and asks how to compute ∇_θJ(θ): by hand, or by backpropagation.

5. **Matrix calculus** (slides 19–33). Gradients can be computed by hand, fully vectorized (C13). The tools: the gradient (C14), its generalization the Jacobian (C15), the chain rule as a product of Jacobians (C16), and three worked Jacobians: the elementwise activation (C17) and the affine map and score (C18).

6. **The worked gradient** (slides 34–55; notes 3 §1.5–1.6). The NER net's score is differentiated by breaking it into pieces, applying the chain rule and writing out Jacobians (C19). The shared factor δ is computed once (C20), the weight-matrix gradient is an outer product (C21), and the Jacobian form gives way to the shape convention for implementation (C22).

7. **Backpropagation** (slides 56–75). The equations become a computation graph run forward (C23); backpropagation runs it backward, each node multiplying its upstream gradient by its local gradient (C24). Gradients sum where paths branch (C25); +, max and * nodes distribute, route and switch them (C26). Done for all parameters at once in reverse topological order, backprop costs the same big-O as the forward pass (C27).

8. **Backprop in practice** (slides 76–80; notes 3 §2.1). Frameworks automate it (C28); numeric gradients check it (C29). The summary (slide 79): "Backpropagation: recursively (and hence efficiently) apply the chain rule along computation graph"; "[downstream gradient] = [upstream gradient] x [local gradient]"; "Forward pass: compute results of operations and save intermediate values"; "Backward pass: apply chain rule to compute gradients". And why learn it when frameworks do it (C30).

9. **Tips and tricks** (notes 3 §2.2–2.9; no slides). Regularize (C31) and drop out (C32); preprocess the data (C33); initialize well (C34); tune and schedule the learning rate (C35); add momentum (C36); or let adaptive optimizers set per-parameter rates (C37).

10. **Assignment 2** (released with the lecture; due January 22). Part 1 works the word2vec gradients by hand; Part 2 asks about Adam (C37) and dropout (C32); Part 3 builds a transition-based dependency parser (C38) with a neural transition classifier (C39), scored by UAS (C40).

## Assumed background

What the chapter uses without teaching it is recorded as **studs** (B01…, below).

**Chapter scope:** the Winter 2026 Lecture 3 deck, all 80 PDF pages (slide N = PDF page N; printed numbers run to 93, and printed slides 67–77 and 89–90 are not in the PDF); notes 3 (Winter 2019, "Neural Networks, Backpropagation", 18 pp., including margin notes, snippets and figures); and Assignment 2 (handout, 8 pp., and the starter code's instructions in parser_model.py and run.py). Consulted where the scope leans on them: Clark's gradient notes and the calculus review (U5), He et al. 2015 (PReLU), Glorot and Bengio 2010, Kingma and Ba 2015, Chen and Manning 2014, Karpathy 2016. Lectures 1 and 2 (ch01, ch02) are the earlier chapters; repeats are referenced as `Taught in` or `**Extends:**`.

## Figures

Concepts cite figures by number. Images are linked, never committed.

| Fig | Shows | Teaches | Image |
|---|---|---|---|
| 1.10 | NER examples: Paris Hilton (PER PER); Samuel Quinn, Hilton Hotel, Paris, April 1989 | C01 | [1.10](materials/slides/cs224n-2026-lecture03-neuralnets/p10.png) |
| 1.11 | window classification of "Paris" with x_window ∈ ℝ^{5d} | C02 | [1.11](materials/slides/cs224n-2026-lecture03-neuralnets/p11.png) |
| 1.13 | typical softmax classifier vs neural network classifier; linear and non-linear boundaries; x = Le | C03, C04 | [1.13](materials/slides/cs224n-2026-lecture03-neuralnets/p13.png) |
| 1.14 | the NER network: x ∈ ℝ^{5d}, h = f(Wx + b), s = uᵀh, J_t(θ) = σ(s) | C06 | [1.14](materials/slides/cs224n-2026-lecture03-neuralnets/p14.png) |
| 1.15 | logistic, tanh, ReLU and leaky/parametric ReLU with their plots; tanh(z) = 2logistic(2z) − 1 | C07 | [1.15](materials/slides/cs224n-2026-lecture03-neuralnets/p15.png) |
| 1.16 | GELU, SiLU, Swish, GLU and SwiGLU with a plot comparing them to ReLU and the sigmoid | C08, C09 | [1.16](materials/slides/cs224n-2026-lecture03-neuralnets/p16.png) |
| 1.17 | why non-linearities: W₁W₂x = Wx; curve fits of increasing flexibility; linear vs non-linear boundaries | C10 | [1.17](materials/slides/cs224n-2026-lecture03-neuralnets/p17.png) |
| 1.18 | cross entropy H(p, q) = −Σ p(c) log q(c); with one-hot p it is −log p(y_i \| x_i) | C11 | [1.18](materials/slides/cs224n-2026-lecture03-neuralnets/p18.png) |
| 1.19 | SGD recap: θ^{new} = θ^{old} − α∇_θJ(θ); θ includes the word vectors; by hand or by backpropagation | C13 | [1.19](materials/slides/cs224n-2026-lecture03-neuralnets/p19.png) |
| 1.20 | "3. Computing Gradients by Hand": matrix calculus, fully vectorized | C13 | [1.20](materials/slides/cs224n-2026-lecture03-neuralnets/p20.png) |
| 1.21 | gradient of f(x) = x³; 1.01³ and 4.01³ | C14 | [1.21](materials/slides/cs224n-2026-lecture03-neuralnets/p21.png) |
| 1.22 | gradient of a function with n inputs: a vector of partial derivatives | C14 | [1.22](materials/slides/cs224n-2026-lecture03-neuralnets/p22.png) |
| 1.23 | the Jacobian, an m × n matrix of partial derivatives | C15 | [1.23](materials/slides/cs224n-2026-lecture03-neuralnets/p23.png) |
| 1.24 | chain rule: multiply derivatives; multiply Jacobians | C16 | [1.24](materials/slides/cs224n-2026-lecture03-neuralnets/p24.png) |
| 1.25 | example Jacobian, elementwise activation (build 1) | C17 | [1.25](materials/slides/cs224n-2026-lecture03-neuralnets/p25.png) |
| 1.26 | "n outputs and n inputs → n by n Jacobian" (build 2) | C17 | [1.26](materials/slides/cs224n-2026-lecture03-neuralnets/p26.png) |
| 1.27 | elementwise Jacobian (build 3) | C17 | [1.27](materials/slides/cs224n-2026-lecture03-neuralnets/p27.png) |
| 1.28 | elementwise Jacobian (build 4) | C17 | [1.28](materials/slides/cs224n-2026-lecture03-neuralnets/p28.png) |
| 1.29 | ∂h/∂z = diag(f′(z)) | C17 | [1.29](materials/slides/cs224n-2026-lecture03-neuralnets/p29.png) |
| 1.30 | other Jacobians (build 1): ∂/∂x(Wx + b) = W | C18 | [1.30](materials/slides/cs224n-2026-lecture03-neuralnets/p30.png) |
| 1.31 | other Jacobians (build 2): ∂/∂b(Wx + b) = I | C18 | [1.31](materials/slides/cs224n-2026-lecture03-neuralnets/p31.png) |
| 1.32 | other Jacobians (build 3): ∂/∂u(uᵀh) = hᵀ, with the shape-convention fine print | C18 | [1.32](materials/slides/cs224n-2026-lecture03-neuralnets/p32.png) |
| 1.33 | all three Jacobians | C18 | [1.33](materials/slides/cs224n-2026-lecture03-neuralnets/p33.png) |
| 1.34 | back to the NER net | C19 | [1.34](materials/slides/cs224n-2026-lecture03-neuralnets/p34.png) |
| 1.35 | the target: ∂s/∂b (the score, not the loss, for simplicity) | C19 | [1.35](materials/slides/cs224n-2026-lecture03-neuralnets/p35.png) |
| 1.36 | a. break up equations: s = uᵀh, h = f(z), z = Wx + b | C19 | [1.36](materials/slides/cs224n-2026-lecture03-neuralnets/p36.png) |
| 1.37 | b. apply the chain rule (build 1) | C19 | [1.37](materials/slides/cs224n-2026-lecture03-neuralnets/p37.png) |
| 1.38 | chain rule (build 2) | C19 | [1.38](materials/slides/cs224n-2026-lecture03-neuralnets/p38.png) |
| 1.39 | chain rule (build 3) | C19 | [1.39](materials/slides/cs224n-2026-lecture03-neuralnets/p39.png) |
| 1.40 | chain rule (build 4) | C19 | [1.40](materials/slides/cs224n-2026-lecture03-neuralnets/p40.png) |
| 1.41 | c. write out the Jacobians (build 1) | C19 | [1.41](materials/slides/cs224n-2026-lecture03-neuralnets/p41.png) |
| 1.42 | Jacobians (build 2) | C19 | [1.42](materials/slides/cs224n-2026-lecture03-neuralnets/p42.png) |
| 1.43 | Jacobians (build 3) | C19 | [1.43](materials/slides/cs224n-2026-lecture03-neuralnets/p43.png) |
| 1.44 | Jacobians (build 4) | C19 | [1.44](materials/slides/cs224n-2026-lecture03-neuralnets/p44.png) |
| 1.45 | ∂s/∂b = uᵀ diag(f′(z)) I = uᵀ ∘ f′(z); the Hadamard product | C19 | [1.45](materials/slides/cs224n-2026-lecture03-neuralnets/p45.png) |
| 1.46 | re-using computation (build 1) | C20 | [1.46](materials/slides/cs224n-2026-lecture03-neuralnets/p46.png) |
| 1.47 | "The same! Let's avoid duplicated computation" | C20 | [1.47](materials/slides/cs224n-2026-lecture03-neuralnets/p47.png) |
| 1.48 | δ = uᵀ ∘ f′(z), the upstream gradient ("error signal") | C20 | [1.48](materials/slides/cs224n-2026-lecture03-neuralnets/p48.png) |
| 1.49 | ∂s/∂W as a 1 × nm Jacobian: "Inconvenient" | C21, C22 | [1.49](materials/slides/cs224n-2026-lecture03-neuralnets/p49.png) |
| 1.50 | the shape convention: the gradient has the shape of the parameters | C22 | [1.50](materials/slides/cs224n-2026-lecture03-neuralnets/p50.png) |
| 1.51 | ∂s/∂W = δᵀxᵀ | C21 | [1.51](materials/slides/cs224n-2026-lecture03-neuralnets/p51.png) |
| 1.52 | why the transposes: outer product | C21 | [1.52](materials/slides/cs224n-2026-lecture03-neuralnets/p52.png) |
| 1.53 | W_ij only contributes to z_i; ∂z_i/∂W_ij = x_j | C21 | [1.53](materials/slides/cs224n-2026-lecture03-neuralnets/p53.png) |
| 1.54 | Jacobian form vs shape convention for ∂s/∂b | C22 | [1.54](materials/slides/cs224n-2026-lecture03-neuralnets/p54.png) |
| 1.55 | two options; "Use Jacobian to compute; Use shape convention to format" | C22 | [1.55](materials/slides/cs224n-2026-lecture03-neuralnets/p55.png) |
| 1.56 | "4. Backpropagation": the chain rule plus re-using derivatives from higher layers | C24 | [1.56](materials/slides/cs224n-2026-lecture03-neuralnets/p56.png) |
| 1.57 | computation graph of the NER net (build 1) | C23 | [1.57](materials/slides/cs224n-2026-lecture03-neuralnets/p57.png) |
| 1.58 | computation graph (build 2): edges pass results | C23 | [1.58](materials/slides/cs224n-2026-lecture03-neuralnets/p58.png) |
| 1.59 | "Forward Propagation" | C23 | [1.59](materials/slides/cs224n-2026-lecture03-neuralnets/p59.png) |
| 1.60 | backpropagation: go backwards along edges, pass gradients | C24 | [1.60](materials/slides/cs224n-2026-lecture03-neuralnets/p60.png) |
| 1.61 | single node: upstream and downstream gradients | C24 | [1.61](materials/slides/cs224n-2026-lecture03-neuralnets/p61.png) |
| 1.62 | single node: local gradient | C24 | [1.62](materials/slides/cs224n-2026-lecture03-neuralnets/p62.png) |
| 1.63 | single node: "Chain rule!" | C24 | [1.63](materials/slides/cs224n-2026-lecture03-neuralnets/p63.png) |
| 1.64 | [downstream gradient] = [upstream gradient] x [local gradient] | C24 | [1.64](materials/slides/cs224n-2026-lecture03-neuralnets/p64.png) |
| 1.65 | node with multiple inputs | C24 | [1.65](materials/slides/cs224n-2026-lecture03-neuralnets/p65.png) |
| 1.66 | z = Wx: multiple local gradients | C24 | [1.66](materials/slides/cs224n-2026-lecture03-neuralnets/p66.png) |
| 1.67 | gradients sum at outward branches (build 1) | C25 | [1.67](materials/slides/cs224n-2026-lecture03-neuralnets/p67.png) |
| 1.68 | a = x + y, b = max(y, z), f = ab: ∂f/∂y sums two paths | C25 | [1.68](materials/slides/cs224n-2026-lecture03-neuralnets/p68.png) |
| 1.69 | node intuitions for f = (x + y) max(y, z), x = 1, y = 2, z = 0: + distributes | C26 | [1.69](materials/slides/cs224n-2026-lecture03-neuralnets/p69.png) |
| 1.70 | max routes | C26 | [1.70](materials/slides/cs224n-2026-lecture03-neuralnets/p70.png) |
| 1.71 | * switches | C26 | [1.71](materials/slides/cs224n-2026-lecture03-neuralnets/p71.png) |
| 1.72 | efficiency: the incorrect way (build 1) | C27 | [1.72](materials/slides/cs224n-2026-lecture03-neuralnets/p72.png) |
| 1.73 | duplicated computation | C27 | [1.73](materials/slides/cs224n-2026-lecture03-neuralnets/p73.png) |
| 1.74 | the correct way: all gradients at once | C27 | [1.74](materials/slides/cs224n-2026-lecture03-neuralnets/p74.png) |
| 1.75 | back-prop in a general computation graph (fprop, bprop, same big-O) | C27 | [1.75](materials/slides/cs224n-2026-lecture03-neuralnets/p75.png) |
| 1.76 | automatic differentiation | C28 | [1.76](materials/slides/cs224n-2026-lecture03-neuralnets/p76.png) |
| 1.77 | backprop implementation: forward() over topologically sorted nodes, backward() in reverse | C28 | [1.77](materials/slides/cs224n-2026-lecture03-neuralnets/p77.png) |
| 1.78 | numeric gradient: f′(x) ≈ (f(x + h) − f(x − h))/2h, h ≈ 1e-4 | C29 | [1.78](materials/slides/cs224n-2026-lecture03-neuralnets/p78.png) |
| 1.79 | summary | — | [1.79](materials/slides/cs224n-2026-lecture03-neuralnets/p79.png) |
| 1.80 | why learn all these details about gradients? (Karpathy) | C30 | [1.80](materials/slides/cs224n-2026-lecture03-neuralnets/p80.png) |
| 2.1 | a non-linear decision boundary separating the data | C03 | [2.1](materials/figures/notes3-p01.png) |
| 2.2 | a sigmoid neuron: inputs scaled, summed, biased and squashed | C05 | [2.2](materials/figures/notes3-p02.png) |
| 2.3 | a layer of sigmoid units receiving the same input x | C06 | [2.3](materials/figures/notes3-p02.png) |
| 2.4 | a feed-forward network computing its score | C06 | [2.4](materials/figures/notes3-p03.png) |
| 2.5 | the 4-2-1 network used for backpropagation | C19, C20 | [2.5](materials/figures/notes3-p04.png) |
| 2.6 | the subnetwork for updating W⁽¹⁾₁₄ | C19 | [2.6](materials/figures/notes3-p06.png) |
| 2.7 | propagating error from δ⁽ᵏ⁾ to δ⁽ᵏ⁻¹⁾ (a single path) | C20 | [2.7](materials/figures/notes3-p07.png) |
| 2.8 | propagating error from δ⁽ᵏ⁾ to δ⁽ᵏ⁻¹⁾ (all paths) | C20 | [2.8](materials/figures/notes3-p07.png) |
| 2.D | dropout applied to an artificial neural network (Srivastava et al.) | C32 | [2.D](materials/figures/notes3-p12.png) |
| 2.9 | the response of a sigmoid | C07 | [2.9](materials/figures/notes3-p12.png) |
| 2.10 | the response of a tanh | C07 | [2.10](materials/figures/notes3-p13.png) |
| 2.11 | the response of a hard tanh | C07 | [2.11](materials/figures/notes3-p13.png) |
| 2.12 | the response of a soft sign | C07 | [2.12](materials/figures/notes3-p13.png) |
| 2.13 | the response of a ReLU | C07 | [2.13](materials/figures/notes3-p14.png) |
| 2.14 | the response of a leaky ReLU | C07 | [2.14](materials/figures/notes3-p14.png) |
| 2.15 | a large learning rate makes the error diverge | C35 | [2.15](materials/figures/notes3-p16.png) |
| 3.1 | A2 Figure 1: the skip-gram prediction model with window size 2 | — | [3.1](materials/figures/a2-p01.png) |
| 3.2 | A2 Figure 2: Leaky ReLU | C07 | [3.2](materials/figures/a2-p03.png) |
| 3.5 | A2's dependency tree for "I presented my findings at the NLP conference", with the first transitions | C38 | [3.5](materials/figures/a2-p05.png) |

<!-- Row shape: | 3.1 | what the figure shows | C01, C02 | [3.1](<image url>) |
     For a picture extracted from a prepared EPUB (overlay section 1), the link is its absolute file URI: | 4.1 | ... | C05 | [4.1](file:///.../prepared/books/BK-<hash>.assets/OPS/images/<file>) | -->

Figure numbers: 1.N is slide N (PDF page N of the W26 deck). 2.N is notes 3's Figure N (2.D is its unnumbered dropout diagram); 3.N is an A2 figure. Notes 3 and A2 figures link to a render of the page that holds them (`materials/figures/notes3-pNN.png`, `materials/figures/a2-pNN.png`). Image links are relative to the published lecture folder, where the slide images sit beside the map.

## Concepts

<!-- Entry shape (write "###" with no leading space; numbering runs C01, C02, …):
 ### C01 <Label (aliases in parentheses)>
- **Meaning:** what it is, in the chapter's sense, with its scope and distinctions from near neighbours.
- **Symbols:** notation the chapter defines, e.g. φ = parameters. (Omit when none.)
- **Key points:**
  - the author's substantive claims, each with a short quote and its section, e.g. "…" (§3.1)
- **Example:** the chapter's own example or intuition, with its section.
- **Scope:** where a claim applies only under stated conditions. (Omit when none.)
- *Revised at <unit> (#<msg>):* what changed and why, when a later passage refined the entry.
- **Extends:** ch02:C05, what this chapter adds. (Book mode only, overlay §8; omit otherwise.)
-->

### C01 Named entity recognition as token labelling
- **Meaning:** "find and classify names in text, by labeling word tokens" (slide 10).
- **Key points:**
  - Possible uses (slide 10): "Tracking mentions of particular entities in documents"; "For question answering, answers are usually named entities"; "Relating sentiment analysis to the entity under discussion"; "Often followed by Entity Linking/Canonicalization into a Knowledge Base such as Wikidata".
- **Example:** "Last night , Paris Hilton wowed in a sequin gown ." labels Paris Hilton PER PER; "Samuel Quinn was arrested in the Hilton Hotel in Paris in April 1989 ." labels Samuel, Quinn PER; Hilton, Hotel, Paris LOC; April, 1989 DATE ("the" is unlabelled) (slide 10). "Paris" and "Hilton" change type with context.
- **Figures:** 1.10.
- **Extends:** ch02:C33, the task itself rather than its use as an extrinsic test.

### C02 Window classification with a binary logistic classifier per class
- **Meaning:** "classify each word in its context window of neighboring words"; "Train logistic classifier on hand-labeled data to classify center word {yes/no} for each class based on a concatenation of word vectors in a window"; "To classify all words: run classifier for each class on the vector centered on each word" (slide 11).
- **Key points:**
  - "Really, we usually use multi-class softmax, but we're trying to keep it simple ☺" (slide 11).
- **Example:** "Paris" as ± location in "the museums in Paris are amazing to see .", window length 2: x_window = [x_museums x_in x_Paris x_are x_amazing]ᵀ ∈ ℝ^{5d} (slide 11).
- **Figures:** 1.11.
- **Extends:** ch02:C39, adding the binary one-classifier-per-class setup.

### C03 Neural network classifier vs the linear softmax classifier
- **Meaning:** A "Typical ML/stats softmax classifier", p(y \| x) = exp(W_y·x) / Σ_c exp(W_c·x), learns only W ("not input representation x, which has sparse symbolic features") and "gives linear decision boundary, which can be limiting". A neural network classifier "differs in that: We learn both W and (distributed!) representations for words", and "We use deep networks—more layers—that let us re-represent and compose our data multiple times, giving a non-linear classifier" (slide 13).
- **Key points:**
  - "The word vectors x re-represent one-hot vectors, moving them around in an intermediate layer vector space, for easy classification with a (linear) softmax classifier"; "But typically, it is linear relative to the pre-final layer representation" (slide 13).
  - "Neural networks are a family of classifiers with non-linear decision boundary" (notes 3 §1).
- **Figures:** 1.13.
- **Extends:** ch02:C37 and ch02:C40, learning the representation as well (ch02:C36 retrained the vectors).

### C04 The embedding layer (x = Le)
- **Meaning:** "Conceptually, we have an embedding layer: x = Le" (slide 13): a word's vector is the embedding matrix L times its one-hot vector e, which picks out one column of L.
- **Key points:**
  - Slide 14 labels the network's input "Embedding of 1-hot words".
- **Figures:** 1.13, 1.14.

### C05 The artificial neuron (sigmoid unit)
- **Meaning:** "a generic computational unit that takes n inputs and produces a single output"; the sigmoid ("binary logistic regression") unit computes a = 1/(1 + exp(−(wᵀx + b))) from weights w and bias b, or equivalently a = 1/(1 + exp(−[wᵀ b]·[x 1])) (notes 3 §1.1, Fig. 2).
- **Key points:**
  - Neural networks are "biologically inspired", but "it is usually better to not draw too many parallels between the two" (notes 3 §1.1, "Fun Fact").

### C06 A one-hidden-layer feed-forward network (the NER scorer)
- **Meaning:** A layer of m neurons computes z = Wx + b and a = σ(z), with W ∈ ℝ^{m×n} (notes 3 §1.2, Fig. 3); a score is s = Uᵀ f(Wx + b) (notes 3 §1.3). Slide 14: x (input) ∈ ℝ^{5d}, h = f(Wx + b), s = uᵀh, J_t(θ) = σ(s) = 1/(1 + e^{−s}), the "predicted model probability of class"; "f = Some element-wise non-linear function, e.g., logistic, tanh, ReLU"; "We do supervised training and want high score if it's a location".
- **Key points:**
  - Why a hidden layer: "maybe it should matter that 'Museums' is the first word only if 'in' is the second word. Such non-linear decisions can often not be captured by inputs fed directly to a Softmax function" (notes 3 §1.3).
  - Activations are "indicators of the presence of some weighted combination of features" (notes 3 §1.2).
  - Source defect: notes 3 §1.2 writes a₁ … a_m (and σ(z)) as 1/(1 + exp(w⁽ⁱ⁾ᵀx + b_i)), dropping §1.1's minus sign.
- **Example:** 4-dimensional word vectors and a 5-word window give x ∈ ℝ^20; with 8 sigmoid units, W ∈ ℝ^{8×20}, b ∈ ℝ^8, U ∈ ℝ^{8×1}, s ∈ ℝ (notes 3 §1.3).
- **Figures:** 1.14.

### C07 Classic activation functions (logistic, tanh, ReLU, leaky and parametric ReLU)
- **Meaning:** The element-wise f of the network (slide 14): logistic ("sigmoid") f(z) = 1/(1 + exp(−z)); tanh(z) = (e^z − e^{−z})/(e^z + e^{−z}); ReLU(z) = max(z, 0); and Leaky ReLU / Parametric ReLU (slide 15). Leaky ReLU fixes a small negative slope; Parametric ReLU (PReLU) learns it: "when a_i is a learnable parameter" it is PReLU, and "If a_i is a small and fixed value" it becomes Leaky ReLU (He et al. 2015 §2.1).
- **Key points:**
  - "tanh is just a rescaled and shifted sigmoid (2× as steep, [−1,1]): tanh(z) = 2logistic(2z) − 1" (slide 15).
  - "Logistic and tanh are still used (e.g., logistic to get a probability); however, often, for deep networks, the first thing to try is ReLU: it trains quickly and performs well due to good gradient backflow"; "ReLU has a negative 'dead zone' that recent proposals mitigate" (slide 15).
  - Notes 3 §2.4 gives each with its gradient: σ′(z) = σ(z)(1 − σ(z)); tanh "is often found to converge faster in practice", tanh′(z) = 1 − tanh²(z); hard tanh, clip(z, −1, 1), is cheaper but saturates for \|z\| > 1, with hardtanh′ = 1 for \|z\| < 1 and 0 for \|z\| > 1; soft sign z/(1 + \|z\|) "does not saturate as easily"; ReLU "does not saturate even for larger values of z", with ReLU′ = 1 for z > 0 and 0 for z < 0; leaky(z) = max(z, k·z), 0 < k < 1, with leaky′ = 1 for z > 0 and k for z < 0, so "a small error is allowed to propagate backwards even when z is negative".
  - The derivatives at the kinks (z = 0; \|z\| = 1 for hard tanh) are undefined; values assigned there are conventions, not ordinary derivatives (notes 3 pp. 13–14).
  - Source defect: notes 3 gives softsign′(z) = sgn(z)/(1 + z)²; the derivative of z/(1 + \|z\|) is 1/(1 + \|z\|)².
- **Figures:** 1.15.

### C08 Smooth modern activations (GELU, SiLU, Swish)
- **Meaning:** GELU (Gaussian error linear unit), GELU(x) = x · P(X ≤ x), X ~ N(0,1), ≈ x · logistic(1.702x), "frequently used with Transformers"; SiLU (Sigmoid linear unit), SiLU(x) = x · σ(x); Swish(x) = x · σ(βx) (slide 16).
- **Key points:**
  - The slide's plot compares them with ReLU and the sigmoid: each is near 0 for large negative x and near x for large positive x, with a small negative dip (slide 16).
- **Figures:** 1.16.

### C09 Gated linear units (GLU, SwiGLU)
- **Meaning:** "GLU (gated linear unit) uses a gate/switch": GLU(x) = (xV + v) ⊗ σ(xW + b). "SwiGLU (Swish-gated linear unit)": SwiGLU(x) = (xV + c) ⊗ Swish_β(xW + b) (slide 16). One linear projection of the input is multiplied element-wise by a gated (sigmoid or Swish) projection of the same input.
- **Key points:**
  - SwiGLU is "Frequently used, e.g., in LLama3, Qwen3"; "Check out GLU Variants Improve Transformer" (slide 16; Shazeer 2020).
- **Figures:** 1.16.

### C10 Why networks need non-linearities
- **Meaning:** "Neural networks do function approximation, e.g., regression or classification"; "Without non-linearities, deep neural networks can't do anything more than a linear transform": "Extra layers could just be compiled down into a single linear transform: W₁W₂x = Wx" (slide 17).
- **Key points:**
  - "But, with more layers that include non-linearities, they can approximate any complex function!" (slide 17).
- **Example:** three curve fits of increasing flexibility, and a linear vs a non-linear decision boundary (slide 17).
- **Figures:** 1.17.

### C11 Cross-entropy loss
- **Meaning:** "Until now, our objective was stated as to maximize the probability of the correct class y or equivalently we can minimize the negative log probability of that class"; "Now restated in terms of cross entropy, a concept from information theory": for the true distribution p and the model's distribution q, H(p, q) = −Σ_{c=1}^{C} p(c) log q(c) (slide 18).
- **Key points:**
  - With a one-hot ground truth p = [0, …, 0, 1, 0, …, 0], "the only term left is the negative log probability of the true class y_i: −log p(y_i \| x_i)" (slide 18); "you use this in PyTorch!" (slide 18's title).
- **Figures:** 1.18.
- **Extends:** ch02:C37 and ch02:C10, adding the H(p, q) definition and the information-theory framing.

### C12 The max-margin objective
- **Meaning:** Make "the score computed for 'true' labeled data points … higher than the score computed for 'false' labeled data points": with s the score of a true window ("Museums in Paris are amazing") and s_c of a corrupt one ("Not all museums in Paris"), minimize J = max(Δ + s_c − s, 0), with Δ scaled to 1: J = max(1 + s_c − s, 0), where s_c = Uᵀf(Wx_c + b) and s = Uᵀf(Wx + b) (notes 3 §1.4).
- **Key points:**
  - Built in steps: minimize max(s_c − s, 0), so "error is only computed if s_c > s"; that "does not attempt to create a margin of safety", so add the margin Δ (notes 3 §1.4).
  - "We can scale this margin such that it is Δ = 1 and let the other parameters in the optimization problem adapt to this without any change in performance"; see "functional and geometric margins … Support Vector Machines" (notes 3 §1.4).

### C13 Vectorized gradients by hand (matrix calculus)
- **Meaning:** Training needs ∇_θJ(θ): "How can we compute ∇_θJ(θ)? 1. By hand 2. Algorithmically: the backpropagation algorithm" (slide 19). By hand means "Matrix calculus: Fully vectorized gradients" (slide 20).
- **Key points:**
  - "In deep learning, θ includes the data representation (e.g., word vectors) too!" (slide 19).
  - "'Multivariable calculus is just like single-variable calculus if you use matrices'"; "Much faster and more useful than non-vectorized gradients"; "But doing a non-vectorized gradient can be good for intuition; recall the second lecture for an example" (slide 20).
  - "Lecture notes and matrix calculus notes cover this material in more detail"; Math 51's online textbook for review (slide 20).
- **Figures:** 1.19, 1.20.

### C14 The gradient (of a scalar function of n inputs)
- **Meaning:** For one input, "It's gradient (slope) is its derivative": "How much will the output change if we change the input a bit?" (slide 21). For n inputs, "Its gradient is a vector of partial derivatives with respect to each input", ∂f/∂x = [∂f/∂x₁, …, ∂f/∂x_n] (slide 22).
- **Example:** f(x) = x³, df/dx = 3x²: "At x = 1 it changes about 3 times as much: 1.01³ = 1.03"; "At x = 4 it changes about 48 times as much: 4.01³ = 64.48" (slide 21).
- **Figures:** 1.21, 1.22.

### C15 The Jacobian matrix
- **Meaning:** "Given a function with m outputs and n inputs", f(x) = [f₁(x₁, …, x_n), …, f_m(x₁, …, x_n)], "It's Jacobian is an m x n matrix of partial derivatives": (∂f/∂x)_ij = ∂f_i/∂x_j (slide 23), the "Generalization of the Gradient".
- **Figures:** 1.23.

### C16 The multivariable chain rule (multiply Jacobians)
- **Meaning:** "For composition of one-variable functions: multiply derivatives"; "For multiple variables functions: multiply Jacobians" (slide 24).
- **Example:** z = 3y, y = x²: dz/dx = (dz/dy)(dy/dx) = (3)(2x) = 6x. h = f(z), z = Wx + b: ∂h/∂x = (∂h/∂z)(∂z/∂x) (slide 24).
- **Figures:** 1.24.

### C17 The Jacobian of an elementwise activation
- **Meaning:** For h = f(z), h_i = f(z_i), with h, z ∈ ℝⁿ: "Function has n outputs and n inputs → n by n Jacobian"; (∂h/∂z)_ij = ∂f(z_i)/∂z_j = f′(z_i) if i = j and 0 otherwise, so ∂h/∂z = diag(f′(z)) (slides 25–29).
- **Figures:** 1.25–1.29.

### C18 Useful Jacobians of the affine map and the score
- **Meaning:** ∂/∂x (Wx + b) = W; ∂/∂b (Wx + b) = I (identity matrix); ∂/∂u (uᵀh) = hᵀ (slides 30–33).
- **Key points:**
  - "Fine print: This is the correct Jacobian. Later we discuss the 'shape convention'; using it the answer would be h" (slide 32).
  - "Compute these at home for practice! Check your answers with the lecture notes" (slides 30–33).
- **Figures:** 1.30–1.33.

### C19 The worked gradient of the NER score (break up, chain rule, Jacobians)
- **Meaning:** "Back to our Neural Net!" The target is ∂s/∂b: "Really, we care about the gradient of the loss J but we will compute the gradient of the score for simplicity" (slide 35). Three steps: "a. Break up equations into simple pieces": s = uᵀh, h = f(z), z = Wx + b, x (input); "Carefully define your variables and keep track of their dimensionality!" (slide 36). "b. Apply the chain rule": ∂s/∂b = (∂s/∂h)(∂h/∂z)(∂z/∂b) (slides 37–40). "c. Write out the Jacobians": ∂s/∂b = uᵀ diag(f′(z)) I = uᵀ ∘ f′(z) (slides 41–45).
- **Symbols:** "⊙ = Hadamard product = element-wise multiplication of 2 vectors to give vector" (slide 45). In the slides' row (Jacobian) convention, uᵀ diag(f′(z)) = (u ⊙ f′(z))ᵀ, a row vector: this is the Jacobian ∂s/∂b.
- **Key points:**
  - Notes 3 §1.5 does the same element by element on a 4-2-1 network: "backpropagated gradients are only affected by values they contribute to"; its worked score gradient is ∂s/∂W⁽¹⁾₁₄ = a⁽¹⁾₄ f′(z⁽²⁾₁) W⁽²⁾₁ (notes 3 p. 6).
- **Figures:** 1.34–1.45.

### C20 Reusing computation: the upstream gradient δ ("error signal")
- **Meaning:** Computing ∂s/∂W "Using the chain rule again" repeats the first two factors: "The same! Let's avoid duplicated computation". So define δ = (∂s/∂h)(∂h/∂z) = uᵀ ∘ f′(z); then ∂s/∂W = δ ∂z/∂W and ∂s/∂b = δ ∂z/∂b = δ; "δ is the upstream gradient ('error signal')" (slides 46–48).
- **Symbols:** On the slides δ is a row: δ_row = uᵀ diag(f′(z)) = (u ⊙ f′(z))ᵀ, so the Jacobian ∂s/∂b = δ_row and the shaped gradient ∇_b s = δ_rowᵀ. Notes 3's δ is a column, δ⁽ᵏ⁾ = ∇_{z⁽ᵏ⁾}F for the chosen scalar score or loss F.
- **Key points:**
  - Notes 3 calls δ⁽ᵏ⁾ "the backpropagated error calculated at z⁽ᵏ⁾" (§1.5). Through many layers, in its column convention: δ⁽ᵏ⁾ = f′(z⁽ᵏ⁾) ⊙ ((W⁽ᵏ⁾)ᵀ δ⁽ᵏ⁺¹⁾): "Error propagates from layer (k + 1) to (k)" (notes 3 §1.6, p. 8).
- **Figures:** 1.46–1.48.

### C21 The gradient with respect to a weight matrix (outer product)
- **Meaning:** ∂s/∂W = δᵀxᵀ, where "δ is upstream gradient ('error signal') at z" and "x is local input signal" (slide 51). With the slides' row δ, ∇_W s = δ_rowᵀ xᵀ.
- **Key points:**
  - As a Jacobian it would be "1 output, nm inputs: 1 by nm Jacobian? … Inconvenient" (slide 49).
  - "Why the Transposes? Hacky answer: this makes the dimensions work out! Useful trick for checking your work!"; "Each input goes to each output – you want to get outer product"; "Full explanation in the lecture notes" (slide 52).
  - "W_ij only contributes to z_i": "W₂₃ is only used to compute z₂ not z₁", so ∂z_i/∂W_ij = ∂/∂W_ij Σ_k W_ik x_k = x_j (slide 53).
  - In notes 3's layer notation, ∇_{W⁽ᵏ⁾}F = δ⁽ᵏ⁺¹⁾ (a⁽ᵏ⁾)ᵀ, with bias gradient δ⁽ᵏ⁺¹⁾: "the outer product of the error vector … and the activations forwarded" (notes 3 §1.6, p. 8).
- **Figures:** 1.49–1.53.

### C22 The shape convention vs the Jacobian form
- **Meaning:** "the shape of the gradient is the shape of the parameters!" (slide 50): for W ∈ ℝ^{m×n} (as in C06), ∇_W s ∈ ℝ^{m×n}. Slide 50 instead names W's dimensions n×m; its gradient still has W's shape. There is a "Disagreement between Jacobian form (which makes the chain rule easy) and the shape convention (which makes implementing SGD easy)": by the Jacobian, ∂s/∂b is a row vector, "But shape convention says our gradient should be a column vector because b is a column vector" (slide 54).
- **Key points:**
  - "We expect answers in the assignment to follow the shape convention. But Jacobian form is useful for computing the answers" (slide 54).
  - Two options: "Use Jacobian form as much as possible, reshape to follow the shape convention at the end", or "Always follow the shape convention", looking "at dimensions to figure out when to transpose and/or reorder terms"; "The error message δ that arrives at a hidden layer has the same dimensionality as that hidden layer"; "Use Jacobian to compute; Use shape convention to format" (slide 55).
- **Figures:** 1.49, 1.50, 1.54, 1.55.
- **Extends:** ch02:C14's "by convention, we set the gradient of any object to be the shape of that object".

### C23 Computation graphs and forward propagation
- **Meaning:** "Software represents our neural net equations as a graph": "Source nodes: inputs"; "Interior nodes: operations"; "Edges pass along result of the operation"; computing along it is "Forward Propagation" (slides 57–59).
- **Example:** the NER net as a chain: x → (· W) → Wx → (+ b) → z → f → h → (· u) → s (slides 57–59).
- **Figures:** 1.57–1.59.

### C24 Backpropagation: upstream × local = downstream
- **Meaning:** "Then go backwards along edges. Pass along gradients" (slide 60). A node "receives an 'upstream gradient'"; its "Goal is to pass on the correct 'downstream gradient'"; "Each node has a local gradient: The gradient of its output with respect to its input"; "[downstream gradient] = [upstream gradient] x [local gradient]" (slides 61–64; "Chain rule!").
- **Key points:**
  - "Multiple inputs → multiple local gradients": for z = Wx, ∂s/∂W = (∂s/∂z)(∂z/∂W) and ∂s/∂x = (∂s/∂z)(∂z/∂x) (slides 65–66).
  - "We've almost shown you backpropagation. It's taking derivatives and using the (generalized, multivariate, or matrix) chain rule"; "Other trick: We re-use derivatives computed for higher layers in computing derivatives for lower layers to minimize computation" (slide 56).
- **Figures:** 1.56, 1.60–1.66.

### C25 Gradients sum at outward branches
- **Meaning:** When a node's output feeds several nodes, the gradients coming back along the branches add (slides 67–68).
- **Example:** a = x + y, b = max(y, z), f = ab: ∂f/∂y = (∂f/∂a)(∂a/∂y) + (∂f/∂b)(∂b/∂y) (slide 68).
- **Figures:** 1.67, 1.68.

### C26 Node intuitions: + distributes, max routes, * switches
- **Meaning:** "+ 'distributes' the upstream gradient" to each summand; "max 'routes' the upstream gradient" to the larger input; "* 'switches' the upstream gradient", each input receiving it times the other input (slides 69–71).
- **Example:** f(x, y, z) = (x + y) max(y, z) at x = 1, y = 2, z = 0: forward, x + y = 3, max(y, z) = 2, f = 6. Backward from 1: * sends 2 to the + branch and 3 to the max branch; + gives 2 to x and 2 to y; max gives 3 to y and 0 to z (slides 69–71). So ∂f/∂x = 2, ∂f/∂y = 2 + 3 = 5 (summing at the branch, C25), ∂f/∂z = 0; the slides show the per-edge values, and the total for y follows from C25.
- **Figures:** 1.69–1.71.

### C27 Backprop in a general graph: all gradients at once, in reverse topological order
- **Meaning:** "Incorrect way of doing backprop": computing one gradient and then "independently" another is "Duplicated computation!"; "Correct way: Compute all the gradients at once", "Analogous to using δ when we computed gradients by hand" (slides 72–74). For a computation DAG with a single scalar output z (slide 75, "Single scalar output"): "1. Fprop: visit nodes in topological sort order", computing each node's value "given predecessors"; "2. Bprop": "initialize output gradient = 1" (∂z/∂z = 1), then "visit nodes in reverse order", computing each node's gradient "using gradient wrt successors": ∂z/∂x = Σ_{i=1}^{n} (∂z/∂y_i)(∂y_i/∂x), where {y₁, …, y_n} are the successors of x (slide 75).
- **Key points:**
  - "Done correctly, big O() complexity of fprop and bprop is the same"; "In general, our nets have regular layer-structure and so we can use matrices and Jacobians…" (slide 75).
- **Figures:** 1.72–1.75.

### C28 Automatic differentiation (and how frameworks implement backprop)
- **Meaning:** "The gradient computation can be automatically inferred from the symbolic expression of the fprop"; "Each node type needs to know how to compute its output and how to compute the gradient wrt its inputs given the gradient wrt its output" (slide 76).
- **Key points:**
  - "Modern DL frameworks (Tensorflow, PyTorch, etc.) do backpropagation for you but mainly leave layer/node writer to hand-calculate the local derivative" (slide 76).
- **Example:** slide 77's ComputationalGraph: forward() runs gate.forward() "for gate in self.graph.nodes_topologically_sorted()" and returns "loss # the final gate in the graph outputs the loss"; backward() runs gate.backward() over "reversed(self.graph.nodes_topologically_sorted())" ("little piece of backprop (chain rule applied)") and returns inputs_gradients.
- **Figures:** 1.76, 1.77.

### C29 Numeric gradient checking
- **Meaning:** "For small h (≈ 1e-4)", f′(x) ≈ (f(x + h) − f(x − h))/2h (slide 78).
- **Key points:**
  - "Easy to implement correctly"; "But approximate and very slow: You have to recompute f for every parameter of our model"; "Useful for checking your implementation"; "In the old days, we hand-wrote everything, doing this everywhere was the key test"; "Now much less needed; you can use it to check layers are correctly implemented" (slide 78).
  - Notes 3 §2.1 uses the same centered difference with ε "usually around 1e−5": each parameter element needs two forward passes, so it is "too computationally inefficient to be used directly for training"; the centered formula "has an error proportional to ε²", whereas the one-sided (f(x + ε) − f(x))/ε "is more error-prone". The slide and notes choose different step sizes (1e-4, 1e−5).
- **Figures:** 1.78.

### C30 Why learn backprop when frameworks do it
- **Meaning:** "Modern deep learning frameworks compute gradients for you!"; "But why take a class on compilers or systems when they are implemented for you? Understanding what is going on under the hood is useful!"; "Backpropagation doesn't always work perfectly out of the box"; "Understanding why is crucial for debugging and improving models" (slide 80).
- **Key points:**
  - The slide points to Karpathy's article: "The problem with Backpropagation is that it is a leaky abstraction"; one example is the "dead ReLU" problem: if a ReLU unit is "clamped to zero in the forward pass … its weights will get zero gradient" (Karpathy 2016).
  - "Example in future lecture: exploding and vanishing gradients" (slide 80).
- **Figures:** 1.80.

### C31 L2 regularization of the weights
- **Meaning:** Against overfitting ("the 'high-variance problem'"), add a penalty: J_R = J + λ Σ_{i=1}^{L} ‖W⁽ⁱ⁾‖_F, with λ "the hyper-parameter controlling how much weight the regularization term has relative to the original cost function" (notes 3 §2.2).
- **Key points:**
  - It is "penalizing weights for being too large while optimizing over the original cost function", interpretable "as the prior Bayesian belief that the optimal weights are close to zero" (notes 3 §2.2).
  - λ "must be chosen via hyperparameter-tuning": too high sets "most of the weights … too close to 0"; too low and "we fall into the domain of overfitting once again" (notes 3 §2.2).
  - "the bias terms are not regularized"; L1 regularization is "less commonly applied in practice since it leads to sparsity of parameter weights" (notes 3 §2.2).
  - Source defect: the notes write the unsquared Frobenius norm ‖W‖_F (defined as √(Σ_i Σ_j U²_ij)) while describing "the sum of the squared elements"; the standard L2 penalty is λΣ‖W⁽ⁱ⁾‖²_F.
- **Extends:** ch02:C38.

### C32 Dropout
- **Meaning:** "during training, we will randomly 'drop' with some probability (1 − p) a subset of neurons during each forward/backward pass (or equivalently, we will keep alive each neuron with a probability p). Then, during testing, we will use the full network" (notes 3 §2.3; Srivastava et al. 2014).
- **Key points:**
  - Intuition: "training exponentially many smaller networks at once and averaging over their predictions" (notes 3 §2.3).
  - "during back-propagation, we only pass gradients through neurons that were kept alive during the forward pass" (notes 3 §2.3).
  - "the expected output of a neuron during testing should be approximately the same as it was during training", so outputs must be rescaled; the factor is "left as an exercise to the reader" (notes 3 §2.3; A2 Q2(b) asks it).
  - A2 Q2(b) formalizes it: h_drop = γ d ⊙ h, where d ∈ {0,1}^{D_h} is a mask that is 0 with probability p_drop ("dropping different units each minibatch") and γ "is chosen such that the expected value of h_drop is h": E_{p_drop}[h_drop]_i = h_i. The value of γ is the exercise.

### C33 Data preprocessing (mean subtraction, normalization, whitening)
- **Meaning:** "zero-center the data by subtracting the mean feature vector", with "the mean … calculated only across the training set" and subtracted from training, validation and test sets; normalization, "dividing the features by their respective standard deviation calculated across the training set"; and whitening, which "converts the data to a have an identity covariance matrix" (notes 3 §2.5).
- **Key points:**
  - Source defect in the whitening recipe: for centered samples in rows, X_c = USVᵀ with covariance X_cᵀX_c/N, whitening is Z = X_c V diag(√N/s_i) on the nonzero singular directions; the notes' "compute UX′" and divide by s_i alone omit the rotation by V and give covariance I/N. Equivalently, diagonalize the covariance and divide the rotated coordinates by the square roots of its eigenvalues.

### C34 Parameter initialization (Xavier/Glorot)
- **Meaning:** Start with "small random numbers normally distributed around 0"; for sigmoid and tanh units, Glorot and Bengio (2010) find "faster convergence and lower error rates" with W ~ U[−√(6/(n⁽ˡ⁾ + n⁽ˡ⁺¹⁾)), √(6/(n⁽ˡ⁾ + n⁽ˡ⁺¹⁾))], fan-in n⁽ˡ⁾ and fan-out n⁽ˡ⁺¹⁾, biases initialized to 0 (notes 3 §2.6).
- **Key points:**
  - It "attempts to maintain activation variances as well as backpropagated gradient variances across layers. Without such initialization, the gradient variances (which are a proxy for information) generally decrease with backpropagation across layers" (notes 3 §2.6).
  - The notes write "Xavier et al", using the first author's given name.

### C35 Learning-rate strategies
- **Meaning:** α "must be tuned effectively": with "very large learning rates, we might experience that the loss function actually diverges" (Fig. 15); "too small a learning rate" may not "converge in a reasonable amount of time, or might get caught in local minima" (notes 3 §2.7).
- **Key points:**
  - Collobert's scaling of a weight's rate "by the inverse square root of the fan-in of the neuron"; annealing, reducing α "by a factor x after every n iterations"; exponential decay α(t) = α₀e^{−kt}; and α(t) = α₀τ/max(t, τ), which "has been found to work quite well" (notes 3 §2.7).
- **Extends:** ch02:C15.

### C36 Momentum updates
- **Meaning:** "inspired by the study of dynamics and motion in physics", momentum uses the "velocity" of updates: v = mu*v − alpha*grad_x; x += v (notes 3 §2.8, Snippet 2.2).

### C37 Adaptive optimizers (AdaGrad, RMSProp, Adam)
- **Meaning:** AdaGrad lets "the learning rate … vary for each parameter": θ_{t,i} = θ_{t−1,i} − α g_{t,i}/√(Σ_{τ=1}^{t} g²_{τ,i}); "parameters that have not been updated much in the past are likelier to have higher learning rates now" (snippet: cache += dx**2; x += −learning_rate*dx/np.sqrt(cache + 1e−8)). RMSProp keeps a moving average: cache = ρ·cache + (1 − ρ)g², θ ← θ − αg/(√cache + ε) (Snippet 2.4). Adam "is in turn a variant of RMSProp, but with the addition of momentum-like updates": m = beta1*m + (1−beta1)*dx; v = beta2*v + (1−beta2)*(dx**2); x += −learning_rate*m/(np.sqrt(v) + eps) (notes 3 §2.9, "courtesy of Andrej Karpathy").
- **Key points:**
  - AdaGrad's effective per-parameter learning rate is nonincreasing (the accumulated cache only grows), though the actual update magnitudes need not be; RMSProp's effective rate need not be nonincreasing. The notes' phrasing, "unlike AdaGrad, its updates do not become monotonically smaller", is about this effective rate.
  - The simplified Adam omits Adam's bias correction (Kingma and Ba 2015); A2's footnote 7: "The actual Adam update uses a few additional tricks".
  - A2 Q2(a) builds Adam in two steps: momentum, m_{t+1} ← β₁m_t + (1 − β₁)∇J_minibatch(θ_t), θ_{t+1} ← θ_t − αm_{t+1}, with β₁ "often set to 0.9"; then adaptive learning rates, v_{t+1} ← β₂v_t + (1 − β₂)(∇J ⊙ ∇J), θ_{t+1} ← θ_t − αm_{t+1}/√v_{t+1}, with β₂ "often set to 0.99".

### C38 Transition-based dependency parsing
- **Meaning:** "A dependency parser analyzes the grammatical structure of a sentence, establishing relationships between head words, and words which modify those heads"; a transition-based parser "incrementally builds up a parse one step at a time" (A2 §3).
- **Key points:**
  - A partial parse holds "A stack of words that are currently being processed", "A buffer of words yet to be processed" and "A list of dependencies predicted by the parser"; initially the stack is [ROOT], the dependency list is empty and the buffer holds the sentence in order; transitions apply "until its buffer is empty and the stack size is 1" (A2 §3).
  - The transitions, with "first" meaning the stack top (the most recently pushed item, the last element of the displayed list) and "second" the item below it: SHIFT moves the first buffer word onto the stack; LEFT-ARC makes the second item a dependent of the first and removes the second (first → second); RIGHT-ARC makes the first item a dependent of the second and removes the first (second → first) (A2 §3).
  - Other kinds are named: "graph-based parsers, and feature-based parsers" (A2 §3).
  - Algorithm 1 (minibatch parsing, A2 Q3(d)): initialize a PartialParse for each sentence and an "unfinished parses" list referencing them; while it is not empty, take the first batch_size parses as a minibatch, use the model to predict one transition for each, apply it, and remove the completed parses (empty buffer, stack of size 1); return the dependencies for each parse, in the original sentence order.
- **Example:** "I presented my findings at the NLP conference", with its dependency tree; the supplied first steps are SHIFT, SHIFT, then LEFT-ARC adding presented → I (A2 Q3(a)); the remaining steps are the exercise.
- **Figures:** 3.5.

### C39 A2's neural transition classifier (after Chen and Manning 2014)
- **Meaning:** A2's adaptation, using the feature set of "A Fast and Accurate Dependency Parser using Neural Networks": features are "a list of tokens (e.g., the last word in the stack, first word in the buffer, dependent of the second-to-last word in the stack if there is one, etc.)", w = [w₁, …, w_m]; x = [E_{w₁}, …, E_{w_m}] ∈ ℝ^{dm}, with E ∈ ℝ^{\|V\|×d} (each row a word's vector); h = ReLU(xW + b₁), l = hU + b₂, ŷ = softmax(l); trained to minimize cross-entropy CE(y, ŷ) = −Σ_j y_j log ŷ_j, averaged over training examples (A2 Q3(e)). The original paper's classifier uses a cubic activation and labeled transitions; A2 uses ReLU and three unlabeled actions.
- **Key points:**
  - Implement the linear and embedding layers yourself: "Please DO NOT use torch.nn.Linear or torch.nn.Embedding" (A2 Q3(e)).
  - The starter code: parser_model.py inserts dropout "after ReLU function" and returns logits, "because" softmax "is included in the loss function nn.CrossEntropyLoss"; it asks for weights initialized with nn.init.xavier_uniform_ and biases with nn.init.uniform_. run.py asks for an Adam optimizer and cross-entropy with "mean" reduction, with model.train() and model.eval() switching dropout on and off.
  - Targets: in debug mode "a loss smaller than 0.2 and a UAS larger than 65"; in full training "a loss smaller than 0.08 on the train set and an Unlabeled Attachment Score larger than 87 on the dev set"; "the model in the original neural dependency parsing paper gets 92.5 UAS" (A2 Q3(e)).

### C40 Unlabeled attachment score (UAS)
- **Meaning:** "the ratio between number of correctly predicted dependencies and the number of total dependencies despite of the relations (our model doesn't predict this)" (A2 Q3(e)iii).

## Problems

What each end-of-chapter problem asks and which concepts it exercises. Record no solutions the source doesn't give.

| Problem | Asks | Exercises |
|---|---|---|
| A2 Q1(a) | show the naïve-softmax loss equals the cross-entropy between one-hot y and ŷ | C11; ch02:C12 |
| A2 Q1(b) | ∂J/∂v_c in vectorized, shape-convention form; when it is zero; interpret its two terms | C13, C22; ch02:C14 |
| A2 Q1(c) | when L2-normalizing embeddings loses useful information for a sum-based binary sentiment classifier (hint: u_x = αu_y) | ch02:C07 |
| A2 Q1(d) | ∂J/∂U, by columns | C22 |
| A2 Q1(e) | ∂J/∂u_w for w = o and w ≠ o | C16; ch02:C12 |
| A2 Q1(f) | the derivative of Leaky ReLU, f(x) = max(αx, x) | C07 |
| A2 Q1(g) | σ′(x) in terms of σ(x) | C07, B02 |
| A2 Q2(a) | why momentum reduces update variance; which parameters Adam's division by √v gives larger updates | C36, C37 |
| A2 Q2(b) | the dropout scale γ in terms of p_drop; why dropout is applied in training but not evaluation | C32 |
| A2 Q3(a) | the transition sequence for "I presented my findings at the NLP conference" (first three steps given) | C38 |
| A2 Q3(b) | how many steps parse a sentence of n words | C38 |
| A2 Q3(c)–(d) | implement PartialParse's init and parse_step, and minibatch parsing (Algorithm 1) | C38 |
| A2 Q3(e) | implement and train the parser; ∂h_i/∂x_j for h = ReLU(xW + b₁); ∂CE/∂l_i; report dev and test UAS | C39, C40, C07, C11 |

A2 stores each word's u_w and v_w as columns of U and V; notes 1 (ch02) stores them as rows.

## Forward references

Topics the chapter defers. None is taught here; later extraction EXTENDs from these hooks.

| Topic | Chapter | Claim / promised treatment |
|---|---|---|
| Exploding and vanishing gradients | a future lecture (slide 80 names none; RNNs come in Lecture 4) | "Example in future lecture: exploding and vanishing gradients" (slide 80) |

## Studs

Concepts the chapter uses but does not teach. A stud means "not taught here", not "nothing more to learn".

<!-- Entry shape (numbering runs B01, B02, …):
 ### B01 <Label>
- **Meaning:** one line.
- **Used at:** §3.1 "short quote"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.
  (Book mode, overlay §8: when an earlier chapter's map teaches it, write `Taught in ch02:C05` instead.)
-->

### B01 Matrix–vector multiplication
- **Meaning:** Wx, the product of a matrix and a vector.
- **Used at:** slide 13 "x = Le"; slide 14 "Wx + b"; notes 3 §1.2.
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B02 The sigmoid (logistic) function
- **Meaning:** σ(x) = 1/(1 + e^{−x}).
- **Used at:** slide 14 "J_t(θ) = σ(s)"; notes 3 §1.1.
- **Status:** Taught in ch02:C21.

### B03 The softmax function
- **Meaning:** softmax(xᵢ) = exp(xᵢ)/Σⱼ exp(xⱼ).
- **Used at:** slide 13 "Typical ML/stats softmax classifier".
- **Status:** Taught in ch02:C13.

### B04 Binary logistic regression
- **Meaning:** a classifier giving the probability of one of two classes through the sigmoid of a score.
- **Used at:** slide 11 "Train logistic classifier".
- **Status:** not taught here (ch02:B07 was also a stud). Further prerequisites are not assessed in this chapter.

### B05 One-hot vectors
- **Meaning:** a vector with a single 1, one dimension per vocabulary word.
- **Used at:** slide 13 "re-represent one-hot vectors"; "embedding layer: x = Le".
- **Status:** Taught in ch02:C04.

### B06 The standard normal distribution and its CDF
- **Meaning:** X ~ N(0,1); P(X ≤ x), the probability that X is at most x.
- **Used at:** slide 16 "GELU(x) = x · P(X ≤ x), X ~ N(0,1)".
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B07 Single-variable derivatives
- **Meaning:** the derivative of a function of one variable and its rules.
- **Used at:** notes 3 §2.4 σ′(z), tanh′(z), ReLU′.
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B08 Stochastic gradient descent
- **Meaning:** θ^{new} = θ^{old} − α∇_θJ(θ) on sampled data; α = step size or learning rate.
- **Used at:** slide 19 "Remember: Stochastic Gradient Descent".
- **Status:** Taught in ch02:C16.

### B09 Logarithms and probability distributions
- **Meaning:** log and its rules; a distribution over C classes.
- **Used at:** slide 18 H(p, q) = −Σ p(c) log q(c).
- **Status:** not taught here (ch02:B02 and ch02:B03 were also studs). Further prerequisites are not assessed in this chapter.

### B10 Partial derivatives
- **Meaning:** the derivative of a multivariable function with respect to one input, the others held fixed.
- **Used at:** slides 22–23.
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B11 Identity and diagonal matrices
- **Meaning:** I, with 1s on the diagonal; diag(v), with v on the diagonal and 0s elsewhere.
- **Used at:** slide 29 diag(f′(z)); slide 33 "I (Identity matrix)".
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B12 Outer product of two vectors
- **Meaning:** abᵀ, the matrix of all products a_i b_j.
- **Used at:** slides 51–52 δᵀxᵀ, "you want to get outer product".
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B13 Directed acyclic graphs and topological order
- **Meaning:** a graph with directed edges and no cycles; an ordering in which every node comes after its predecessors.
- **Used at:** slide 75 "visit nodes in topological sort order".
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B14 Overfitting and high variance
- **Meaning:** fitting the training data so closely that the model fails to generalize.
- **Used at:** notes 3 §2.2 "the 'high-variance problem'".
- **Status:** not taught here (ch02:B14 was also a stud). Further prerequisites are not assessed in this chapter.

### B15 Random distributions and summary statistics (uniform, normal; mean, variance, standard deviation, covariance)
- **Meaning:** U[a, b] and N(μ, σ²); expectation; the mean, variance and standard deviation of a feature, and the covariance matrix of data.
- **Used at:** notes 3 §2.5 (mean, standard deviation, covariance), §2.6 (normal and uniform initialization; variances); A2 Q2(b) (the dropout mask's probabilities and expected activation).
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B16 Singular value decomposition
- **Meaning:** X = USVᵀ.
- **Used at:** notes 3 §2.5 whitening.
- **Status:** Taught in ch02:C24.

### B17 Stacks and buffers (queues)
- **Meaning:** last-in-first-out and first-in-first-out containers.
- **Used at:** A2 §3 "A stack of words", "A buffer of words".
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

## Prerequisites

Every row reads "From `prerequisite_for` To"; a learner walks backward from To. The prerequisite rows must form
no cycle. A **scope** means the prerequisite is needed only for that part of To.

| From | Predicate | To | Family | Status | Scope | Basis |
|---|---|---|---|---|---|---|
| C01 Named entity recognition as token labelling | `prerequisite_for` | C02 Window classification with a binary logistic classifier per class | prerequisite | stated | — | slide 11 "Simple NER" |
| C02 Window classification with a binary logistic classifier per class | `prerequisite_for` | C06 A one-hidden-layer feed-forward network (the NER scorer) | prerequisite | stated | its input x ∈ ℝ^{5d} | slide 14 x = [x_museums … x_amazing] |
| C04 The embedding layer (x = Le) | `prerequisite_for` | C03 Neural network classifier vs the linear softmax classifier | prerequisite | stated | — | slide 13 |
| C04 The embedding layer (x = Le) | `prerequisite_for` | C06 A one-hidden-layer feed-forward network (the NER scorer) | prerequisite | stated | — | slide 14 "Embedding of 1-hot words" |
| C05 The artificial neuron (sigmoid unit) | `prerequisite_for` | C06 A one-hidden-layer feed-forward network (the NER scorer) | prerequisite | stated | — | notes 3 §1.2 "multiple such neurons" |
| B01 Matrix–vector multiplication | `prerequisite_for` | C04 The embedding layer (x = Le) | prerequisite | stated | — | slide 13 x = Le |
| B01 Matrix–vector multiplication | `prerequisite_for` | C05 The artificial neuron (sigmoid unit) | prerequisite | stated | — | notes 3 §1.1 wᵀx |
| B01 Matrix–vector multiplication | `prerequisite_for` | C06 A one-hidden-layer feed-forward network (the NER scorer) | prerequisite | stated | — | slide 14 Wx + b |
| B02 The sigmoid (logistic) function | `prerequisite_for` | C05 The artificial neuron (sigmoid unit) | prerequisite | stated | — | notes 3 §1.1 |
| B02 The sigmoid (logistic) function | `prerequisite_for` | C06 A one-hidden-layer feed-forward network (the NER scorer) | prerequisite | stated | σ(s) | slide 14 |
| B03 The softmax function | `prerequisite_for` | C03 Neural network classifier vs the linear softmax classifier | prerequisite | stated | — | slide 13 |
| B04 Binary logistic regression | `prerequisite_for` | C02 Window classification with a binary logistic classifier per class | prerequisite | stated | — | slide 11 |
| B05 One-hot vectors | `prerequisite_for` | C04 The embedding layer (x = Le) | prerequisite | stated | the column-lookup reading | slide 13 "re-represent one-hot vectors" |
| C06 A one-hidden-layer feed-forward network (the NER scorer) | `prerequisite_for` | C03 Neural network classifier vs the linear softmax classifier | prerequisite | inferred | the deep, non-linear claim | slide 13 "We use deep networks"; slide 14's network |
| C06 A one-hidden-layer feed-forward network (the NER scorer) | `prerequisite_for` | C07 Classic activation functions (logistic, tanh, ReLU, leaky and parametric ReLU) | prerequisite | stated | — | slide 17 "Non-linearities (i.e., 'f' on previous slide)"; slide 14 |
| B02 The sigmoid (logistic) function | `prerequisite_for` | C07 Classic activation functions (logistic, tanh, ReLU, leaky and parametric ReLU) | prerequisite | stated | — | slide 15 |
| B07 Single-variable derivatives | `prerequisite_for` | C07 Classic activation functions (logistic, tanh, ReLU, leaky and parametric ReLU) | prerequisite | stated | notes 3's gradients | notes 3 §2.4 |
| C07 Classic activation functions (logistic, tanh, ReLU, leaky and parametric ReLU) | `prerequisite_for` | C08 Smooth modern activations (GELU, SiLU, Swish) | prerequisite | stated | — | slides 15–16 "Non-linearities, old and new" |
| B02 The sigmoid (logistic) function | `prerequisite_for` | C08 Smooth modern activations (GELU, SiLU, Swish) | prerequisite | stated | — | slide 16 x · σ(x) |
| B06 The standard normal distribution and its CDF | `prerequisite_for` | C08 Smooth modern activations (GELU, SiLU, Swish) | prerequisite | stated | GELU | slide 16 |
| C07 Classic activation functions (logistic, tanh, ReLU, leaky and parametric ReLU) | `prerequisite_for` | C09 Gated linear units (GLU, SwiGLU) | prerequisite | stated | — | slide 16 |
| C08 Smooth modern activations (GELU, SiLU, Swish) | `prerequisite_for` | C09 Gated linear units (GLU, SwiGLU) | prerequisite | stated | SwiGLU's Swish | slide 16 |
| B01 Matrix–vector multiplication | `prerequisite_for` | C09 Gated linear units (GLU, SwiGLU) | prerequisite | stated | xV, xW | slide 16 |
| C06 A one-hidden-layer feed-forward network (the NER scorer) | `prerequisite_for` | C10 Why networks need non-linearities | prerequisite | stated | — | slide 17 |
| B01 Matrix–vector multiplication | `prerequisite_for` | C10 Why networks need non-linearities | prerequisite | stated | W₁W₂x = Wx | slide 17 |
| B09 Logarithms and probability distributions | `prerequisite_for` | C11 Cross-entropy loss | prerequisite | stated | — | slide 18 |
| B03 The softmax function | `prerequisite_for` | C11 Cross-entropy loss | prerequisite | inferred | q from a softmax | slide 13's p(y \| x); slide 18 "our computed model probability" |
| C06 A one-hidden-layer feed-forward network (the NER scorer) | `prerequisite_for` | C12 The max-margin objective | prerequisite | stated | — | notes 3 §1.4 s = Uᵀf(Wx + b) |
| C02 Window classification with a binary logistic classifier per class | `prerequisite_for` | C12 The max-margin objective | prerequisite | stated | true vs corrupt windows | notes 3 §1.4 "Museums in Paris are amazing" |
| B07 Single-variable derivatives | `prerequisite_for` | C14 The gradient (of a scalar function of n inputs) | prerequisite | stated | — | slide 21 |
| B10 Partial derivatives | `prerequisite_for` | C14 The gradient (of a scalar function of n inputs) | prerequisite | stated | — | slide 22 |
| C14 The gradient (of a scalar function of n inputs) | `prerequisite_for` | C15 The Jacobian matrix | prerequisite | stated | — | slide 23 "Generalization of the Gradient" |
| B10 Partial derivatives | `prerequisite_for` | C15 The Jacobian matrix | prerequisite | stated | — | slide 23 |
| C15 The Jacobian matrix | `prerequisite_for` | C16 The multivariable chain rule (multiply Jacobians) | prerequisite | stated | — | slide 24 "multiply Jacobians" |
| B07 Single-variable derivatives | `prerequisite_for` | C16 The multivariable chain rule (multiply Jacobians) | prerequisite | stated | — | slide 24 one-variable case |
| B01 Matrix–vector multiplication | `prerequisite_for` | C16 The multivariable chain rule (multiply Jacobians) | prerequisite | stated | z = Wx + b | slide 24 |
| C15 The Jacobian matrix | `prerequisite_for` | C17 The Jacobian of an elementwise activation | prerequisite | stated | — | slide 29 "definition of Jacobian" |
| C07 Classic activation functions (logistic, tanh, ReLU, leaky and parametric ReLU) | `prerequisite_for` | C17 The Jacobian of an elementwise activation | prerequisite | stated | f′ | slide 29 f′(z_i) |
| B11 Identity and diagonal matrices | `prerequisite_for` | C17 The Jacobian of an elementwise activation | prerequisite | stated | — | slide 29 diag(f′(z)) |
| C15 The Jacobian matrix | `prerequisite_for` | C18 Useful Jacobians of the affine map and the score | prerequisite | stated | — | slides 30–33 |
| B01 Matrix–vector multiplication | `prerequisite_for` | C18 Useful Jacobians of the affine map and the score | prerequisite | stated | — | slide 33 Wx + b |
| B11 Identity and diagonal matrices | `prerequisite_for` | C18 Useful Jacobians of the affine map and the score | prerequisite | stated | — | slide 33 I |
| C14 The gradient (of a scalar function of n inputs) | `prerequisite_for` | C13 Vectorized gradients by hand (matrix calculus) | prerequisite | stated | — | slide 19 ∇_θJ(θ) |
| B08 Stochastic gradient descent | `prerequisite_for` | C13 Vectorized gradients by hand (matrix calculus) | prerequisite | stated | what the gradient is for | slide 19 |
| C06 A one-hidden-layer feed-forward network (the NER scorer) | `prerequisite_for` | C19 The worked gradient of the NER score (break up, chain rule, Jacobians) | prerequisite | stated | — | slide 34 "Back to our Neural Net!" |
| C16 The multivariable chain rule (multiply Jacobians) | `prerequisite_for` | C19 The worked gradient of the NER score (break up, chain rule, Jacobians) | prerequisite | stated | — | slides 37–40 |
| C17 The Jacobian of an elementwise activation | `prerequisite_for` | C19 The worked gradient of the NER score (break up, chain rule, Jacobians) | prerequisite | stated | — | slide 45 "Useful Jacobians from previous slide" |
| C18 Useful Jacobians of the affine map and the score | `prerequisite_for` | C19 The worked gradient of the NER score (break up, chain rule, Jacobians) | prerequisite | stated | — | slide 45 |
| C19 The worked gradient of the NER score (break up, chain rule, Jacobians) | `prerequisite_for` | C20 Reusing computation: the upstream gradient δ ("error signal") | prerequisite | stated | — | slides 46–48 |
| C20 Reusing computation: the upstream gradient δ ("error signal") | `prerequisite_for` | C21 The gradient with respect to a weight matrix (outer product) | prerequisite | stated | — | slide 51 |
| B12 Outer product of two vectors | `prerequisite_for` | C21 The gradient with respect to a weight matrix (outer product) | prerequisite | stated | — | slide 52 |
| C15 The Jacobian matrix | `prerequisite_for` | C21 The gradient with respect to a weight matrix (outer product) | prerequisite | stated | the 1 × nm Jacobian remark | slide 49 |
| C21 The gradient with respect to a weight matrix (outer product) | `prerequisite_for` | C22 The shape convention vs the Jacobian form | prerequisite | stated | — | slide 50 |
| C13 Vectorized gradients by hand (matrix calculus) | `prerequisite_for` | C22 The shape convention vs the Jacobian form | prerequisite | stated | — | slide 54 |
| C06 A one-hidden-layer feed-forward network (the NER scorer) | `prerequisite_for` | C23 Computation graphs and forward propagation | prerequisite | stated | — | slides 57–59, the NER equations as a graph |
| B13 Directed acyclic graphs and topological order | `prerequisite_for` | C23 Computation graphs and forward propagation | prerequisite | inferred | general graphs | slide 75 |
| C23 Computation graphs and forward propagation | `prerequisite_for` | C24 Backpropagation: upstream × local = downstream | prerequisite | stated | — | slide 60 |
| C16 The multivariable chain rule (multiply Jacobians) | `prerequisite_for` | C24 Backpropagation: upstream × local = downstream | prerequisite | stated | — | slide 63 "Chain rule!" |
| C24 Backpropagation: upstream × local = downstream | `prerequisite_for` | C25 Gradients sum at outward branches | prerequisite | stated | — | slides 67–68 |
| B10 Partial derivatives | `prerequisite_for` | C25 Gradients sum at outward branches | prerequisite | stated | — | slide 68 |
| C24 Backpropagation: upstream × local = downstream | `prerequisite_for` | C26 Node intuitions: + distributes, max routes, * switches | prerequisite | stated | — | slides 69–71 |
| C25 Gradients sum at outward branches | `prerequisite_for` | C26 Node intuitions: + distributes, max routes, * switches | prerequisite | stated | — | y feeds both + and max |
| C24 Backpropagation: upstream × local = downstream | `prerequisite_for` | C27 Backprop in a general graph: all gradients at once, in reverse topological order | prerequisite | stated | — | slides 72–75 |
| C25 Gradients sum at outward branches | `prerequisite_for` | C27 Backprop in a general graph: all gradients at once, in reverse topological order | prerequisite | stated | — | slide 75 Σ over successors |
| B13 Directed acyclic graphs and topological order | `prerequisite_for` | C27 Backprop in a general graph: all gradients at once, in reverse topological order | prerequisite | stated | — | slide 75 |
| C20 Reusing computation: the upstream gradient δ ("error signal") | `prerequisite_for` | C27 Backprop in a general graph: all gradients at once, in reverse topological order | prerequisite | stated | the δ analogy | slide 74 "Analogous to using δ" |
| C27 Backprop in a general graph: all gradients at once, in reverse topological order | `prerequisite_for` | C28 Automatic differentiation (and how frameworks implement backprop) | prerequisite | stated | — | slide 77 topological order and its reverse |
| C23 Computation graphs and forward propagation | `prerequisite_for` | C28 Automatic differentiation (and how frameworks implement backprop) | prerequisite | stated | — | slide 76 "the symbolic expression of the fprop" |
| C14 The gradient (of a scalar function of n inputs) | `prerequisite_for` | C29 Numeric gradient checking | prerequisite | stated | — | slide 78 |
| B07 Single-variable derivatives | `prerequisite_for` | C29 Numeric gradient checking | prerequisite | stated | — | notes 3 §2.1 "the definition of the derivative" |
| C28 Automatic differentiation (and how frameworks implement backprop) | `prerequisite_for` | C30 Why learn backprop when frameworks do it | prerequisite | stated | — | slide 80 |
| C24 Backpropagation: upstream × local = downstream | `prerequisite_for` | C30 Why learn backprop when frameworks do it | prerequisite | stated | — | slide 80 |
| C07 Classic activation functions (logistic, tanh, ReLU, leaky and parametric ReLU) | `prerequisite_for` | C30 Why learn backprop when frameworks do it | prerequisite | stated | Karpathy's dead-ReLU example | Karpathy 2016 |
| B14 Overfitting and high variance | `prerequisite_for` | C31 L2 regularization of the weights | prerequisite | stated | — | notes 3 §2.2 |
| C11 Cross-entropy loss | `prerequisite_for` | C31 L2 regularization of the weights | prerequisite | inferred | the loss J | notes 3 §2.2 "append an extra term to our loss function J" |
| C06 A one-hidden-layer feed-forward network (the NER scorer) | `prerequisite_for` | C32 Dropout | prerequisite | stated | — | notes 3 §2.3 "output h of each layer of neurons" |
| B14 Overfitting and high variance | `prerequisite_for` | C32 Dropout | prerequisite | stated | — | notes 3 §2.3 "less likely to overfit" |
| B15 Random distributions and summary statistics (uniform, normal; mean, variance, standard deviation, covariance) | `prerequisite_for` | C33 Data preprocessing (mean subtraction, normalization, whitening) | prerequisite | stated | — | notes 3 §2.5 |
| B16 Singular value decomposition | `prerequisite_for` | C33 Data preprocessing (mean subtraction, normalization, whitening) | prerequisite | stated | whitening | notes 3 §2.5 |
| C06 A one-hidden-layer feed-forward network (the NER scorer) | `prerequisite_for` | C34 Parameter initialization (Xavier/Glorot) | prerequisite | stated | — | notes 3 §2.6 |
| C07 Classic activation functions (logistic, tanh, ReLU, leaky and parametric ReLU) | `prerequisite_for` | C34 Parameter initialization (Xavier/Glorot) | prerequisite | stated | sigmoid and tanh units | notes 3 §2.6 |
| B15 Random distributions and summary statistics (uniform, normal; mean, variance, standard deviation, covariance) | `prerequisite_for` | C34 Parameter initialization (Xavier/Glorot) | prerequisite | stated | — | notes 3 §2.6 |
| B08 Stochastic gradient descent | `prerequisite_for` | C35 Learning-rate strategies | prerequisite | stated | — | notes 3 §2.7 |
| B08 Stochastic gradient descent | `prerequisite_for` | C36 Momentum updates | prerequisite | stated | — | notes 3 §2.8 "a variant of gradient descent" |
| B08 Stochastic gradient descent | `prerequisite_for` | C37 Adaptive optimizers (AdaGrad, RMSProp, Adam) | prerequisite | stated | — | notes 3 §2.9 |
| C35 Learning-rate strategies | `prerequisite_for` | C37 Adaptive optimizers (AdaGrad, RMSProp, Adam) | prerequisite | stated | — | notes 3 §2.9 "the learning rate can vary for each parameter" |
| C36 Momentum updates | `prerequisite_for` | C37 Adaptive optimizers (AdaGrad, RMSProp, Adam) | prerequisite | stated | Adam | notes 3 §2.9 "momentum-like updates" |
| B17 Stacks and buffers (queues) | `prerequisite_for` | C38 Transition-based dependency parsing | prerequisite | stated | — | A2 §3 |
| C38 Transition-based dependency parsing | `prerequisite_for` | C39 A2's neural transition classifier (after Chen and Manning 2014) | prerequisite | stated | — | A2 §3 "decide among the three transitions using a neural network classifier" |
| C04 The embedding layer (x = Le) | `prerequisite_for` | C39 A2's neural transition classifier (after Chen and Manning 2014) | prerequisite | stated | — | A2 Q3(e) "looks up an embedding for each word" |
| C06 A one-hidden-layer feed-forward network (the NER scorer) | `prerequisite_for` | C39 A2's neural transition classifier (after Chen and Manning 2014) | prerequisite | stated | — | A2 Q3(e) "the hidden layer" |
| C07 Classic activation functions (logistic, tanh, ReLU, leaky and parametric ReLU) | `prerequisite_for` | C39 A2's neural transition classifier (after Chen and Manning 2014) | prerequisite | stated | ReLU | A2 Q3(e) |
| C11 Cross-entropy loss | `prerequisite_for` | C39 A2's neural transition classifier (after Chen and Manning 2014) | prerequisite | stated | — | A2 Q3(e) |
| B03 The softmax function | `prerequisite_for` | C39 A2's neural transition classifier (after Chen and Manning 2014) | prerequisite | stated | — | A2 Q3(e) ŷ = softmax(l) |
| C32 Dropout | `prerequisite_for` | C39 A2's neural transition classifier (after Chen and Manning 2014) | prerequisite | stated | the starter code's dropout layer | parser_model.py |
| C34 Parameter initialization (Xavier/Glorot) | `prerequisite_for` | C39 A2's neural transition classifier (after Chen and Manning 2014) | prerequisite | stated | the starter code's Xavier-uniform weights | parser_model.py |
| C37 Adaptive optimizers (AdaGrad, RMSProp, Adam) | `prerequisite_for` | C39 A2's neural transition classifier (after Chen and Manning 2014) | prerequisite | stated | the starter code's Adam optimizer | run.py |
| C38 Transition-based dependency parsing | `prerequisite_for` | C40 Unlabeled attachment score (UAS) | prerequisite | stated | — | A2 Q3(e)iii |
| B15 Random distributions and summary statistics (uniform, normal; mean, variance, standard deviation, covariance) | `prerequisite_for` | C32 Dropout | prerequisite | stated | the mask's probabilities and expected activation | A2 Q2(b) E_{p_drop}[h_drop]_i = h_i |

<!-- Row shape: | B01 Label | `prerequisite_for` | C01 Label | prerequisite | inferred | — | §3.1 "short quote" | -->

### Per-concept review

| Concept | Direct prerequisites and notes |
|---|---|
| C01 Named entity recognition as token labelling | none in this lecture (Lecture 2: ch02:C33) |
| C02 Window classification | C01; B04 |
| C03 Neural network classifier | C04; B03; C06 (the deep, non-linear claim) |
| C04 The embedding layer | B01; B05 (column lookup) |
| C05 The artificial neuron | B01; B02 |
| C06 One-hidden-layer network | C02 (input x); C04; C05; B01; B02 (σ(s)) |
| C07 Classic activation functions | C06; B02; B07 (gradients) |
| C08 Smooth modern activations | C07; B02; B06 (GELU) |
| C09 Gated linear units | C07; C08 (SwiGLU); B01 |
| C10 Why networks need non-linearities | C06; B01 |
| C11 Cross-entropy loss | B09; B03 (q from a softmax) |
| C12 The max-margin objective | C06; C02 (true vs corrupt windows) |
| C13 Vectorized gradients by hand | C14; B08 (what the gradient is for) |
| C14 The gradient | B07; B10 |
| C15 The Jacobian matrix | C14; B10 |
| C16 The multivariable chain rule | C15; B07; B01 |
| C17 Jacobian of an elementwise activation | C15; C07 (f′); B11 |
| C18 Useful Jacobians | C15; B01; B11 |
| C19 The worked gradient of the NER score | C06; C16; C17; C18 |
| C20 Reusing computation: δ | C19 |
| C21 Gradient with respect to a weight matrix | C20; B12; C15 (1 × nm remark) |
| C22 Shape convention vs Jacobian form | C21; C13 |
| C23 Computation graphs and forward propagation | C06; B13 (general graphs) |
| C24 Backpropagation | C23; C16 |
| C25 Gradients sum at outward branches | C24; B10 |
| C26 Node intuitions | C24; C25 |
| C27 Backprop in a general graph | C24; C25; B13; C20 (δ analogy) |
| C28 Automatic differentiation | C27; C23 |
| C29 Numeric gradient checking | C14; B07 |
| C30 Why learn backprop | C28; C24; C07 (dead ReLU) |
| C31 L2 regularization | B14; C11 (the loss J) |
| C32 Dropout | C06; B14; B15 (mask probabilities, expected activation) |
| C33 Data preprocessing | B15; B16 (whitening) |
| C34 Parameter initialization | C06; C07 (sigmoid/tanh); B15 |
| C35 Learning-rate strategies | B08 |
| C36 Momentum updates | B08 |
| C37 Adaptive optimizers | B08; C35; C36 (Adam) |
| C38 Transition-based dependency parsing | B17 |
| C39 A2's neural transition classifier | C38; C04; C06; C07 (ReLU); C11; B03; C32, C34, C37 (starter-code requirements) |
| C40 Unlabeled attachment score | C38 |

## Predicates

The controlled list is `predicates.md` in this pack. List here only predicates this map ADDED with the
Reviewer's agreement, each with its family and meaning.

## Links

Functional links from the text, each marked **stated** (the author says it) or **inferred** (our reading, with its reason).

| From | Predicate | To | Family | Status | Scope / original wording | Basis |
|---|---|---|---|---|---|---|
| C02 Window classification with a binary logistic classifier per class | `applies_to` | C01 Named entity recognition as token labelling | functional | stated | "Simple NER: Window classification using binary logistic classifier" | slide 11 |
| C06 A one-hidden-layer feed-forward network (the NER scorer) | `illustrates` | C03 Neural network classifier vs the linear softmax classifier | functional | inferred | slide 14 is the lecture's instance of a neural classifier | slides 13–14 |
| C06 A one-hidden-layer feed-forward network (the NER scorer) | `has_part` | C05 The artificial neuron (sigmoid unit) | part-whole | stated | "multiple such neurons" | notes 3 §1.2 |
| C06 A one-hidden-layer feed-forward network (the NER scorer) | `uses` | C04 The embedding layer (x = Le) | functional | stated | "Embedding of 1-hot words" | slide 14 |
| C09 Gated linear units (GLU, SwiGLU) | `uses` | B02 The sigmoid (logistic) function | functional | stated | GLU's σ(xW + b) | slide 16 |
| C09 Gated linear units (GLU, SwiGLU) | `uses` | C08 Smooth modern activations (GELU, SiLU, Swish) | functional | stated | SwiGLU's Swish_β(xW + b) | slide 16 |
| C08 Smooth modern activations (GELU, SiLU, Swish) | `contrasts_with` | C07 Classic activation functions (logistic, tanh, ReLU, leaky and parametric ReLU) | functional | stated | the plot compares ReLU and the sigmoid with GELU, SiLU and SwiGLU; "Non-linearities, old and new" | slides 15–16 |
| C11 Cross-entropy loss | `applies_to` | C06 A one-hidden-layer feed-forward network (the NER scorer) | functional | inferred | "Training with 'cross entropy loss'" follows the NER network | slides 14, 18 |
| C12 The max-margin objective | `applies_to` | C06 A one-hidden-layer feed-forward network (the NER scorer) | functional | stated | s = Uᵀf(Wx + b), s_c = Uᵀf(Wx_c + b) | notes 3 §1.4 |
| C12 The max-margin objective | `contrasts_with` | C11 Cross-entropy loss | functional | inferred | two training objectives for the same scorer | notes 3 §1.4; slide 18 |
| C14 The gradient (of a scalar function of n inputs) | `kind_of` | C15 The Jacobian matrix | taxonomic | stated | "Jacobian Matrix: Generalization of the Gradient" | slide 23 |
| C16 The multivariable chain rule (multiply Jacobians) | `uses` | C15 The Jacobian matrix | functional | stated | "multiply Jacobians" | slide 24 |
| C17 The Jacobian of an elementwise activation | `illustrates` | C15 The Jacobian matrix | functional | stated | "Example Jacobian" | slides 25–29 |
| C18 Useful Jacobians of the affine map and the score | `illustrates` | C15 The Jacobian matrix | functional | stated | "Other Jacobians" | slides 30–33 |
| C13 Vectorized gradients by hand (matrix calculus) | `uses` | C16 The multivariable chain rule (multiply Jacobians) | functional | inferred | the by-hand method is the chain rule over Jacobians | slides 20, 24 |
| C19 The worked gradient of the NER score (break up, chain rule, Jacobians) | `illustrates` | C16 The multivariable chain rule (multiply Jacobians) | functional | inferred | the chain rule applied to the NER net | slides 37–45 |
| C19 The worked gradient of the NER score (break up, chain rule, Jacobians) | `uses` | C17 The Jacobian of an elementwise activation | functional | stated | "Useful Jacobians from previous slide" | slides 41–45 |
| C19 The worked gradient of the NER score (break up, chain rule, Jacobians) | `uses` | C18 Useful Jacobians of the affine map and the score | functional | stated | "Useful Jacobians from previous slide" | slides 41–45 |
| C21 The gradient with respect to a weight matrix (outer product) | `uses` | C20 Reusing computation: the upstream gradient δ ("error signal") | functional | stated | ∂s/∂W = δ ∂z/∂W | slides 48, 51 |
| C22 The shape convention vs the Jacobian form | `applies_to` | C21 The gradient with respect to a weight matrix (outer product) | functional | stated | "Derivative with respect to Matrix: Output shape" | slides 49–50 |
| C24 Backpropagation: upstream × local = downstream | `moves_over` | C23 Computation graphs and forward propagation | functional | stated | "Then go backwards along edges" | slide 60 |
| C24 Backpropagation: upstream × local = downstream | `uses` | C16 The multivariable chain rule (multiply Jacobians) | functional | stated | "Chain rule!" | slide 63 |
| C25 Gradients sum at outward branches | `property_of` | C24 Backpropagation: upstream × local = downstream | functional | stated | "Gradients sum at outward branches" | slides 67–68 |
| C26 Node intuitions: + distributes, max routes, * switches | `illustrates` | C24 Backpropagation: upstream × local = downstream | functional | stated | "Node Intuitions" | slides 69–71 |
| C27 Backprop in a general graph: all gradients at once, in reverse topological order | `applies_to` | C23 Computation graphs and forward propagation | functional | stated | "Back-Prop in General Computation Graph" | slide 75 |
| C27 Backprop in a general graph: all gradients at once, in reverse topological order | `uses` | C25 Gradients sum at outward branches | functional | inferred | the sum over successors is the branch rule | slides 68, 75 |
| C28 Automatic differentiation (and how frameworks implement backprop) | `uses` | C27 Backprop in a general graph: all gradients at once, in reverse topological order | functional | inferred | the framework runs the algorithm to compute gradients | slides 76–77 |
| C29 Numeric gradient checking | `assesses` | C28 Automatic differentiation (and how frameworks implement backprop) | functional | stated | "Useful for checking your implementation" | slide 78 |
| C32 Dropout | `contrasts_with` | C31 L2 regularization of the weights | functional | stated | dropout "acts as another form of regularization" | notes 3 §2.2 |
| C37 Adaptive optimizers (AdaGrad, RMSProp, Adam) | `uses` | C36 Momentum updates | functional | stated | Adam adds "momentum-like updates" | notes 3 §2.9 |
| C36 Momentum updates | `adjusts` | B08 Stochastic gradient descent | functional | inferred | a variant update rule for SGD | notes 3 §2.8 |
| C35 Learning-rate strategies | `adjusts` | B08 Stochastic gradient descent | functional | stated | the α in θ^{new} = θ^{old} − α∇_θJ_t(θ) | notes 3 §2.7 |
| C39 A2's neural transition classifier (after Chen and Manning 2014) | `method_for` | C38 Transition-based dependency parsing | functional | stated | "your parser will decide among the three transitions using a neural network classifier" | A2 §3 |
| C40 Unlabeled attachment score (UAS) | `assesses` | C38 Transition-based dependency parsing | functional | stated | "We will use UAS score as our evaluation metric" | A2 Q3(e)iii |
| C39 A2's neural transition classifier (after Chen and Manning 2014) | `uses` | C04 The embedding layer (x = Le) | functional | stated | "looks up an embedding for each word" | A2 Q3(e) |
| C39 A2's neural transition classifier (after Chen and Manning 2014) | `uses` | C11 Cross-entropy loss | functional | stated | "minimize cross-entropy loss" | A2 Q3(e) |
| C39 A2's neural transition classifier (after Chen and Manning 2014) | `uses` | C32 Dropout | functional | stated | "include a dropout layer … after ReLU function" | parser_model.py |
| C39 A2's neural transition classifier (after Chen and Manning 2014) | `uses` | C34 Parameter initialization (Xavier/Glorot) | functional | stated | "nn.init.xavier_uniform_" | parser_model.py |
| C39 A2's neural transition classifier (after Chen and Manning 2014) | `uses` | C37 Adaptive optimizers (AdaGrad, RMSProp, Adam) | functional | stated | "Construct Adam Optimizer" | run.py |

<!-- Row shape: | C02 Label | `has_parameters` | C03 Label | part-whole | stated | — | "short quote" (§3.1) | -->
