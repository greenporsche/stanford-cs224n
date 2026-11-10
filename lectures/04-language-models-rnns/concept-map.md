# Concept map: *Stanford CS224N (Winter 2026)*, Chapter 4 (Lecture 4: Language Models and RNNs)

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

1. **The plan, and the task** (slides 1–8; notes 5 §1.1). The title slide reads "Lecture 4: Language Models and Recurrent Neural Networks" (Diyi Yang). The plan has four timed parts: "A new NLP task: Language Modeling (20 mins)", "Language models with neural nets: Recurrent Neural Networks (RNNs) (25 mins)", "Problems with RNNs: exploding and vanishing gradients (20 mins)" and "Machine translation (10 mins)"; language modeling is marked "This is the most important concept in the class! Leads to most of modern NLP" (slide 2). The lecture defines the task (C01), says why it matters (C02) and shows what predicting the next word can require (C03).

2. **Counting: n-gram language models** (slides 9–18; notes 5 §1.2). Before deep learning, an LM was learned by counting n-grams under a Markov assumption (C04). Counting runs into sparsity, patched by smoothing and backoff (C05), and storage (C06). Generating text by sampling (C07) shows the output is "Surprisingly grammatical! …but incoherent": the short context is the real limit, and enlarging it makes both problems worse (slide 18).

3. **A fixed-window neural LM** (slides 19–22; notes 5 §1.3). The first neural answer reuses Lecture 3's window network with a softmax over the vocabulary, after Bengio et al. (C08). It removes sparsity and n-gram storage, but its window is fixed, can never be large enough and treats each position with different weights (C09): "We need a neural architecture that can process any length input" (slide 22).

4. **Recurrent neural networks** (slides 23–25; notes 5 §2, §2.2). Answering slide 22's call for "a neural architecture that can process any length input", an RNN applies the same weights at every step to a running hidden state (C10). As a language model it reads the words one by one and predicts the next at each step (C11). It can take any length of input with a fixed model size, but it is slow and in practice struggles to use information from many steps back, "More on these later" (C12). RNNs also tag, classify and encode sentences (C13).

5. **Training an RNN LM** (slides 26–35; notes 5 §2.1). It is trained like any LM: cross-entropy on the true next word at every step, averaged, with the corpus words as inputs ("Teacher forcing") and SGD on batches of sentences (C14). The new step is backpropagation through time: one weight matrix is used at every step, so its gradient is the sum of its gradients at each appearance, by the multivariable chain rule (C15).

6. **Using and measuring an RNN LM** (slides 36–39; notes 5 §2.1). The trained model generates text by feeding each sampled word back in (C07, extended), in the style of whatever it was trained on. It is measured by perplexity, the inverse probability of the corpus normalized by its length, which equals the exponentiated training loss (C16).

7. **Exploding and vanishing gradients** (slides 40–50; notes 5 §2.3–2.4). Backpropagation through time multiplies one local Jacobian per step. When those factors are small the gradient vanishes (C17), so the model learns only near effects and misses long-distance dependencies (C18). Large factors can make it explode, with bad updates and NaNs (C19), which gradient clipping fixes easily (C20). Vanishing needs a change of architecture: an RNN with separate memory (LSTM), then more direct pass-through connections such as attention and residual connections; notes 5 adds identity initialization and ReLU (C21).

8. **More RNN shapes** (notes 5 §2.5; no slides). Notes 5 widens the family: read the sequence in both directions and combine the two hidden states (C22), and stack RNN layers so each layer's states feed the next (C23). The stacking returns in the multi-layer encoder–decoder (slide 57).

9. **Machine translation and seq2seq** (slides 51–59; notes 5 §2.6). Machine translation was "the first big success story of NLP Deep Learning" (C24). Its model is the sequence-to-sequence encoder–decoder: an encoder RNN encodes the source and a decoder RNN, a language model, generates the target from that encoding (C25). It is a conditional language model, trained end to end on a parallel corpus (C26). The lecture ends on a question about the encoder's single hand-off, "the bottleneck problem in RNNs" (slide 58), and slide 59 repeats the plan.

10. **Gated units** (notes 5 §3–4; slide 50's pointer). Notes 5 answers slide 50's first fix with gated recurrent units: the GRU, whose update and reset gates mix the old state with a new memory (C27), and the LSTM, which adds an output gate and keeps a separate memory cell that is updated by addition (C28).

## Assumed background

What the chapter uses without teaching it is recorded as **studs** (B01…, below).

**Chapter scope** names what was read, including footnotes or notes wherever the converter placed them.

**Chapter scope:** the Winter 2026 Lecture 4 deck, all 59 PDF pages (slide N = PDF page N; printed numbers run two ahead from page 9 and four ahead from page 46, and pages 1–2, 8, 45 and 59 show none); notes 5 (Winter 2019, "Language Models, RNN, GRU and LSTM", 14 pp., §1–4 with Figures 1–13); and the four schedule readings where the lecture leans on them (Bengio, Simard & Frasconi 1994; Pascanu, Mikolov & Bengio 2013 with its supplement; the vanishing-gradient notebook; Vaswani et al. 2017). Olah's "Understanding LSTM Networks" (slide 50's link) and the papers the slides and notes name are used where they point.

## Figures

Concepts cite figures by number. Images are linked, never committed.

| Fig | Shows | Teaches | Image |
|---|---|---|---|
| 1.2 | the lecture plan: four timed parts; language modeling "the most important concept in the class" | C02 | [1.2](materials/slides/cs224n-2026-lecture04-rnnlm/p02.png) |
| 1.3 | language modeling: "the students opened their ___" (books, laptops, exams, minds); P(x^(t+1) \| x^(t), …, x^(1)) | C01 | [1.3](materials/slides/cs224n-2026-lecture04-rnnlm/p03.png) |
| 1.4 | the probability of a text as a product of next-word conditionals, "This is what our LM provides" | C01 | [1.4](materials/slides/cs224n-2026-lecture04-rnnlm/p04.png) |
| 1.5 | everyday LMs: phone keyboard suggestions and search autocomplete | C02 | [1.5](materials/slides/cs224n-2026-lecture04-rnnlm/p05.png) |
| 1.6 | everyday LMs: ChatGPT and a coding assistant | C02 | [1.6](materials/slides/cs224n-2026-lecture04-rnnlm/p06.png) |
| 1.7 | why care about language modeling: benchmark, subcomponent of many tasks, "ChatGPT is an LM!" | C02 | [1.7](materials/slides/cs224n-2026-lecture04-rnnlm/p07.png) |
| 1.8 | what next-word prediction can do: trivia, syntax, coreference, topic, sentiment, reasoning, arithmetic | C03 | [1.8](materials/slides/cs224n-2026-lecture04-rnnlm/p08.png) |
| 1.9 | n-grams defined: unigrams to four-grams of "the students opened their" | C04 | [1.9](materials/slides/cs224n-2026-lecture04-rnnlm/p09.png) |
| 1.10 | the Markov assumption, the conditional as n-gram over (n−1)-gram probability, and the counting approximation | C04 | [1.10](materials/slides/cs224n-2026-lecture04-rnnlm/p10.png) |
| 1.11 | a 4-gram example: "students opened their" 1000, "… books" 400, "… exams" 100; "Should we have discarded the 'proctor' context?" | C04, C06 | [1.11](materials/slides/cs224n-2026-lecture04-rnnlm/p11.png) |
| 1.12 | sparsity problems 1 and 2, with smoothing and backoff; "we can't have n bigger than 5" | C05 | [1.12](materials/slides/cs224n-2026-lecture04-rnnlm/p12.png) |
| 1.13 | the storage problem: counts for all n-grams; size grows with n and corpus | C06 | [1.13](materials/slides/cs224n-2026-lecture04-rnnlm/p13.png) |
| 1.14 | a Reuters trigram LM: "today the ___" (company 0.153, bank 0.153, …); "not much granularity" | C04, C05 | [1.14](materials/slides/cs224n-2026-lecture04-rnnlm/p14.png) |
| 1.15 | generating: condition on "today the", get the distribution, sample "price" (0.077) | C07 | [1.15](materials/slides/cs224n-2026-lecture04-rnnlm/p15.png) |
| 1.16 | generating: condition on "the price", sample "of" (0.308) | C07 | [1.16](materials/slides/cs224n-2026-lecture04-rnnlm/p16.png) |
| 1.17 | generating: condition on "price of", sample "gold" (0.018) | C07 | [1.17](materials/slides/cs224n-2026-lecture04-rnnlm/p17.png) |
| 1.18 | the generated text: "Surprisingly grammatical! …but incoherent" | C06, C07 | [1.18](materials/slides/cs224n-2026-lecture04-rnnlm/p18.png) |
| 1.19 | how to build a neural LM: input a word sequence, output the next-word distribution; the NER window model recalled | C08 | [1.19](materials/slides/cs224n-2026-lecture04-rnnlm/p19.png) |
| 1.20 | the fixed window: keep "the students opened their", discard "as the proctor started the clock" | C08 | [1.20](materials/slides/cs224n-2026-lecture04-rnnlm/p20.png) |
| 1.21 | the fixed-window neural LM: one-hot words, concatenated embeddings e, hidden layer h = f(We + b₁), ŷ = softmax(Uh + b₂) ∈ ℝ^\|V\| | C08 | [1.21](materials/slides/cs224n-2026-lecture04-rnnlm/p21.png) |
| 1.22 | improvements over n-gram LMs and remaining problems; "We need a neural architecture that can process any length input" | C08, C09 | [1.22](materials/slides/cs224n-2026-lecture04-rnnlm/p22.png) |
| 2.1 | notes 5 Figure 1: Bengio et al.'s neural LM (table look-up in a shared matrix C, tanh hidden layer, softmax output) | C08 | [2.1](materials/figures/notes5-p03.png) |
| 2.2 | notes 5 Figure 2: a simplified representation of Figure 1 (slide 21's model) | C08 | [2.2](materials/figures/notes5-p03.png) |
| 1.23 | an RNN: input sequence of any length, hidden states, optional outputs; "Core idea: Apply the same weights W repeatedly" | C10 | [1.23](materials/slides/cs224n-2026-lecture04-rnnlm/p23.png) |
| 1.24 | a simple RNN LM: one-hot words, embeddings e^(t) = Ex^(t), h^(t) = σ(W_h h^(t−1) + W_e e^(t) + b₁), ŷ^(t) = softmax(Uh^(t) + b₂) | C11 | [1.24](materials/slides/cs224n-2026-lecture04-rnnlm/p24.png) |
| 1.25 | RNN LM advantages and disadvantages, "More on these later" | C12 | [1.25](materials/slides/cs224n-2026-lecture04-rnnlm/p25.png) |
| 2.3 | notes 5 Figure 3: a recurrent neural network, three time-steps | C10 | [2.3](materials/figures/notes5-p03.png) |
| 2.4 | notes 5 Figure 4: the inputs and outputs to a neuron of an RNN | C10 | [2.4](materials/figures/notes5-p04.png) |
| 2.5 | notes 5 Figure 5: an RNN language model (slide 24's figure) | C11 | [2.5](materials/figures/notes5-p04.png) |
| 2.6 | notes 5 Figure 6: an RNN drawn as a loop over time-steps | C10 | [2.6](materials/figures/notes5-p06.png) |
| 1.26 | training an RNN LM: per-step cross-entropy J^(t)(θ) = −log ŷ^(t)_{x_{t+1}} and the average J(θ) | C14 | [1.26](materials/slides/cs224n-2026-lecture04-rnnlm/p26.png) |
| 1.27 | the loss at step 1: negative log prob of "students" | C14 | [1.27](materials/slides/cs224n-2026-lecture04-rnnlm/p27.png) |
| 1.28 | the loss at step 2: negative log prob of "opened" | C14 | [1.28](materials/slides/cs224n-2026-lecture04-rnnlm/p28.png) |
| 1.29 | the loss at step 3: negative log prob of "their" | C14 | [1.29](materials/slides/cs224n-2026-lecture04-rnnlm/p29.png) |
| 1.30 | the loss at step 4: negative log prob of "exams" | C14 | [1.30](materials/slides/cs224n-2026-lecture04-rnnlm/p30.png) |
| 1.31 | the summed losses over the corpus, labelled "Teacher forcing" | C14 | [1.31](materials/slides/cs224n-2026-lecture04-rnnlm/p31.png) |
| 1.32 | in practice: a sentence or document at a time, SGD on batches of sentences | C14 | [1.32](materials/slides/cs224n-2026-lecture04-rnnlm/p32.png) |
| 1.33 | backpropagation for RNNs: the derivative w.r.t. the repeated W_h is the sum over its appearances | C15 | [1.33](materials/slides/cs224n-2026-lecture04-rnnlm/p33.png) |
| 1.34 | the multivariable chain rule, with Lecture 3's "Gradients sum at outward branches" | C15 | [1.34](materials/slides/cs224n-2026-lecture04-rnnlm/p34.png) |
| 1.35 | backpropagation through time: each W_h\|_(i) "equals" W_h; sum as you go back; truncated after ~20 steps | C15 | [1.35](materials/slides/cs224n-2026-lecture04-rnnlm/p35.png) |
| 1.36 | generating roll outs: from `<s>`, "my favorite season is spring `</s>`", each sample fed back as the next input | C07 | [1.36](materials/slides/cs224n-2026-lecture04-rnnlm/p36.png) |
| 1.37 | an RNN-LM trained on Harry Potter | C07 | [1.37](materials/slides/cs224n-2026-lecture04-rnnlm/p37.png) |
| 1.38 | an RNN-LM trained on recipes | C07 | [1.38](materials/slides/cs224n-2026-lecture04-rnnlm/p38.png) |
| 1.39 | perplexity: the inverse probability of the corpus, normalized by the number of words, equal to exp(J(θ)) | C16 | [1.39](materials/slides/cs224n-2026-lecture04-rnnlm/p39.png) |
| 1.40 | section slide: "3. Problems with RNNs: Vanishing and Exploding Gradients" | C17 | [1.40](materials/slides/cs224n-2026-lecture04-rnnlm/p40.png) |
| 1.41 | vanishing gradient intuition: J^(4)(θ) above h^(4), with h^(1)–h^(4) joined by W | C17 | [1.41](materials/slides/cs224n-2026-lecture04-rnnlm/p41.png) |
| 1.42 | the chain rule, first split: ∂J^(4)/∂h^(1) = (∂h^(2)/∂h^(1)) × (∂J^(4)/∂h^(2)) | C17 | [1.42](materials/slides/cs224n-2026-lecture04-rnnlm/p42.png) |
| 1.43 | the chain rule, the remaining gradient expanded through h^(3): ∂h^(2)/∂h^(1) × ∂h^(3)/∂h^(2) × ∂J^(4)/∂h^(3) | C17 | [1.43](materials/slides/cs224n-2026-lecture04-rnnlm/p43.png) |
| 1.44 | the full chain ∂J^(4)/∂h^(1) = ∂h^(2)/∂h^(1) × ∂h^(3)/∂h^(2) × ∂h^(4)/∂h^(3) × ∂J^(4)/∂h^(4) | C17 | [1.44](materials/slides/cs224n-2026-lecture04-rnnlm/p44.png) |
| 1.45 | "What happens if these are small?": the vanishing gradient problem (source: Pascanu et al. 2013) | C17 | [1.45](materials/slides/cs224n-2026-lecture04-rnnlm/p45.png) |
| 1.46 | why vanishing is a problem: far-away gradient signal lost against close-by signal | C18 | [1.46](materials/slides/cs224n-2026-lecture04-rnnlm/p46.png) |
| 1.47 | the effect on an RNN-LM: the "tickets" example | C18 | [1.47](materials/slides/cs224n-2026-lecture04-rnnlm/p47.png) |
| 1.48 | why exploding is a problem: too large an SGD step, "suddenly you're in Iowa", Inf or NaN | C19 | [1.48](materials/slides/cs224n-2026-lecture04-rnnlm/p48.png) |
| 1.49 | gradient clipping: Algorithm 1, norm clipping (source: Pascanu et al. 2013) | C20 | [1.49](materials/slides/cs224n-2026-lecture04-rnnlm/p49.png) |
| 1.50 | how to fix the vanishing gradient: separate memory (LSTM), then attention, residual connections | C21 | [1.50](materials/slides/cs224n-2026-lecture04-rnnlm/p50.png) |
| 2.7 | notes 5 Figure 7: gradient explosion clipping visualization, a one-unit RNN's error surface with its high wall | C20 | [2.7](materials/figures/notes5-p08.png) |
| 2.8 | notes 5 Figure 8: a bi-directional RNN model | C22 | [2.8](materials/figures/notes5-p09.png) |
| 2.9 | notes 5 Figure 9: a deep bi-directional RNN with three RNN layers | C23 | [2.9](materials/figures/notes5-p09.png) |
| 1.51 | machine translation: x "I like deep learning" → y "我喜欢深度学习" | C24 | [1.51](materials/slides/cs224n-2026-lecture04-rnnlm/p51.png) |
| 1.52 | NMT: the first big success story of NLP deep learning (2014–2018) | C24 | [1.52](materials/slides/cs224n-2026-lecture04-rnnlm/p52.png) |
| 1.53 | the sequence-to-sequence model at test time: encoder RNN, encoding, decoder RNN with argmax outputs fed back (il m' a entarté → he hit me with a pie) | C25 | [1.53](materials/slides/cs224n-2026-lecture04-rnnlm/p53.png) |
| 1.54 | sequence-to-sequence is versatile: summarization, dialogue, parsing, code generation | C25 | [1.54](materials/slides/cs224n-2026-lecture04-rnnlm/p54.png) |
| 1.55 | seq2seq as a conditional language model: P(y\|x) as a product of next-target-word conditionals | C26 | [1.55](materials/slides/cs224n-2026-lecture04-rnnlm/p55.png) |
| 1.56 | training an NMT system: J = (1/T)Σ J_t over the target words; backpropagation "end-to-end" | C26 | [1.56](materials/slides/cs224n-2026-lecture04-rnnlm/p56.png) |
| 1.57 | a multi-layer deep encoder–decoder (Die Proteste waren am Wochenende eskaliert → The protests escalated over the weekend); "Conditioning = Bottleneck" | C25 | [1.57](materials/slides/cs224n-2026-lecture04-rnnlm/p57.png) |
| 1.58 | the bottleneck problem in RNNs: "Encoding of the source sentence"; "Problems with this architecture?" | C25 | [1.58](materials/slides/cs224n-2026-lecture04-rnnlm/p58.png) |
| 1.59 | the lecture plan repeated | C01 | [1.59](materials/slides/cs224n-2026-lecture04-rnnlm/p59.png) |
| 2.10 | notes 5 Figure 10: an RNN-based translation model (Echt dicke Kiste → Awesome sauce) | C25 | [2.10](materials/figures/notes5-p10.png) |
| 2.11 | notes 5 Figure 11: a decoder with three inputs per step (h_{t−1}, c, y_{t−1}) | C25 | [2.11](materials/figures/notes5-p10.png) |
| 2.12 | notes 5 Figure 12: the detailed internals of a GRU (reset, update, new memory) | C27 | [2.12](materials/figures/notes5-p12.png) |
| 2.13 | notes 5 Figure 13: the detailed internals of an LSTM (input, forget, output gates; memory cell) | C28 | [2.13](materials/figures/notes5-p13.png) |

<!-- Row shape: | 3.1 | what the figure shows | C01, C02 | [3.1](<image url>) |
     For a picture extracted from a prepared EPUB (overlay section 1), the link is its absolute file URI: | 4.1 | ... | C05 | [4.1](file:///.../prepared/books/BK-<hash>.assets/OPS/images/<file>) | -->

Figure numbers: 1.N is slide N (PDF page N of the W26 deck); 2.N is notes 5 Figure N. Image links are relative to the published lecture folder, where the slide images sit beside the map.

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

### C01 Language model (next-word distribution; the probability of a text)
- **Meaning:** "Language Modeling is the task of predicting what word comes next": given a sequence of words x^(1), x^(2), …, x^(t), compute the probability distribution of the next word, P(x^(t+1) \| x^(t), …, x^(1)), "where x^(t+1) can be any word in the vocabulary V = {w_1, …, w_\|V\|}". "A system that does this is called a Language Model" (slide 3).
- **Key points:**
  - Equivalently, an LM "assigns a probability to a piece of text": P(x^(1), …, x^(T)) = P(x^(1)) × P(x^(2) \| x^(1)) × ⋯ × P(x^(T) \| x^(T−1), …, x^(1)) = ∏_{t=1}^{T} P(x^(t) \| x^(t−1), …, x^(1)), where the next-word conditional is "what our LM provides" (slide 4).
  - Notes 5 §1.1: "Language models compute the probability of occurrence of a number of words in a particular sequence", written P(w_1, …, w_m).
- **Example:** "the students opened their ___": books, laptops, exams, minds (slide 3).
- **Extends:** ch01:C36, which told the history of the idea; this lecture defines the task, the next-word distribution and the chain-rule factorisation of a text's probability.
- **Figures:** 1.3, 1.4.

### C02 Why language modeling matters (uses, and the base of modern NLP)
- **Meaning:** Language models are in everyday use and underlie most of NLP.
- **Key points:**
  - "You use Language Models every day!": predictive typing on a phone keyboard, search autocomplete, ChatGPT, a coding assistant (slides 5–6).
  - "Language Modeling is a benchmark task that helps us measure our progress on predicting language use" and "a subcomponent of many NLP tasks, especially those involving generating text or estimating the probability of text": predictive typing, speech recognition, handwriting recognition, spelling/grammar correction, authorship identification, machine translation, summarization, dialogue (slide 7).
  - "Everything else in NLP has been rebuilt upon Language Modeling: ChatGPT is an LM!" (slide 7); the plan marks language modeling "This is the most important concept in the class! Leads to most of modern NLP" (slide 2).
- **Example:** notes 5 §1.1: a translation system scores alternative word sequences and outputs the highest-scoring one, rating "the cat is small" above "small the is cat" and "walking home after school" above "walking house after school".
- **Figures:** 1.2, 1.5, 1.6, 1.7.

### C03 What next-word prediction can teach a model
- **Meaning:** "A sufficiently strong (!) language model can do many, many things": filling in the next word draws on many kinds of knowledge (slide 8).
- **Example:** slide 8's blanks: "Stanford University is located in ___, California." [Trivia]; "I put ___ fork down on the table." [syntax]; "The woman walked across the street, checking for traffic over ___ shoulder." [coreference]; "I went to the ocean to see the fish, turtles, seals, and ___." [lexical semantics/topic]; the popcorn movie review [sentiment]; "Iroh went into the kitchen to make some tea. Standing next to Iroh, Zuko pondered his destiny. Zuko left the ___." [some reasoning – this is harder]; "1, 1, 2, 3, 5, 8, 13, 21, ___" [some basic arithmetic].
- **Figures:** 1.8.

### C04 n-gram language model
- **Meaning:** "An n-gram is a chunk of n consecutive words": unigrams "the", "students", "opened", "their"; bigrams "the students", "students opened", "opened their"; trigrams; the four-gram "the students opened their". "Idea: Collect statistics about how frequent different n-grams are and use these to predict next word" (slide 9), the pre-deep-learning answer to "How to learn a Language Model?".
- **Key points:**
  - "First we make a Markov assumption: x^(t+1) depends only on the preceding n-1 words": P(x^(t+1) \| x^(t), …, x^(1)) = P(x^(t+1) \| x^(t), …, x^(t−n+2)) (assumption) = P(x^(t+1), x^(t), …, x^(t−n+2)) / P(x^(t), …, x^(t−n+2)), "prob of a n-gram" over "prob of a (n-1)-gram" (definition of conditional prob) ≈ count(x^(t+1), x^(t), …, x^(t−n+2)) / count(x^(t), …, x^(t−n+2)) (statistical approximation), the counts obtained "By counting them in some large corpus of text!" (slide 10).
  - Notes 5 §1.2 Eqs. 2–3: p(w_2 \| w_1) = count(w_1, w_2)/count(w_1); p(w_3 \| w_1, w_2) = count(w_1, w_2, w_3)/count(w_1, w_2).
  - Source note: notes 5 §1.1 Eq. 1 conditions on "a window of n previous words", P(w_i \| w_{i−n}, …, w_{i−1}), which is n + 1 words per gram; the slides' n-gram conditions on the preceding n − 1.
- **Example:** a 4-gram LM on "as the proctor started the clock, the students opened their ___" discards all but "students opened their"; if that occurred 1000 times, "students opened their books" 400 times and "… exams" 100 times, then P(books \| students opened their) = 0.4 and P(exams \| students opened their) = 0.1, and the slide asks "Should we have discarded the 'proctor' context?" (slide 11). "You can build a simple trigram Language Model over a 1.7 million word corpus (Reuters) in a few seconds on your laptop"; for "today the ___" it gives company 0.153, bank 0.153, price 0.077, italian 0.039, emirate 0.039: "Otherwise, seems reasonable!" (slide 14).
- **Figures:** 1.9, 1.10, 1.11, 1.14.

### C05 Sparsity in n-gram LMs (smoothing and backoff)
- **Meaning:** Counts of long word sequences are often zero, which breaks the counting estimate in two ways (slide 12; notes 5 §1.2).
- **Key points:**
  - Sparsity Problem 1: "What if 'students opened their w' never occurred in data? Then w has probability 0!" "(Partial) Solution: Add small δ to the count for every w ∈ V. This is called smoothing."
  - Sparsity Problem 2: "What if 'students opened their' never occurred in data? Then we can't calculate probability for any w!" "(Partial) Solution: Just condition on 'opened their' instead. This is called backoff."
  - "Increasing n makes sparsity problems worse. Typically, we can't have n bigger than 5" (slide 12; notes 5 §1.2 "Typically, n ≤ 5").
  - Slide 14 labels a further symptom "Sparsity problem: not much granularity in the probability distribution" (company and bank both at 0.153).
- **Figures:** 1.12, 1.14.

### C06 Storage and the context-length trade-off
- **Meaning:** An n-gram LM stores a count for every n-gram it has seen, and the only way to give it more context, a larger n, makes both storage and sparsity worse.
- **Key points:**
  - "Storage: Need to store count for all n-grams you saw in the corpus. Increasing n or increasing corpus increases model size!" (slide 13; notes 5 §1.2).
  - The generated text is "Surprisingly grammatical! …but incoherent. We need to consider more than three words at a time if we want to model language well. But increasing n worsens sparsity problem, and increases model size…" (slide 18).
- **Example:** the "proctor" context, which a 4-gram LM discards (slide 11); notes 5 §1.2: with only "the students opened their" the model may suggest "books", "however, if n had been large enough to include the 'proctor' context, the probability might have suggested 'exam'".
- **Figures:** 1.11, 1.13, 1.18.

### C07 Generating text with a language model
- **Meaning:** "You can also use a Language Model to generate text": condition on the context, get the probability distribution of the next word, sample a word, append it, and repeat (slides 15–17).
- **Key points:**
  - The trigram conditions on the last two words only, and the window moves along at each step.
  - Sampling can pick a word that is not the most likely one.
- **Example:** "today the" → sample "price" (0.077; company 0.153, bank 0.153, price 0.077, italian 0.039, emirate 0.039); "the price" → sample "of" (0.308; for 0.050, it 0.046, to 0.046, is 0.031); "price of" → sample "gold" (0.018; the 0.072, 18 0.043, oil 0.043, its 0.036) (slides 15–17). The full output: "today the price of gold per ton , while production of shoe lasts and shoe industry , the bank intervened just after it considered and rejected an imf demand to rebuild depleted european stocks , sept 30 end primary 76 cts a share ." (slide 18).
- *Extended at U6 (#351):* "Just like an n-gram Language Model, you can use a RNN Language Model to generate text by repeated sampling. Sampled output becomes next step's input" ("Generating roll outs", slide 36): from `<s>`, the samples are my, favorite, season, is, spring, `</s>`. "You can train an RNN-LM on any kind of text, then generate text in that style": an RNN-LM trained on Harry Potter ("'Sorry,' Harry shouted, panicking—'I'll leave those brooms in London, are they?'", slide 37) and on recipes ("Title: CHOCOLATE RANCH BARBECUE", "Categories: Game, Casseroles, Cookies, Cookies", "Place each pasta over layers of lumps", slide 38). Figure note: slide 36 labels every step after the fourth h^(4), ŷ^(4) and e^(4) again, instead of h^(5), ŷ^(5), e^(5), ….
- **Figures:** 1.15, 1.16, 1.17, 1.18, 1.36, 1.37, 1.38.

### C08 Fixed-window neural language model
- **Meaning:** "How about a window-based neural model? We saw this applied to Named Entity Recognition" (slide 19). The model keeps a fixed window of the most recent words, e.g. "the students opened their", and discards the rest, "as the proctor started the clock" (slide 20). Words / one-hot vectors x^(1), …, x^(4) become concatenated word embeddings e = [e^(1); e^(2); e^(3); e^(4)], a hidden layer h = f(We + b₁), and an output distribution ŷ = softmax(Uh + b₂) ∈ ℝ^\|V\| over the vocabulary (slide 21).
- **Key points:**
  - "Approximately: Y. Bengio, et al. (2000/2003): A Neural Probabilistic Language Model" (slide 22). Notes 5 §1.3: it "introduced the first large-scale deep learning for natural language processing model", which "learns a distributed representation of words, along with the probability function for word sequences expressed in terms of these representations".
  - Bengio's form: ŷ = softmax(W^(2) tanh(W^(1)x + b^(1)) + W^(3)x + b^(3)) (notes 5 Eq. 4); W^(1) is applied to the word vectors, W^(2) to the hidden layer, and W^(3) to the word vectors directly (the dashed arrows of Figure 1). Figure 2 is "A simplified representation of Figure 1", with e = [e^(1); e^(2); e^(3); e^(4)], h = f(We + b₁) and ŷ = softmax(Uh + b₂).
- **Example:** "the students opened their" → a distribution over the vocabulary, from "a" to "zoo", with "books" and "laptops" high (slide 21).
- **Extends:** ch03:C06, the one-hidden-layer network, and ch03:C02, the window classifier: the output here is a softmax over the whole vocabulary instead of a class score.
- **Figures:** 1.19, 1.20, 1.21, 1.22, 2.1, 2.2.

### C09 What the fixed window fixes, and what it can't
- **Meaning:** Against n-gram LMs the fixed-window neural LM has two "Improvements" and four "Remaining problems", which call for an architecture that takes any length of input (slide 22).
- **Key points:**
  - Improvements over n-gram LM: "No sparsity problem"; "Don't need to store all observed n-grams".
  - Remaining problems: "Fixed window is too small"; "Enlarging window enlarges W"; "Window can never be large enough!"; "x^(1) and x^(2) are multiplied by completely different weights in W. No symmetry in how the inputs are processed."
  - "We need a neural architecture that can process any length input" (slide 22).
- **Figures:** 1.22.

### C10 Recurrent neural network (RNN)
- **Meaning:** "A family of neural architectures" whose "Core idea" is to "Apply the same weights W repeatedly": an "input sequence (any length)" x^(1), x^(2), … feeds "hidden states" h^(1), h^(2), …, each computed from the previous one with the same W, with "outputs (optional)" ŷ^(t) (slide 23).
- **Key points:**
  - Notes 5 §2: RNNs "are capable of conditioning the model on all previous words in the corpus". At each time-step the hidden layer has two inputs, the previous output h_{t−1} "multiplied by a weight matrix W^(hh)" and the input x_t multiplied by W^(hx) (Figures 3–4).
  - "the same weights W^(hh) and W^(hx) are applied repeatedly at each timestep. Thus, the number of parameters the model has to learn is less, and most importantly, is independent of the length of the input sequence - thus defeating the curse of dimensionality!" (notes 5 §2).
  - Figure 6 is "an alternative representation of RNNs used in some publications", drawing the hidden layer "as a loop" (notes 5 §2.2).
- **Figures:** 1.23, 2.3, 2.4, 2.6.

### C11 RNN language model
- **Meaning:** An RNN that reads a text word by word and outputs, at each step, a distribution over the next word (slide 24): words / one-hot vectors x^(t) ∈ ℝ^\|V\|; word embeddings e^(t) = Ex^(t); hidden states h^(t) = σ(W_h h^(t−1) + W_e e^(t) + b₁), where "h^(0) is the initial hidden state"; output distribution ŷ^(t) = softmax(Uh^(t) + b₂) ∈ ℝ^\|V\|.
- **Key points:**
  - "Note: this input sequence could be much longer now!" (slide 24).
  - Notes 5 Eqs. 5–6: h_t = σ(W^(hh)h_{t−1} + W^(hx)x_t); ŷ_t = softmax(W^(S)h_t). Dimensions: x_t ∈ ℝ^d; W^hx ∈ ℝ^{D_h×d}; W^hh ∈ ℝ^{D_h×D_h}; h_{t−1} ∈ ℝ^{D_h}; h_0 ∈ ℝ^{D_h} "an initialization vector for the hidden layer at time-step t = 0"; σ "the non-linearity function (sigmoid here)"; W^(S) ∈ ℝ^{\|V\|×D_h} and ŷ ∈ ℝ^\|V\|.
  - Notation map (notes 5 Figure 5): W^(hh) is W_h, W^(hx) is W_e, W^(S) is U, and E converts word inputs x^(t) to word embeddings e^(t).
- **Example:** ŷ^(4) = P(x^(5) \| the students opened their), with "books" and "laptops" high (slide 24).
- **Figures:** 1.24, 2.5.

### C12 RNN advantages and disadvantages
- **Meaning:** What recurrence buys and what it costs (slide 25; notes 5 §2.2).
- **Key points:**
  - RNN Advantages: "Can process any length input"; "Computation for step t can (in theory) use information from many steps back"; "Model size doesn't increase for longer input context"; "Same weights applied on every timestep, so there is symmetry in how inputs are processed" (slide 25).
  - RNN Disadvantages: "Recurrent computation is slow"; "In practice, difficult to access info from many steps back", "More on these later" (slide 25). Notes 5 §2.2: "Computation is slow - because it is sequential, it cannot be parallelized"; the difficulty is "due to problems like vanishing and exploding gradients".
  - Source note: notes 5 §2.2 says "The amount of memory required to run a layer of RNN is proportional to the number of words in the corpus", then narrows it to a sentence treated as a minibatch ("a sentence with k words would have k word vectors to be stored in memory"); its "For a RNN with 1000 recurrent layers, the matrix would be 1000 × 1000 regardless of the corpus size" means a hidden size of 1000, not 1000 layers.
- **Figures:** 1.25.

### C13 RNNs beyond language modeling (sentence representations)
- **Meaning:** "RNNs can be used for many tasks, such as tagging (e.g. part-of-speech, named entity recognition), sentence classification (e.g. sentiment classification), and encoder modules (e.g. questiton answering, machine translation, and many other tasks)" (notes 5 §2.2).
- **Key points:**
  - For the latter two, "we want a representation for the sentence, which we can obtain by taking the element-wise max or mean of all hidden states of the timesteps in that sentence" (notes 5 §2.2).

### C14 Training an RNN language model
- **Meaning:** "Get a big corpus of text which is a sequence of words x^(1), …, x^(T)"; "Feed into RNN-LM; compute output distribution ŷ^(t) for every step t", "i.e., predict probability dist of every word, given words so far". The "Loss function on step t is cross-entropy between predicted probability distribution ŷ^(t), and the true next word y^(t) (one-hot for x^(t+1))": J^(t)(θ) = CE(y^(t), ŷ^(t)) = −Σ_{w∈V} y_w^(t) log ŷ_w^(t) = −log ŷ^(t)_{x_{t+1}}. "Average this to get overall loss for entire training set": J(θ) = (1/T) Σ_{t=1}^{T} J^(t)(θ) = (1/T) Σ_{t=1}^{T} −log ŷ^(t)_{x_{t+1}} (slide 26; notes 5 §2.1 Eqs. 7–8).
- **Key points:**
  - Slides 27–30 add the terms one at a time over "the students opened their … exams": J^(1)(θ) is the "negative log prob of 'students'", then of "opened", "their" and "exams". Slide 31 labels the picture "Teacher forcing": the inputs at every step are the corpus words, whatever the model predicted (our reading of the figure; the slide gives only the label).
  - "However: Computing loss and gradients across entire corpus at once is too expensive (memory-wise)!" "In practice, consider x^(1), …, x^(T) as a sentence (or a document)". "Recall: Stochastic Gradient Descent allows us to compute loss and gradients for small chunk of data, and update": "Compute loss for a sentence (actually, a batch of sentences), compute gradients and update weights. Repeat on a new batch of sentences." (slide 32).
  - Source note: slide 26 names T corpus words but averages T next-word losses, requiring an unstated target x^(T+1). Without a terminal target, the supplied sequence gives T−1 losses, averaged by T−1.
- **Figures:** 1.26, 1.27, 1.28, 1.29, 1.30, 1.31, 1.32.

### C15 Backpropagation through time (BPTT)
- **Meaning:** "What's the derivative of J^(t)(θ) w.r.t. the repeated weight matrix W_h?" "Answer: ∂J^(t)/∂W_h = Σ_{i=1}^{t} ∂J^(t)/∂W_h \|_(i)": "The gradient w.r.t. a repeated weight is the sum of the gradient w.r.t. each time it appears" (slide 33).
- **Key points:**
  - Why: the multivariable chain rule, d/dt f(x(t), y(t)) = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt), drawn as one input t, "Two intermediate outputs" x(t) and y(t), and "One final output"; beside it, Lecture 3's "Gradients sum at outward branches" (slide 34).
  - Each W_h\|_(i) "equals" W_h, so ∂W_h\|_(i)/∂W_h = 1, and applying the multivariable chain rule gives the sum. "Answer: Backpropagate over timesteps i = t, … ,0, summing gradients as you go. This algorithm is called 'backpropagation through time' [Werbos, P.G., 1988, Neural Networks 1, and others]" (slide 35).
  - "In practice, often 'truncated' after ~20 timesteps for training efficiency reasons" (slide 35).
  - Source note: slide 35's text runs i = t, …, 0, while its sum runs over i = 1, …, t.
- **Extends:** ch03:C25, ch03:C16: the summing of gradients at branches and the multivariable chain rule, applied to one weight matrix shared across time steps.
- **Figures:** 1.33, 1.34, 1.35.

### C16 Perplexity
- **Meaning:** "The standard evaluation metric for Language Models is perplexity": perplexity = ∏_{t=1}^{T} (1 / P_LM(x^(t+1) \| x^(t), …, x^(1)))^{1/T}, the "Inverse probability of corpus, according to Language Model", "Normalized by number of words" (slide 39).
- **Key points:**
  - "This is equal to the exponential of the cross-entropy loss J(θ)": ∏_{t=1}^{T} (1/ŷ^(t)_{x_{t+1}})^{1/T} = exp((1/T) Σ_{t=1}^{T} −log ŷ^(t)_{x_{t+1}}) = exp(J(θ)). "Lower perplexity is better!" (slide 39).
  - Notes 5 §2.1: "Perplexity = 2^J" (Eq. 9), "a measure of confusion where lower values imply more confidence in predicting the next word in the sequence (compared to the ground truth outcome)".
  - Source note: Eq. 9 agrees with the slide only when J is measured with base-2 logarithms; notes 5 Eqs. 7–8 write log without a base, and the slide's exp(J) uses natural logarithms. The notes' wording, "2 to the power of the negative log probability of the cross entropy error function", is garbled: 2^J is 2 to the power of the cross-entropy itself.
- **Figures:** 1.39.

### C17 The vanishing gradient (mechanism)
- **Meaning:** Backpropagation through time multiplies one factor per step; "When these are small, the gradient signal gets smaller and smaller as it backpropagates further" (slide 45, "Vanishing gradient problem").
- **Key points:**
  - Slides 41–44 build the chain for the loss at step 4: ∂J^(4)/∂h^(1) = ∂h^(2)/∂h^(1) × ∂h^(3)/∂h^(2) × ∂h^(4)/∂h^(3) × ∂J^(4)/∂h^(4) ("chain rule!"); slide 45 asks "What happens if these are small?" (Source: "On the difficulty of training recurrent neural networks", Pascanu et al. 2013, "and supplemental materials").
  - In output-by-input Jacobian form, with A_j = diag(σ′(z_j)) W^(hh) (z_j the pre-activation at step j), ∂h_t/∂h_k = A_t A_{t−1} ⋯ A_{k+1}, and with gradients as columns ∇_{h^(1)}J^(4) = A_2ᵀ A_3ᵀ A_4ᵀ ∇_{h^(4)}J^(4). Slides 42–45's left-to-right product is the source's notation: written that way with column gradients it needs the transposed local Jacobians, and cannot use the A_j above unchanged.
  - Notes 5 §2.3: ∂E/∂W = Σ_{t=1}^{T} ∂E_t/∂W (Eq. 10); ∂E_t/∂W = Σ_{k=1}^{t} (∂E_t/∂y_t)(∂y_t/∂h_t)(∂h_t/∂h_k)(∂h_k/∂W) (Eq. 11), where ∂h_k/∂W is the local contribution at timestep k, holding h_{k−1} fixed (equivalently, differentiating that step's untied copy of W); a total derivative there would double-count earlier occurrences, which is why Pascanu et al. (Eq. 4) write the "immediate" derivative ∂⁺. Then ∂h_t/∂h_k = ∏_{j=k+1}^{t} ∂h_j/∂h_{j−1} (Eq. 12), each factor the Jacobian of h (Eq. 13); ‖∂h_j/∂h_{j−1}‖ ≤ ‖Wᵀ‖ ‖diag[f′(h_{j−1})]‖ ≤ β_W β_h (Eq. 15), so ‖∂h_t/∂h_k‖ ≤ (β_W β_h)^{t−k} (Eq. 16), which "can easily become a very small or large number when β_W β_h is much smaller or larger than 1 and t−k is sufficiently large".
  - β_W β_h < 1 gives an exponentially decaying upper bound. β_W β_h > 1 makes that upper bound grow; it does not by itself prove that the gradient explodes (Pascanu et al. §2.1 give the bound as sufficient for vanishing and the matching condition as necessary for explosion).
  - The readings: Bengio, Simard & Frasconi 1994, "one of the original vanishing gradient papers"; Pascanu et al. 2013, "proof of vanishing gradient problem"; and the "Vanishing Gradients Jupyter Notebook", a "demo for feedforward networks" comparing sigmoid and ReLU (schedule; notes 5 §2.3 "example website").
  - Source notes on notes 5: Eq. 12's right side has the typo "f′(j_{j−1})"; for h_j = σ(W^(hh)h_{j−1} + W^(hx)x_j) the Jacobian is diag(σ′(z_j)) W^(hh), not the notes' Wᵀ × diag[f′(h_{j−1})]; "The norm of f′(h_{j−1}) can only be as large as 1 given the sigmoid" is a valid but loose bound, since the sigmoid's derivative is at most 1/4; Eq. 13 indexes the hidden size as D_n where §2 used D_h.
- **Extends:** the ch03 forward reference "Example in future lecture: exploding and vanishing gradients" (ch03 slide 80): this lecture is where it is taught.
- **Figures:** 1.40, 1.41, 1.42, 1.43, 1.44, 1.45.

### C18 Why vanishing gradients hurt: long-distance dependencies
- **Meaning:** "Gradient signal from far away is lost because it's much smaller than gradient signal from close-by. So, model weights are updated only with respect to near effects, not long-term effects." (slide 46).
- **Key points:**
  - "To learn from this training example, the RNN-LM needs to model the dependency between 'tickets' on the 7th step and the target word 'tickets' at the end. But if the gradient is small, the model can't learn this dependency. So, the model is unable to predict similar long-distance dependencies at test time" (slide 47).
  - Notes 5 §2.3: "Due to vanishing gradients, we don't know whether there is no dependency between steps t and t + n in the data, or we just cannot capture the true dependency due to this issue."
- **Example:** "When she tried to print her tickets, she found that the printer was out of toner. She went to the stationery store to buy more toner. It was very overpriced. After installing the toner into the printer, she finally printed her ___" (slide 47). Notes 5 §2.3: "Jane walked into the room. John walked in too. Jane said hi to ___" and the same with a long sentence inserted; the answer is "John" in both, but "RNNs are more likely to correctly predict the blank spot in Sentence 1 than in Sentence 2".
- **Figures:** 1.46, 1.47.

### C19 The exploding gradient
- **Meaning:** "If the gradient becomes too big, then the SGD update step becomes too big": θ^new = θ^old − α∇_θJ(θ) (slide 48).
- **Key points:**
  - "This can cause bad updates: we take too large a step and reach a weird and bad parameter configuration (with large loss)"; "You think you've found a hill to climb, but suddenly you're in Iowa" (slide 48).
  - "In the worst case, this will result in Inf or NaN in your network (then you have to restart training from an earlier checkpoint)" (slide 48).
  - Notes 5 §2.3: the overflow "(i.e. NaN) … is easily detectable at runtime; this issue is called the Gradient Explosion Problem", whereas a vanishing gradient "can go undetected while drastically reducing the learning quality of the model for far-away words".
- **Figures:** 1.48.

### C20 Gradient clipping
- **Meaning:** "if the norm of the gradient is greater than some threshold, scale it down before applying SGD update" (slide 49). Algorithm 1, "Pseudo-code for norm clipping": ĝ ← ∂ℰ/∂θ; if ‖ĝ‖ ≥ threshold then ĝ ← (threshold/‖ĝ‖) ĝ.
- **Key points:**
  - "Intuition: take a step in the same direction, but a smaller step"; "In practice, remembering to clip gradients is important, but exploding gradients are an easy problem to solve" (slide 49; source: Pascanu et al. 2013).
  - Notes 5 §2.4: "Thomas Mikolov first introduced a simple heuristic solution that clips gradients". Figure 7 shows "the decision surface of a small recurrent neural network with respect to its W matrix and its bias terms, b": when gradient descent "hits the high error wall in the objective function, the gradient is pushed off to a far-away location"; "The clipping model produces the dashed line where it instead pulls back the error gradient to somewhere close to the original gradient landscape".
  - Source note: notes 5 says gradients are "set back to a small number"; Algorithm 1 rescales the gradient to norm equal to the threshold, keeping its direction.
- **Figures:** 1.49, 2.7.

### C21 Fixing the vanishing gradient
- **Meaning:** "The main problem is that it's too difficult for the RNN to learn to preserve information over many timesteps. In a vanilla RNN, the hidden state is constantly being rewritten" (slide 50). The fixes change the architecture or the training set-up.
- **Key points:**
  - "First: How about an RNN with separate memory which is added to? Long Short-Term Memory (LSTM)" (slide 50, linked to Olah's "Understanding LSTM Networks"; taught in notes 5 §4).
  - "And then: Creating more direct and linear pass-through connections in model: Attention, residual connections, etc." (slide 50).
  - Notes 5 §2.4 gives two techniques: "instead of initializing W^(hh) randomly, start off from an identity matrix initialization"; and "use the Rectified Linear Units (ReLU) instead of the sigmoid function. The derivative for the ReLU is either 0 or 1. This way, gradients would flow through the neurons whose derivative is 1 without getting attenuated while propagating back through time-steps."
- **Figures:** 1.50.

### C22 Bidirectional RNN
- **Meaning:** "It is possible to make predictions based on future words by having the RNN model read through the corpus backwards. Irsoy et al. shows a bi-directional deep neural network; at each time-step, t, this network maintains two hidden layers, one for the left-to-right propagation and another for the right-to-left propagation" (notes 5 §2.5).
- **Key points:**
  - →h_t = f(→W x_t + →V →h_{t−1} + →b) (Eq. 17); ←h_t = f(←W x_t + ←V ←h_{t+1} + ←b) (Eq. 18); "The only difference between these two relationships is in the direction of recursing through the corpus". ŷ_t = g(Uh_t + c) = g(U[→h_t; ←h_t] + c) (Eq. 19): "The final classification result, ŷ_t, is generated through combining the score results produced by both RNN hidden layers".
  - "To maintain two hidden layers at any time, this network consumes twice as much memory space for its weight and bias parameters".
  - Our reading of Eqs. 17–19: the concatenated state h_t = [→h_t; ←h_t] summarizes the words before and after position t.
  - Source note: the notes call Eq. 19 "the classification relationship used for predicting the next word via summarizing past and future word representations"; a model that reads the future words cannot serve as a next-word language model. The bidirectional state suits tagging and classification, where the whole sentence is given.
- **Figures:** 2.8.

### C23 Deep (multi-layer) RNN
- **Meaning:** "RNNs can also be multi-layered": "each lower layer feeds the next layer", and "the input to each intermediate neuron at level i is the output of the RNN at layer i−1 at the same time-step, t" (notes 5 §2.5).
- **Key points:**
  - Deep bidirectional form: →h_t^(i) = f(→W^(i) h_t^(i−1) + →V^(i) →h_{t−1}^(i) + →b^(i)) (Eq. 20); ←h_t^(i) = f(←W^(i) h_t^(i−1) + ←V^(i) ←h_{t+1}^(i) + ←b^(i)) (Eq. 21); ŷ_t = g(Uh_t + c) = g(U[→h_t^(L); ←h_t^(L)] + c) (Eq. 22): "The output, ŷ, at each time-step is the result of propagating input parameters through all hidden layers". The higher-layer input h_t^(i−1) is the concatenation [→h_t^(i−1); ←h_t^(i−1)].
  - Figure 9: each intermediate neuron "receives one set of parameters from the previous time-step (in the same RNN layer), and two sets of parameters from the previous RNN hidden layer; one input comes from the left-to-right RNN and the other from the right-to-left RNN". These are hidden-state activations: the notes loosely call them "parameters", but weights and biases are not what flows between layers or time-steps.
- **Figures:** 2.9.

### C24 Machine translation and the rise of NMT
- **Meaning:** "Machine Translation (MT) is the task of translating a sentence x from one language (the source language) to a sentence y in another language (the target language)" (slide 51).
- **Key points:**
  - "NMT: the first big success story of NLP Deep Learning": "Neural Machine Translation went from a fringe research attempt in 2014 to the leading standard method in 2016"; "2014: First seq2seq paper published [Sutskever et al. 2014]"; "2016: Google Translate switches from SMT to NMT – and by 2018 everyone has"; "SMT systems, built by hundreds of engineers over many years, outperformed by NMT systems trained by small groups of engineers in a few months" (slide 52).
  - Source note: "First seq2seq paper" is slide 52's wording, not a reliable priority claim. Cho et al.'s encoder–decoder preprint already mapped one sequence into another on 3 June 2014 (https://arxiv.org/abs/1406.1078); Sutskever et al.'s appeared on 10 September 2014 (https://arxiv.org/abs/1409.3215).
  - Notes 5 §2.6: "Traditional translation models are quite complex; they consist of numerous machine learning algorithms applied to different stages of the language translation pipeline."
- **Example:** x "I like deep learning" → y "我喜欢深度学习" (slide 51).
- **Extends:** ch01:C34 (neural MT overtaking statistical MT); Lecture 1 used the task itself as stud ch01:B02, and this lecture defines it.
- **Figures:** 1.51, 1.52.

### C25 The sequence-to-sequence (encoder–decoder) model
- **Meaning:** "Encoder RNN produces an encoding of the source sentence"; that "Encoding of the source sentence" "Provides initial hidden state for Decoder RNN"; and the "Decoder RNN is a Language Model that generates target sentence, conditioned on encoding" (slide 53). "The general notion here is an encoder-decoder model": "One neural network takes input and produces a neural representation"; "Another network produces output based on that neural representation"; "If the input and output are sequences, we call it a seq2seq model" (slide 54).
- **Key points:**
  - "Note: This diagram shows test time behavior: decoder output is fed in as next step's input", taking the argmax word at each step (slide 53).
  - "Sequence-to-sequence is useful for more than just MT. Many NLP tasks can be phrased as sequence-to-sequence: Summarization (long text → short text); Dialogue (previous utterances → next utterance); Parsing (input text → output parse as sequence); Code generation (natural language → Python code)" (slide 54).
  - "Multi-layer deep encoder-decoder machine translation net" [Sutskever et al. 2014; Luong et al. 2015]: "The hidden states from RNN layer i are the inputs to RNN layer i+1"; "Encoder: Builds up sentence meaning"; "Feeding in last word"; "Translation generated"; "Conditioning = Bottleneck" (slide 57).
  - "The final piece: the bottleneck problem in RNNs": the whole source reaches the decoder through one "Encoding of the source sentence", and the slide asks "Problems with this architecture?" (slide 58). That a single fixed-size vector must carry the whole source sentence is our reading of the slide, which leaves the question open for a later lecture.
  - Notes 5 §2.6: encoder h_t = φ(h_{t−1}, x_t) = f(W^(hh)h_{t−1} + W^(hx)x_t) (Eq. 23); decoder h_t = φ(h_{t−1}) = f(W^(hh)h_{t−1}) (Eq. 24) and y_t = softmax(W^(S)h_t) (Eq. 25). Extensions: "Extension I: train different RNN weights for encoding and decoding"; "Extension II": each decoder state from "The previous hidden state", the "Last hidden layer of the encoder (c = h_T in Figure 11)" and the "Previous predicted output word, ŷ_{t−1}", h_t = φ(h_{t−1}, c, y_{t−1}) (Eq. 27); "Extension III": deep RNNs; "Extension IV": "bi-directional encoders"; "Extension V": train on C B A → X Y instead of A B C → X Y, since "A is more likely to be translated to X. Thus, given the vanishing gradient problem discussed earlier, reversing the order of the input words can help reduce the error rate".
  - Source note: slide 58 writes the source sentence "il a m' entarté", where slide 53 has "il m' a entarté".
- **Example:** il m' a entarté → he hit me with a pie `<END>` (slide 53); Die Proteste waren am Wochenende eskaliert → The protests escalated over the weekend (slide 57); Echt dicke Kiste → Awesome sauce (notes 5 Figure 10).
- **Extends:** ch01:C33, which gave the encoder–decoder's history, diagrams and bottleneck; this lecture gives the model, its test-time decoding, its uses beyond MT and its variants.
- **Figures:** 1.53, 1.54, 1.57, 1.58, 2.10, 2.11.

### C26 Seq2seq as a conditional language model, trained end to end
- **Meaning:** "The sequence-to-sequence model is an example of a Conditional Language Model": "Language Model because the decoder is predicting the next word of the target sentence y"; "Conditional because its predictions are also conditioned on the source sentence x". "NMT directly calculates P(y\|x)": P(y\|x) = P(y₁\|x) P(y₂\|y₁, x) P(y₃\|y₁, y₂, x) … P(y_T\|y₁, …, y_{T−1}, x), each factor the "Probability of next target word, given target words so far and source sentence x" (slide 55).
- **Key points:**
  - "Question: How to train an NMT system?" "(Easy) Answer: Get a big parallel corpus…" "But there is now exciting work on 'unsupervised NMT', data augmentation, etc." (slide 55).
  - Training: J = (1/T) Σ_{t=1}^{T} J_t, each J_t the negative log probability of the true next target word ("he", "with", `<END>`), with the source and target sentences "from corpus"; "Seq2seq is optimized as a single system. Backpropagation operates 'end-to-end'" (slide 56).
  - Notes 5 Eq. 26: max_θ (1/N) Σ_{n=1}^{N} log p_θ(y^(n)\|x^(n)).
- **Figures:** 1.55, 1.56.

### C27 Gated recurrent unit (GRU)
- **Meaning:** "although RNNs can theoretically capture long-term dependencies, they are very hard to actually train to do this. Gated recurrent units are designed in a manner to have more persistent memory thereby making it easier for RNNs to capture long-term dependencies" (notes 5 §3). The unit: z_t = σ(W^(z)x_t + U^(z)h_{t−1}) (Update gate); r_t = σ(W^(r)x_t + U^(r)h_{t−1}) (Reset gate); h̃_t = tanh(r_t ∘ Uh_{t−1} + Wx_t) (New memory); h_t = (1 − z_t) ∘ h̃_t + z_t ∘ h_{t−1} (Hidden state).
- **Key points:**
  - The four stages (Figure 12): "New memory generation", where h̃_t "is the consolidation of a new input word x_t with the past hidden state h_{t−1}"; the "Reset Gate", "responsible for determining how important h_{t−1} is to the summarization h̃_t", which "has the ability to completely diminish past hidden state if it finds that h_{t−1} is irrelevant to the computation of the new memory"; the "Update Gate", "responsible for determining how much of h_{t−1} should be carried forward to the next state. For instance, if z_t ≈ 1, then h_{t−1} is almost entirely copied out to h_t. Conversely, if z_t ≈ 0, then mostly the new memory h̃_t is forwarded to the next hidden state"; and the "Hidden state", generated from h_{t−1} and h̃_t "with the advice of the update gate".
  - "to train a GRU, we need to learn all the different parameters: W, U, W^(r), U^(r), W^(z), U^(z). These follow the same backpropagation procedure we have seen in the past."
  - Source note: the notes apply the reset gate after the recurrent matrix, r_t ∘ Uh_{t−1}; Cho et al. 2014 (Eq. 8) apply it before, h̃ = φ(Wx + U(r ⊙ h_{t−1})). Both use the update convention h_t = z ∘ h_{t−1} + (1 − z) ∘ h̃. Cho et al. present the unit as "a new type of hidden unit … motivated by the LSTM unit but is much simpler to compute and implement". The notes' equations have no bias terms.
- **Figures:** 2.12.

### C28 Long short-term memory (LSTM)
- **Meaning:** "Long-Short-Term-Memories are another type of complex activation unit that differ a little from GRUs. The motivation for using these is similar to those for GRUs however the architecture of such units does differ" (notes 5 §4): i_t = σ(W^(i)x_t + U^(i)h_{t−1}) (Input gate); f_t = σ(W^(f)x_t + U^(f)h_{t−1}) (Forget gate); o_t = σ(W^(o)x_t + U^(o)h_{t−1}) (Output/Exposure gate); c̃_t = tanh(W^(c)x_t + U^(c)h_{t−1}) (New memory cell); c_t = f_t ∘ c_{t−1} + i_t ∘ c̃_t (Final memory cell); h_t = o_t ∘ tanh(c_t).
- **Key points:**
  - The five stages (Figure 13): "New memory generation", "analogous to the new memory generation stage we saw in GRUs"; the "Input Gate", which determines "whether or not the input is worth preserving"; the "Forget Gate", which assesses "whether the past memory cell is useful for the computation of the current memory cell"; "Final memory generation", which "takes the advice of the forget gate f_t and accordingly forgets the past memory c_{t−1}", gates the new memory with i_t, and "sums these two results"; and the "Output/Exposure Gate", which "does not explicitly exist in GRUs. It's purpose is to separate the final memory from the hidden state", deciding "what parts of the memory c_t needs to be exposed/present in the hidden state h_t".
  - Slide 50: "How about an RNN with separate memory which is added to? Long Short-Term Memory (LSTM)", linked to Olah's "Understanding LSTM Networks", whose figures draw the same cell. Our reading: c_t = f_t ∘ c_{t−1} + i_t ∘ c̃_t is that "separate memory which is added to".
  - Lecture 1 used LSTM internals as a stud (ch01:B12); this lecture teaches them.
  - The original: Hochreiter & Schmidhuber 1997 build each memory cell "around a central linear unit with a fixed self-connection", the "constant error carrousel" (CEC), with "a multiplicative input gate unit" and "a multiplicative output gate unit"; the 1997 paper has no forget gate (the word does not occur in it).
- **Figures:** 1.50, 2.13.

## Problems

What each end-of-chapter problem asks and which concepts it exercises. Record no solutions the source doesn't give.

| Problem | Asks | Exercises |
|---|---|---|

No problems: Lecture 4 releases no assignment (A2, out with Lecture 3, is due Jan 22), and notes 5 has no exercises.

## Forward references

Topics the chapter defers. None is taught here; later extraction EXTENDs from these hooks.

| Topic | Chapter | Claim / promised treatment |
|---|---|---|
| Attention, and residual connections | later lectures (Lecture 5, Transformers, on the W26 schedule) | "And then: Creating more direct and linear pass-through connections in model: Attention, residual connections, etc." (slide 50) |
| The bottleneck's answer (attention) | a later lecture | "Problems with this architecture?" (slide 58), left open |

## Studs

Concepts the chapter uses but does not teach. A stud means "not taught here", not "nothing more to learn".

<!-- Entry shape (numbering runs B01, B02, …):
 ### B01 <Label>
- **Meaning:** one line.
- **Used at:** §3.1 "short quote"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.
  (Book mode, overlay §8: when an earlier chapter's map teaches it, write `Taught in ch02:C05` instead.)
-->

### B01 Conditional probability and the chain rule of probability
- **Meaning:** P(A \| B) = P(A, B)/P(B), and a joint probability factors into a product of conditionals.
- **Used at:** slide 4's factorisation of P(x^(1), …, x^(T)); slide 10 ("definition of conditional prob").
- **Status:** not taught here (also a stud in ch01:B10 and ch02:B02). Further prerequisites are not assessed in this chapter.

### B02 Word embeddings and the embedding lookup
- **Meaning:** a word's dense vector, picked out of an embedding matrix by its one-hot vector.
- **Used at:** slide 21 "concatenated word embeddings"; slide 24 e^(t) = Ex^(t).
- **Status:** Taught in ch03:C04 (also ch02:C07).

### B03 The softmax function
- **Meaning:** turns a vector of scores into a probability distribution.
- **Used at:** slide 21 ŷ = softmax(Uh + b₂); notes 5 Eqs. 4, 6.
- **Status:** Taught in ch02:C13.

### B04 A one-hidden-layer feed-forward network
- **Meaning:** h = f(Wx + b) followed by an output layer.
- **Used at:** slide 21 h = f(We + b₁); notes 5 Eq. 4.
- **Status:** Taught in ch03:C06.

### B05 Stochastic gradient descent
- **Meaning:** update the parameters with the gradient of the loss on a small sample (a batch) of the data.
- **Used at:** slide 32 "Recall: Stochastic Gradient Descent".
- **Status:** Taught in ch02:C16.

### B06 Cross-entropy loss
- **Meaning:** −Σ_c p(c) log q(c); with a one-hot target, the negative log probability of the true class.
- **Used at:** slide 26; notes 5 Eq. 7 ("the cross entropy error introduced in earlier notes").
- **Status:** Taught in ch03:C11.

### B07 Backpropagation
- **Meaning:** computing every gradient of a scalar loss in one backward pass over the computation graph, upstream gradient times local gradient.
- **Used at:** slides 33–35.
- **Status:** Taught in ch03:C24 (general graphs: ch03:C27).

### B08 Logarithms and exponentials, including their bases
- **Meaning:** log and exp as inverses; a logarithm in base 2 differs from the natural logarithm by a constant factor.
- **Used at:** slide 39 "exponential of the cross-entropy loss"; notes 5 Eq. 9 "Perplexity = 2^J".
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B09 Jacobians and the chain rule for vector functions
- **Meaning:** the matrix of partial derivatives of a vector function, and the chain rule as a product of Jacobians.
- **Used at:** slides 42–45; notes 5 Eqs. 12–13.
- **Status:** Taught in ch03:C15, ch03:C16.

### B10 Activation functions and their derivatives (sigmoid, ReLU)
- **Meaning:** σ(z) with σ′(z) = σ(z)(1 − σ(z)); ReLU with derivative 0 or 1.
- **Used at:** notes 5 Eq. 15 and §2.4.
- **Status:** Taught in ch03:C07.

### B11 Matrix and vector norms
- **Meaning:** a size measure ‖·‖ for vectors and matrices, with ‖AB‖ ≤ ‖A‖‖B‖ for the matrix (operator) norm.
- **Used at:** notes 5 Eqs. 15–16 ("taking their L2-norm"); slide 49 ‖ĝ‖.
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B12 Element-wise (Hadamard) product ∘
- **Meaning:** multiplying two vectors entry by entry.
- **Used at:** notes 5 §3–4 (r_t ∘ Uh_{t−1}, f_t ∘ c_{t−1}, o_t ∘ tanh(c_t)).
- **Status:** Taught in ch03:C19.

### B13 Identity matrix
- **Meaning:** the square matrix I with Ix = x.
- **Used at:** notes 5 §2.4 "start off from an identity matrix initialization".
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

## Prerequisites

Every row reads "From `prerequisite_for` To"; a learner walks backward from To. The prerequisite rows must form
no cycle. A **scope** means the prerequisite is needed only for that part of To.

| From | Predicate | To | Family | Status | Scope | Basis |
|---|---|---|---|---|---|---|
| B01 Conditional probability and the chain rule of probability | `prerequisite_for` | C01 Language model | prerequisite | inferred | the probability-of-a-text view | slide 4 factorisation |
| C01 Language model | `prerequisite_for` | C02 Why language modeling matters | prerequisite | inferred | — | slides 5–7 are about LMs |
| C01 Language model | `prerequisite_for` | C03 What next-word prediction can teach a model | prerequisite | inferred | — | slide 8 "next-word prediction" |
| C01 Language model | `prerequisite_for` | C04 n-gram language model | prerequisite | stated | — | slide 9 "How to learn a Language Model?" |
| B01 Conditional probability and the chain rule of probability | `prerequisite_for` | C04 n-gram language model | prerequisite | stated | the conditional-probability step | slide 10 "definition of conditional prob" |
| C04 n-gram language model | `prerequisite_for` | C05 Sparsity in n-gram LMs | prerequisite | stated | — | slide 12 "Sparsity Problems with n-gram Language Models" |
| C04 n-gram language model | `prerequisite_for` | C06 Storage and the context-length trade-off | prerequisite | stated | — | slide 13 "Storage Problems with n-gram Language Models" |
| C01 Language model | `prerequisite_for` | C07 Generating text with a language model | prerequisite | stated | — | slide 15 "use a Language Model to generate text" |
| C04 n-gram language model | `prerequisite_for` | C07 Generating text with a language model | prerequisite | inferred | the n-gram generation example | slides 15–18 |
| C01 Language model | `prerequisite_for` | C08 Fixed-window neural language model | prerequisite | stated | — | slide 19 "Recall the Language Modeling task" |
| B02 Word embeddings and the embedding lookup | `prerequisite_for` | C08 Fixed-window neural language model | prerequisite | stated | — | slide 21 "concatenated word embeddings" |
| B03 The softmax function | `prerequisite_for` | C08 Fixed-window neural language model | prerequisite | stated | — | slide 21 ŷ = softmax(Uh + b₂) |
| B04 A one-hidden-layer feed-forward network | `prerequisite_for` | C08 Fixed-window neural language model | prerequisite | stated | — | slide 21 h = f(We + b₁) |
| C08 Fixed-window neural language model | `prerequisite_for` | C09 What the fixed window fixes, and what it can't | prerequisite | stated | — | slide 22 |
| C05 Sparsity in n-gram LMs | `prerequisite_for` | C09 What the fixed window fixes, and what it can't | prerequisite | stated | the improvements over n-grams | slide 22 "No sparsity problem" |
| C06 Storage and the context-length trade-off | `prerequisite_for` | C09 What the fixed window fixes, and what it can't | prerequisite | stated | the improvements over n-grams | slide 22 "Don't need to store all observed n-grams" |
| B04 A one-hidden-layer feed-forward network | `prerequisite_for` | C10 Recurrent neural network (RNN) | prerequisite | inferred | — | a recurrent hidden layer, slide 23 |
| C10 Recurrent neural network (RNN) | `prerequisite_for` | C11 RNN language model | prerequisite | stated | — | slide 24 "A Simple RNN Language Model" |
| C01 Language model | `prerequisite_for` | C11 RNN language model | prerequisite | stated | — | slide 24 |
| B02 Word embeddings and the embedding lookup | `prerequisite_for` | C11 RNN language model | prerequisite | stated | — | slide 24 e^(t) = Ex^(t) |
| B03 The softmax function | `prerequisite_for` | C11 RNN language model | prerequisite | stated | — | slide 24 ŷ^(t) = softmax(Uh^(t) + b₂) |
| C11 RNN language model | `prerequisite_for` | C12 RNN advantages and disadvantages | prerequisite | stated | — | slide 25 "RNN Language Models" |
| C08 Fixed-window neural language model | `prerequisite_for` | C12 RNN advantages and disadvantages | prerequisite | inferred | the symmetry and model-size comparisons | slides 22, 25 |
| C10 Recurrent neural network (RNN) | `prerequisite_for` | C13 RNNs beyond language modeling | prerequisite | stated | — | notes 5 §2.2 |
| C11 RNN language model | `prerequisite_for` | C14 Training an RNN language model | prerequisite | stated | — | slide 26 "Feed into RNN-LM" |
| B06 Cross-entropy loss | `prerequisite_for` | C14 Training an RNN language model | prerequisite | stated | — | slide 26 "cross-entropy" |
| B05 Stochastic gradient descent | `prerequisite_for` | C14 Training an RNN language model | prerequisite | stated | the batch-of-sentences update | slide 32 |
| C10 Recurrent neural network (RNN) | `prerequisite_for` | C15 Backpropagation through time (BPTT) | prerequisite | stated | — | slide 33 "the repeated weight matrix W_h" |
| C14 Training an RNN language model | `prerequisite_for` | C15 Backpropagation through time (BPTT) | prerequisite | stated | — | slide 33 "derivative of J^(t)(θ)" |
| B07 Backpropagation | `prerequisite_for` | C15 Backpropagation through time (BPTT) | prerequisite | stated | — | slide 33 "Backpropagation for RNNs" |
| C11 RNN language model | `prerequisite_for` | C07 Generating text with a language model | prerequisite | stated | RNN rollouts | slide 36 |
| C01 Language model | `prerequisite_for` | C16 Perplexity | prerequisite | stated | — | slide 39 "evaluation metric for Language Models" |
| C14 Training an RNN language model | `prerequisite_for` | C16 Perplexity | prerequisite | stated | the exp(J(θ)) identity | slide 39 "equal to the exponential of the cross-entropy loss J(θ)" |
| B08 Logarithms and exponentials, including their bases | `prerequisite_for` | C16 Perplexity | prerequisite | stated | the exp(J) and 2^J relationships | slide 39; notes 5 Eq. 9 |
| C15 Backpropagation through time (BPTT) | `prerequisite_for` | C17 The vanishing gradient (mechanism) | prerequisite | stated | — | slide 45 "as it backpropagates further" |
| B09 Jacobians and the chain rule for vector functions | `prerequisite_for` | C17 The vanishing gradient (mechanism) | prerequisite | stated | — | slides 42–44 "chain rule!" |
| B11 Matrix and vector norms | `prerequisite_for` | C17 The vanishing gradient (mechanism) | prerequisite | stated | the norm bound | notes 5 Eqs. 15–16 |
| B10 Activation functions and their derivatives (sigmoid, ReLU) | `prerequisite_for` | C17 The vanishing gradient (mechanism) | prerequisite | stated | the f′ factor | notes 5 Eq. 15 |
| C17 The vanishing gradient (mechanism) | `prerequisite_for` | C18 Why vanishing gradients hurt: long-distance dependencies | prerequisite | stated | — | slide 46 "Why is vanishing gradient a problem?" |
| C11 RNN language model | `prerequisite_for` | C18 Why vanishing gradients hurt: long-distance dependencies | prerequisite | stated | the LM example | slide 47 "Effect of vanishing gradient on RNN-LM" |
| C15 Backpropagation through time (BPTT) | `prerequisite_for` | C19 The exploding gradient | prerequisite | inferred | — | slides 40, 48 |
| B05 Stochastic gradient descent | `prerequisite_for` | C19 The exploding gradient | prerequisite | stated | — | slide 48 "the SGD update step" |
| C19 The exploding gradient | `prerequisite_for` | C20 Gradient clipping | prerequisite | stated | — | slide 49 "solution for exploding gradient" |
| B11 Matrix and vector norms | `prerequisite_for` | C20 Gradient clipping | prerequisite | stated | — | slide 49 "the norm of the gradient" |
| C17 The vanishing gradient (mechanism) | `prerequisite_for` | C21 Fixing the vanishing gradient | prerequisite | stated | — | slide 50 "How to fix the vanishing gradient problem?" |
| B10 Activation functions and their derivatives (sigmoid, ReLU) | `prerequisite_for` | C21 Fixing the vanishing gradient | prerequisite | stated | the ReLU technique | notes 5 §2.4 |
| C10 Recurrent neural network (RNN) | `prerequisite_for` | C22 Bidirectional RNN | prerequisite | stated | — | notes 5 §2.5 |
| C10 Recurrent neural network (RNN) | `prerequisite_for` | C23 Deep (multi-layer) RNN | prerequisite | stated | — | notes 5 §2.5 "RNNs can also be multi-layered" |
| C22 Bidirectional RNN | `prerequisite_for` | C23 Deep (multi-layer) RNN | prerequisite | stated | the deep bidirectional form of Eqs. 20–22 | notes 5 §2.5, Figure 9 |
| C24 Machine translation and the rise of NMT | `prerequisite_for` | C25 The sequence-to-sequence (encoder–decoder) model | prerequisite | stated | — | slide 53 "Neural Machine Translation (NMT)" |
| C11 RNN language model | `prerequisite_for` | C25 The sequence-to-sequence (encoder–decoder) model | prerequisite | stated | — | slide 53 "Decoder RNN is a Language Model" |
| C23 Deep (multi-layer) RNN | `prerequisite_for` | C25 The sequence-to-sequence (encoder–decoder) model | prerequisite | stated | the multi-layer net and Extension III | slide 57; notes 5 §2.6 |
| C22 Bidirectional RNN | `prerequisite_for` | C25 The sequence-to-sequence (encoder–decoder) model | prerequisite | stated | Extension IV | notes 5 §2.6 |
| C17 The vanishing gradient (mechanism) | `prerequisite_for` | C25 The sequence-to-sequence (encoder–decoder) model | prerequisite | stated | Extension V's vanishing-gradient explanation | notes 5 §2.6 |
| C25 The sequence-to-sequence (encoder–decoder) model | `prerequisite_for` | C26 Seq2seq as a conditional language model, trained end to end | prerequisite | stated | — | slide 55 |
| C01 Language model | `prerequisite_for` | C26 Seq2seq as a conditional language model, trained end to end | prerequisite | stated | — | slide 55 "Conditional Language Model" |
| C14 Training an RNN language model | `prerequisite_for` | C26 Seq2seq as a conditional language model, trained end to end | prerequisite | inferred | the training | slide 56 mirrors slide 26 |
| B07 Backpropagation | `prerequisite_for` | C26 Seq2seq as a conditional language model, trained end to end | prerequisite | stated | end-to-end backpropagation | slide 56 |
| C10 Recurrent neural network (RNN) | `prerequisite_for` | C27 Gated recurrent unit (GRU) | prerequisite | stated | — | notes 5 §3 "modifying the RNN architecture" |
| B10 Activation functions and their derivatives (sigmoid, ReLU) | `prerequisite_for` | C27 Gated recurrent unit (GRU) | prerequisite | stated | the sigmoid gates and tanh | notes 5 §3 |
| B12 Element-wise (Hadamard) product ∘ | `prerequisite_for` | C27 Gated recurrent unit (GRU) | prerequisite | stated | — | notes 5 §3 |
| B07 Backpropagation | `prerequisite_for` | C27 Gated recurrent unit (GRU) | prerequisite | stated | the training paragraph | notes 5 §3 "the same backpropagation procedure" |
| C10 Recurrent neural network (RNN) | `prerequisite_for` | C28 Long short-term memory (LSTM) | prerequisite | stated | — | notes 5 §4 |
| B10 Activation functions and their derivatives (sigmoid, ReLU) | `prerequisite_for` | C28 Long short-term memory (LSTM) | prerequisite | stated | the sigmoid gates and tanh | notes 5 §4 |
| B12 Element-wise (Hadamard) product ∘ | `prerequisite_for` | C28 Long short-term memory (LSTM) | prerequisite | stated | — | notes 5 §4 |
| C27 Gated recurrent unit (GRU) | `prerequisite_for` | C28 Long short-term memory (LSTM) | prerequisite | stated | the notes' comparisons with GRUs | notes 5 §4 "differ a little from GRUs" |
| B13 Identity matrix | `prerequisite_for` | C21 Fixing the vanishing gradient | prerequisite | stated | the identity initialization | notes 5 §2.4 |

<!-- Row shape: | B01 Label | `prerequisite_for` | C01 Label | prerequisite | inferred | — | §3.1 "short quote" | -->

### Per-concept review

| Concept | Direct prerequisites and notes |
|---|---|
| C01 Language model | B01 (the probability-of-a-text view) (Lecture 1: ch01:C36) |
| C02 Why language modeling matters | C01 |
| C03 What next-word prediction can teach a model | C01 |
| C04 n-gram language model | C01; B01 (the conditional-probability step) |
| C05 Sparsity in n-gram LMs | C04 |
| C06 Storage and the context-length trade-off | C04 |
| C07 Generating text with a language model | C01; C04 (the n-gram example); C11 (RNN rollouts) |
| C08 Fixed-window neural language model | C01; B02; B03; B04 (Lecture 3: ch03:C06, ch03:C02) |
| C09 What the fixed window fixes, and what it can't | C08; C05, C06 (the improvements over n-grams) |
| C10 Recurrent neural network (RNN) | B04 (motivated by C09 in the teaching arc, not a prerequisite) |
| C11 RNN language model | C10; C01; B02; B03 |
| C12 RNN advantages and disadvantages | C11; C08 (the symmetry and model-size comparisons) |
| C13 RNNs beyond language modeling | C10 |
| C14 Training an RNN language model | C11; B06; B05 (the batch-of-sentences update) |
| C15 Backpropagation through time (BPTT) | C10; C14; B07 (Lecture 3: ch03:C25, ch03:C16) |
| C16 Perplexity | C01; C14 (the exp(J(θ)) identity); B08 (the exp(J) and 2^J relationships) |
| C17 The vanishing gradient (mechanism) | C15; B09; B11 (the norm bound); B10 (the f′ factor) |
| C18 Why vanishing gradients hurt | C17; C11 (the LM example) |
| C19 The exploding gradient | C15; B05 |
| C20 Gradient clipping | C19; B11 |
| C21 Fixing the vanishing gradient | C17; B10 (the ReLU technique); B13 (the identity initialization) |
| C22 Bidirectional RNN | C10 |
| C23 Deep (multi-layer) RNN | C10; C22 (the deep bidirectional form) |
| C24 Machine translation and the rise of NMT | none in this lecture (Lecture 1: ch01:C34) |
| C25 The sequence-to-sequence model | C24; C11; C23 (multi-layer, Extension III); C22 (Extension IV); C17 (Extension V) (Lecture 1: ch01:C33) |
| C26 Seq2seq as a conditional LM, trained end to end | C25; C01; C14 (the training); B07 (end-to-end backpropagation) |
| C27 Gated recurrent unit (GRU) | C10; B10 (gates and tanh); B12; B07 (the training paragraph) |
| C28 Long short-term memory (LSTM) | C10; B10; B12; C27 (the GRU comparisons) |

## Predicates

The controlled list is `predicates.md` in this pack. List here only predicates this map ADDED with the
Reviewer's agreement, each with its family and meaning.

## Links

Functional links from the text, each marked **stated** (the author says it) or **inferred** (our reading, with its reason).

| From | Predicate | To | Family | Status | Scope / original wording | Basis |
|---|---|---|---|---|---|---|
| C03 What next-word prediction can teach a model | `illustrates` | C02 Why language modeling matters | functional | inferred | slide 8 shows what a strong LM can do, part of why LMs are worth building | slides 7–8 |
| C07 Generating text with a language model | `uses` | C01 Language model | functional | stated | "You can also use a Language Model to generate text" | slide 15 |
| C08 Fixed-window neural language model | `contrasts_with` | C04 n-gram language model | functional | stated | "Improvements over n-gram LM" | slide 22 |
| C10 Recurrent neural network (RNN) | `applies_to` | C01 Language model | functional | stated | "Language models with neural nets: Recurrent Neural Networks (RNNs)" | slide 2 |
| C10 Recurrent neural network (RNN) | `contrasts_with` | C08 Fixed-window neural language model | functional | inferred | the "any length" and "symmetry" advantages answer slide 22's remaining problems | slides 22, 25 |
| C15 Backpropagation through time (BPTT) | `method_for` | C14 Training an RNN language model | functional | stated | "Training the parameters of RNNs: Backpropagation for RNNs" | slide 35 |
| C16 Perplexity | `assesses` | C01 Language model | functional | stated | "The standard evaluation metric for Language Models is perplexity" | slide 39 |
| C07 Generating text with a language model | `contrasts_with` | C14 Training an RNN language model | functional | inferred | in generation the "Sampled output becomes next step's input"; training feeds the corpus words ("Teacher forcing") | slides 31, 36 |
| C17 The vanishing gradient (mechanism) | `can_impair` | C14 Training an RNN language model | functional | stated | "model weights are updated only with respect to near effects, not long-term effects" | slide 46 |
| C19 The exploding gradient | `can_impair` | C14 Training an RNN language model | functional | stated | "This can cause bad updates" | slide 48 |
| C20 Gradient clipping | `applies_to` | C19 The exploding gradient | functional | stated | "Gradient clipping: solution for exploding gradient" | slide 49 |
| C21 Fixing the vanishing gradient | `applies_to` | C17 The vanishing gradient (mechanism) | functional | stated | "How to fix the vanishing gradient problem?" | slide 50 |
| C22 Bidirectional RNN | `kind_of` | C10 Recurrent neural network (RNN) | taxonomic | stated | "a bi-directional deep neural network" | notes 5 §2.5 |
| C23 Deep (multi-layer) RNN | `kind_of` | C10 Recurrent neural network (RNN) | taxonomic | stated | "RNNs can also be multi-layered" | notes 5 §2.5 |
| C25 The sequence-to-sequence (encoder–decoder) model | `applies_to` | C24 Machine translation and the rise of NMT | functional | stated | "Neural Machine Translation (NMT): The sequence-to-sequence model" | slide 53 |
| C26 Seq2seq as a conditional language model, trained end to end | `kind_of` | C01 Language model | taxonomic | stated | "an example of a Conditional Language Model" | slide 55 |
| C27 Gated recurrent unit (GRU) | `applies_to` | C18 Why vanishing gradients hurt: long-distance dependencies | functional | stated | "making it easier for RNNs to capture long-term dependencies" | notes 5 §3 |
| C28 Long short-term memory (LSTM) | `applies_to` | C17 The vanishing gradient (mechanism) | functional | stated | "How to fix the vanishing gradient problem? … Long Short-Term Memory (LSTM)" | slide 50 |
| C27 Gated recurrent unit (GRU) | `contrasts_with` | C28 Long short-term memory (LSTM) | functional | stated | the output gate "does not explicitly exist in GRUs" | notes 5 §4 |

<!-- Row shape: | C02 Label | `has_parameters` | C03 Label | part-whole | stated | — | "short quote" (§3.1) | -->
