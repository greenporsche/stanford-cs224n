# Concept map: *Stanford CS224N (Winter 2026)*, Chapter 2 (Lecture 2: Word Vectors)

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

1. **The plan and the goal** (slides 1–2, 44). The lecture (title slide: Diyi Yang) runs in six timed parts: "Course organization (3 mins)", "Word2vec introduction (15 mins)", "Word2vec objective function gradients (25 mins)", "Optimization basics (5 mins)", "Can we capture the essence of word meaning more effectively by counting? (10m)" and "Evaluating word vectors (10 mins)". Its key goal: "understand word meaning can be represented by a high-dimensional vector of real numbers and can read word embeddings papers by the end of class" (slide 2). Slide 44 repeats the plan as the close; slide 3 is course logistics.

2. **What is meaning, and how does a computer hold it?** (slides 4–10; notes 1 §2–3). The lecture starts from denotation (C01), shows the old answer, a hand-built thesaurus (C02), and its failures, and shows that one-hot symbols (C04) make every pair of words equally unlike. Notes 1 adds hand-annotated feature vectors (C03) and the type/token distinction (C05). The way out is "learn to encode similarity in the vectors themselves" (slide 8): represent a word by its contexts (C06), as a dense vector (C07).

3. **Word2vec, the model** (slides 11–17; notes 1 §3.2). Word2vec (C08) walks a window along the text (C09), scores each outside word against the centre word with two vectors per word (C11) and a softmax (C12, C13), and is trained by minimizing the average negative log-likelihood (C10).

4. **The gradient** (slides 18–19; notes 1 §3.3–3.4). θ gathers all 2dV numbers (C11), and training needs "all vector gradients" (slide 18). The deck's interactive session (slide 19) has no worked derivation; notes 1 derives the centre-vector gradient as "observed" minus "expected" (C14).

5. **Optimization** (slides 20–24; notes 1 §3.3). Gradient descent steps against the gradient (C15). Because J covers every window in the corpus, stochastic gradient descent updates after sampled windows (C16). Word2vec's predictions are the same at every position in the window (C17), and training puts similar words near each other (C08).

6. **The word2vec family and negative sampling** (slides 25–29; notes 1 §3.5). Skip-gram has a sibling, CBOW (C18). The naïve softmax's normalizer is expensive (C19), so standard word2vec trains skip-gram with negative sampling (C20), using the sigmoid (C21). As an aside, each window's gradient touches only a few rows of U and V (C22).

7. **Counting instead of predicting** (slides 30–35; notes 1 §3.1). "Why not capture co-occurrence counts directly?" (slide 30): build a co-occurrence matrix (C23), reduce it with SVD (C24), and scale the counts first (C25). In the scaled vectors, linear semantic patterns appear (C26), which sets up GloVe.

8. **GloVe** (slides 36–37; notes 2 §1). Ratios of co-occurrence probabilities carry meaning components (C27). GloVe (C28) fits word vectors whose dot products match log co-occurrence counts, so that vector differences encode those ratios, joining the count-based and prediction-based families.

9. **How good are the vectors?** (slides 38–43; notes 2 §2). Evaluation is intrinsic or extrinsic (C29). Intrinsic tests are analogies (C30), which also tune hyperparameters (C31), and correlation with human similarity ratings (C32); the extrinsic example is named entity recognition (C33). Notes 2 adds one vector per word sense (C34).

10. **Using word vectors for tasks** (notes 2 §3). Extrinsic tasks are mostly classification (C35). Word vectors can be retrained on the task, but only with enough data (C36). The softmax classifier with cross-entropy (C37) needs regularization (C38), works on windows of words (C39), and its linear boundary motivates non-linear classifiers (C40): the neural networks of Lecture 3.

## Assumed background

What the chapter uses without teaching it is recorded as **studs** (B01…, below).

**Chapter scope:** the Winter 2026 Lecture 2 deck, all 44 PDF pages (slide N = PDF page N; printed numbers run to 51, and printed slides 20–26 are not in the PDF); notes 1 (Winter 2023 draft, "Note 1: Introduction and Word2Vec", 13 pp., including its margin notes, appendix headings and references); notes 2 (Winter 2019, "Word Vectors II: GloVe, Evaluation and Training", 13 pp., including margin notes, tables and Figures 1–10). The GloVe paper and the COALS manuscript were consulted where a slide relies on them (Table 1, Eq. 9 and the captions of Tables 3–4; the COALS figure's projection), and are cited as such. Lecture 1's map (ch01) is the earlier chapter: entries it already teaches are referenced as `Taught in` or `**Extends:**`.

## Figures

Concepts cite figures by number. Images are linked, never committed.

Figure numbers: 1.N is slide N (PDF page N of the W26 deck). 3.N is notes 2's Figure N, linked to a render of the notes page that holds it (`materials/figures/notes2-pNN.png`). Image links are relative to the published lecture folder, where the slide images sit beside the map.

| Fig | Shows | Teaches | Image |
|---|---|---|---|
| 1.4 | signifier ⟺ signified: tree ⟺ {pictures of three trees, …} | C01 | [1.4](materials/slides/cs224n-2026-lecture02-wordvecs/p04.png) |
| 1.5 | NLTK code and output: synonym sets containing "good"; hypernyms of "panda" | C02 | [1.5](materials/slides/cs224n-2026-lecture02-wordvecs/p05.png) |
| 1.7 | one-hot vectors for motel and hotel | C04 | [1.7](materials/slides/cs224n-2026-lecture02-wordvecs/p07.png) |
| 1.9 | three contexts of "banking", with Firth's photograph | C06 | [1.9](materials/slides/cs224n-2026-lecture02-wordvecs/p09.png) |
| 1.10 | 8-dimensional vectors for banking and monetary | C07 | [1.10](materials/slides/cs224n-2026-lecture02-wordvecs/p10.png) |
| 1.11 | skip-gram architecture: Input w(t) → projection → output w(t−2) … w(t+2) (Mikolov et al. 2013) | C08, C09 | [1.11](materials/slides/cs224n-2026-lecture02-wordvecs/p11.png) |
| 1.12 | window of size 2 around centre word "into", with P(w_{t+j} \| w_t) arcs | C09 | [1.12](materials/slides/cs224n-2026-lecture02-wordvecs/p12.png) |
| 1.13 | the window moved to centre word "banking" | C09 | [1.13](materials/slides/cs224n-2026-lecture02-wordvecs/p13.png) |
| 1.17 | P(o \| c) annotated: ① dot product, ② exponentiation, ③ normalization; softmax | C12, C13 | [1.17](materials/slides/cs224n-2026-lecture02-wordvecs/p17.png) |
| 1.18 | θ as one long vector in ℝ^{2dV}; contour plot of z = x² + 2y² with a descent path | C11, C15 | [1.18](materials/slides/cs224n-2026-lecture02-wordvecs/p18.png) |
| 1.20 | cost curve: from a random initial value, learning steps down to the minimum θ̂; "Our objectives may not be convex like this" | C15 | [1.20](materials/slides/cs224n-2026-lecture02-wordvecs/p20.png) |
| 1.21 | gradient-descent update equations (matrix and single parameter) and the algorithm's code | C15 | [1.21](materials/slides/cs224n-2026-lecture02-wordvecs/p21.png) |
| 1.22 | SGD: the problem, the solution, the algorithm's code; "Mini Batch Gradient Descent" | C16 | [1.22](materials/slides/cs224n-2026-lecture02-wordvecs/p22.png) |
| 1.23 | matrices U (outside) and V (center); U·v₄ᵀ dot products and softmax(U·v₄ᵀ) probabilities; "Bag of words" model | C17 | [1.23](materials/slides/cs224n-2026-lecture02-wordvecs/p23.png) |
| 1.24 | 2-D projection of word2vec vectors: similar words nearby | C08 | [1.24](materials/slides/cs224n-2026-lecture02-wordvecs/p24.png) |
| 1.25 | word2vec family: why two vectors; skip-gram vs CBOW; naïve softmax, hierarchical softmax, negative sampling | C11, C18, C19 | [1.25](materials/slides/cs224n-2026-lecture02-wordvecs/p25.png) |
| 1.26 | the normalization term ("A big sum over words") and negative sampling's main idea | C19, C20 | [1.26](materials/slides/cs224n-2026-lecture02-wordvecs/p26.png) |
| 1.27 | the SGNS objective, "sigmoid rather than softmax", the sigmoid's plot, and U(w)^{3/4}/Z | C20, C21 | [1.27](materials/slides/cs224n-2026-lecture02-wordvecs/p27.png) |
| 1.28 | ∇_θJ_t(θ) ∈ ℝ^{2dV}: mostly zeros, with ∇v_like, ∇u_I, ∇u_learning | C22 | [1.28](materials/slides/cs224n-2026-lecture02-wordvecs/p28.png) |
| 1.29 | updating only certain rows of the \|V\| × d embedding matrices; "Rows not columns in actual DL packages!" | C22 | [1.29](materials/slides/cs224n-2026-lecture02-wordvecs/p29.png) |
| 1.30 | "Why not capture co-occurrence counts directly?": window vs full-document co-occurrence | C23 | [1.30](materials/slides/cs224n-2026-lecture02-wordvecs/p30.png) |
| 1.31 | window-based co-occurrence matrix for "I like deep learning / I like NLP / I enjoy flying" | C23 | [1.31](materials/slides/cs224n-2026-lecture02-wordvecs/p31.png) |
| 1.32 | sparse count vectors vs low-dimensional dense vectors | C23 | [1.32](materials/slides/cs224n-2026-lecture02-wordvecs/p32.png) |
| 1.33 | SVD: X = UΣVᵀ, keeping k singular values | C24 | [1.33](materials/slides/cs224n-2026-lecture02-wordvecs/p33.png) |
| 1.34 | "Hacks to X": scaling the counts | C25 | [1.34](materials/slides/cs224n-2026-lecture02-wordvecs/p34.png) |
| 1.35 | COALS: verbs and their agent nouns, with parallel arrows drive→driver, swim→swimmer, teach→teacher | C26 | [1.35](materials/slides/cs224n-2026-lecture02-wordvecs/p35.png) |
| 1.36 | GloVe's question: capture ratios of co-occurrence probabilities as linear meaning components | C27 | [1.36](materials/slides/cs224n-2026-lecture02-wordvecs/p36.png) |
| 1.37 | GloVe's answer: log-bilinear model, vector differences, the weighted loss and the weighting function f | C27, C28 | [1.37](materials/slides/cs224n-2026-lecture02-wordvecs/p37.png) |
| 1.38 | intrinsic vs extrinsic evaluation | C29 | [1.38](materials/slides/cs224n-2026-lecture02-wordvecs/p38.png) |
| 1.39 | analogy evaluation: a:b :: c:?, the argmax formula, man:woman :: king:? plotted | C30 | [1.39](materials/slides/cs224n-2026-lecture02-wordvecs/p39.png) |
| 1.40 | GloVe visualization: parallel gender offsets (man–woman, king–queen, uncle–aunt, …) | C30 | [1.40](materials/slides/cs224n-2026-lecture02-wordvecs/p40.png) |
| 1.41 | WordSim353: word pairs with mean human similarity ratings | C32 | [1.41](materials/slides/cs224n-2026-lecture02-wordvecs/p41.png) |
| 1.42 | correlation with human judgments on WS353, MC, RG, SCWS, RW (GloVe paper Table 3) | C32 | [1.42](materials/slides/cs224n-2026-lecture02-wordvecs/p42.png) |
| 1.43 | NER F1 with 50d vectors on Dev, Test, ACE, MUC7 (GloVe paper Table 4) | C33 | [1.43](materials/slides/cs224n-2026-lecture02-wordvecs/p43.png) |
| 3.2 | notes 2 Fig. 2: analogy accuracy vs training time (GloVe iterations; skip-gram negative samples) | C31 | [3.2](materials/figures/notes2-p08.png) |
| 3.3 | notes 2 Fig. 3: semantic, syntactic and overall accuracy by training corpus (1B to 42B tokens) | C31 | [3.3](materials/figures/notes2-p08.png) |
| 3.4 | notes 2 Fig. 4: GloVe accuracy vs vector dimension and window size, symmetric and asymmetric context | C31 | [3.4](materials/figures/notes2-p09.png) |
| 3.5 | notes 2 Fig. 5: 2-D word vectors classified by a linear decision boundary | C35 | [3.5](materials/figures/notes2-p10.png) |
| 3.6 | notes 2 Fig. 6: "Telly", "TV" and "Television" classified correctly before retraining | C36 | [3.6](materials/figures/notes2-p11.png) |
| 3.7 | notes 2 Fig. 7: after retraining on "Telly" and "TV" only, "Television" is misclassified | C36 | [3.7](materials/figures/notes2-p11.png) |
| 3.8 | notes 2 Fig. 8: "museums in Paris are amazing", a symmetric window of length 2 around "Paris" | C39 | [3.8](materials/figures/notes2-p12.png) |
| 3.9 | notes 2 Fig. 9: the best linear boundary misclassifies many points | C40 | [3.9](materials/figures/notes2-p13.png) |
| 3.10 | notes 2 Fig. 10: a non-linear boundary classifies them | C40 | [3.10](materials/figures/notes2-p13.png) |

<!-- Row shape: | 3.1 | what the figure shows | C01, C02 | [3.1](<image url>) |
     For a picture extracted from a prepared EPUB (overlay section 1), the link is its absolute file URI: | 4.1 | ... | C05 | [4.1](file:///.../prepared/books/BK-<hash>.assets/OPS/images/<file>) | -->

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

### C01 Denotational semantics (signifier ⟺ signified)
- **Meaning:** The "Commonest linguistic way of thinking of meaning": a symbol paired with the idea or thing it stands for, "signifier (symbol) ⟺ signified (idea or thing) = denotational semantics" (slide 4).
- **Key points:**
  - The slide opens with Webster's senses of "meaning", e.g. "the idea that is represented by a word, phrase, etc." (slide 4).
  - "The word Zuko is a sign, a symbol that represents an entity Zuko in some (real of imagined) world"; what a word refers to can be a specific instance or "a broader class" (notes 1 §2.1).
  - "Word meaning is endlessly complex, deriving from humans' goals of communicating with each other and achieving goals in the world" (notes 1 §2.1).
- **Example:** tree ⟺ {pictures of three trees, …} (slide 4). "Zuko makes the tea for his uncle": with coffee or drink swapped in, which sentence is "more like" the one about tea? (notes 1 §2.1).
- **Extends:** ch01:C43, the lecture's working definition of the denotational side, in signifier/signified terms.

### C02 WordNet (a hand-built thesaurus as word meaning)
- **Meaning:** The "Previously commonest NLP solution": "a thesaurus containing lists of synonym sets and hypernyms ('is a' relationships)" (slide 5).
- **Key points:**
  - Problems (slide 6): "A useful resource but missing nuance" ("proficient" is listed as a synonym for "good", "only correct in some contexts"); offensive synonyms "without any coverage of the connotations or appropriateness of words"; "Missing new meanings of words" (wicked, badass, nifty, wizard, genius, ninja, bombest); "Impossible to keep up-to-date!"; "Subjective"; "Requires human labor to create and adapt"; "Can't be used to accurately compute word similarity".
  - Relying on WordNet's synonym lists for similarity "is well-known to fail badly: incompleteness, etc." (slide 8).
- **Example:** NLTK code printing the synonym sets containing "good" (noun: good, goodness; adj (sat): estimable, good, honorable, respectable; …) and the hypernym closure of panda (procyonid → carnivore → placental → mammal → … → entity) (slide 5).

### C03 Word vectors from annotated properties
- **Meaning:** A word vector whose dimensions are human-annotated features, such as plural noun, 3rd singular verb, hyponym-of-beverage or synonym-of-chai, built from resources like WordNet [Miller 1995] and UniMorph [Batsuren et al. 2022] (notes 1 §2.3, Eq. 3).
- **Key points:**
  - "In 2023, word vectors resulting from these methods are not the norm" (notes 1 §2.3).
  - Failures (notes 1 §2.3): "human-annotated resources are always lacking in vocabulary" and costly to update; "it takes a very high-dimensional vector (think much larger than the vocabulary size) to represent all of these categories"; and "human ideas of what the right representations should be for text tend to underperform methods that allow data to determine more aspects—at least when one has a lot of data to learn from".
- **Example:** runners: plurality, derivation (to run + "doer"), and hyponymy (a runner is a human) (notes 1 §2.3).

### C04 One-hot (localist) word vectors
- **Meaning:** "In traditional NLP, we regard words as discrete symbols … a localist representation"; each word is a vector with "one 1, the rest 0s", of dimension "number of words in vocabulary (e.g., 500,000+)" (slide 7). Notes 1 §2.2 calls them standard basis vectors (v_tea = e₃).
- **Key points:**
  - motel and hotel "are orthogonal"; "There is no natural notion of similarity for one-hot vectors!" (slide 8).
  - v_teaᵀv_coffee = v_teaᵀv_the = 0: "all words are equally dissimilar from each other", and there is no character-level information beyond identity (notes 1 §2.2).
  - The way out: "learn to encode similarity in the vectors themselves" (slide 8).
- **Example:** a web search for "Seattle motel" should match documents containing "Seattle hotel" (slide 8, citing IR-book §9.2.2).

### C05 Word type vs word token
- **Meaning:** "A (word) type is an element of a vocabulary; a word in abstract. A (word) token is an instance of a type in context" (notes 1 §2.2).
- **Key points:**
  - These representations give "a single representation for each word type", used for every token of it (notes 1 §2.2).

### C06 Distributional semantics (a word's context window)
- **Meaning:** "A word's meaning is given by the words that frequently appear close-by" (slide 9). The distributional hypothesis: "the meaning of a word can be derived from the distribution of contexts in which it appears" (notes 1 §3).
- **Key points:**
  - "You shall know a word by the company it keeps" (J. R. Firth 1957: 11; slide 9), from Firth's *Synopsis*, p. 11. Notes 1 cites Firth's "Applications of General Linguistics" (1957) for the line, but that paper's text does not contain it.
  - "One of the most successful ideas of modern statistical NLP!" (slide 9).
  - "When a word w appears in a text, its context is the set of words that appear nearby (within a fixed-size window)"; "We use the many contexts of w to build up a representation of w" (slide 9).
  - Notes 1 §3 frames it as self-supervised learning: "taking part of the data (maybe a word in a sentence) and attempting to predict other parts of the data (other words) with it".
- **Example:** "…government debt problems turning into banking crises as happened in 2009…" and two more contexts: "These context words will represent banking" (slide 9). Tea shows up around "drank, the, pot, kettle, bag, delicious, oolong…", so coffee, with similar neighbours, gets a similar distribution (notes 1 §3).
- **Extends:** ch01:C32, adding the fixed-size context window and the distributional hypothesis stated as such.

### C07 Dense word vectors (word embeddings; a distributed representation)
- **Meaning:** "a dense vector for each word, chosen so that it is similar to vectors of words that appear in similar contexts, measuring similarity as the vector dot (scalar) product" (slide 10).
- **Key points:**
  - "word vectors are also called (word) embeddings or (neural) word representations"; "They are a distributed representation" (slide 10), the opposite of C04's localist one.
  - Notes 1's summary: "representing words as low-dimensional real-valued vectors learned from distributional signal".
- **Example:** banking = (0.286, 0.792, −0.177, −0.107, 0.109, −0.542, 0.349, 0.271) and monetary = (0.413, 0.582, −0.007, 0.247, 0.216, −0.718, 0.147, 0.051) (slide 10). The banking numbers are the ones Lecture 1 showed for "versatile" (ch01:C31).
- **Extends:** ch01:C31, adding dense vs localist, similarity by dot product, and the embedding and distributed-representation terms.

### C08 Word2vec (a framework for learning word vectors)
- **Meaning:** "Word2vec is a framework for learning word vectors (Mikolov et al. 2013)" (slide 11).
- **Key points:**
  - The idea (slide 11): "We have a large corpus ('body') of text: a long list of words"; "Every word in a fixed vocabulary is represented by a vector"; "Go through each position t in the text, which has a center word c and context ('outside') words o"; "Use the similarity of the word vectors for c and o to calculate the probability of o given c (or vice versa)"; "Keep adjusting the word vectors to maximize this probability".
  - Each word is "a low-dimensional (much smaller than vocabulary size) vector", learned "to be predictive via a simple function of the distribution of words in a (usually short; 2-4 words) context" (notes 1 §3.2).
  - "Word2vec maximizes objective function by putting similar words nearby in space" (slide 24). With slide 14's notation, training minimizes the negative log-likelihood J, equivalently maximizes the likelihood L; "maximizes objective function" must not be read as maximizing J.
- **Example:** a 2-D projection of learned vectors in which neighbours cluster: days of the week, number words, scientists' names, and science words (physics, chemistry, biology, mathematics) (slide 24).
- **Figures:** 1.11, 1.24.

### C09 Skip-gram (predict the context window from the centre word)
- **Meaning:** "For each position t = 1, …, T, predict context words within a window of fixed size m, given center word w_t" (slide 14); "The model we'll describe here is called the skipgram word2vec algorithm" (notes 1 §3.2).
- **Symbols:** C, O = random variables for a centre word and an outside word, with values c, o (notes 1 §3.2); m = window size.
- **Example:** in "… problems turning into banking crises as …", the centre word "into" at position t, with P(w_{t−2} \| w_t), P(w_{t−1} \| w_t), P(w_{t+1} \| w_t), P(w_{t+2} \| w_t) over a window of size 2 (slide 12); the window then moves to the centre word "banking" (slide 13).
- **Figures:** 1.11, 1.12, 1.13.

### C10 The word2vec objective (average negative log-likelihood)
- **Meaning:** Data likelihood L(θ) = ∏_{t=1}^{T} ∏_{−m≤j≤m, j≠0} P(w_{t+j} \| w_t; θ), and objective J(θ) = −(1/T) log L(θ) = −(1/T) Σ_{t=1}^{T} Σ_{−m≤j≤m, j≠0} log P(w_{t+j} \| w_t; θ) (slide 14).
- **Symbols:** "θ is all variables to be optimized"; J is "sometimes called a cost or loss function" (slide 14).
- **Key points:**
  - "Minimizing objective function ⟺ Maximizing predictive accuracy" (slide 14).
  - As cross-entropy with the true distribution P*(O \| C): min_{U,V} E_{o,c}[−log p_{U,V}(o \| c)] (notes 1 §3.2, Eq. 5).
  - Over a corpus (notes 1 §3.3), the sum runs "over (1) all documents of the sum over (2) all words in the document of the sum over (3) all words occuring in the window": L(U,V) = Σ_{d∈D} Σ_i Σ_{−k≤j≤k, j≠0} −log p_{U,V}(w^{(d)}_{i+j} \| w^{(d)}_i), over valid positions, with k the window size (notes 1 uses m for document length and k for the window; the slides use m for the window). Source defect: notes 1's Eq. 6 prints only the left-context terms (j = 1…k with w_{i−j}), although the window defined just above it is two-sided.

### C11 Two vectors per word (centre v_w, context u_w)
- **Meaning:** "We will use two vectors per word w: v_w when w is a center word; u_w when w is a context word" (slide 15).
- **Symbols:** U, V ∈ ℝ^{\|V\|×d}; u_w and v_w are word w's rows (notes 1 §3.2).
- **Key points:**
  - θ stacks every word's two vectors: θ = [v_aardvark, v_a, …, v_zebra, u_aardvark, u_a, …, u_zebra] ∈ ℝ^{2dV}. "Recall: θ represents all the model parameters, in one long vector"; "with d-dimensional vectors and V-many words"; "Remember: every word has two vectors" (slide 18).
  - "Why two vectors? → Easier optimization. Average both at the end"; "But can implement the algorithm with just one vector per word … and it helps a bit" (slide 25).
- **Example:** P(u_problems \| v_into) is "short for P(problems \| into; u_problems, v_into, θ)" (slide 16).
- **Figures:** 1.18.

### C12 The word2vec prediction function
- **Meaning:** For a centre word c and a context word o, P(o \| c) = exp(u_oᵀv_c) / Σ_{w∈V} exp(u_wᵀv_c) (slides 15, 17).
- **Key points:**
  - Slide 17's annotations: "① Dot product compares similarity of o and c … Larger dot product = larger probability"; "② Exponentiation makes anything positive"; "③ Normalize over entire vocabulary to give probability distribution".
  - The vector p_{U,V}(·\|c) "is a lot like a row of our old co-occurrence matrix X_c" (notes 1 §3.2). Notes 1 also asks why not every distribution over o given c can be represented by this model, and leaves it open ("Should this be good? Bad? Surprising? Obvious?").
- **Figures:** 1.17.

### C13 The softmax function
- **Meaning:** softmax: ℝⁿ → (0,1)ⁿ, softmax(x_i) = exp(x_i) / Σ_{j=1}^{n} exp(x_j) = p_i; it "maps arbitrary values x_i to a probability distribution p_i" (slide 17).
- **Key points (slide 17):**
  - "'max' because amplifies probability of largest x_i"; "'soft' because still assigns some probability to smaller x_i"; "But sort of a weird name because it returns a distribution!"
  - "Frequently used in Deep Learning". (0,1)ⁿ is labelled "Open region".
- **Figures:** 1.17.

### C14 The skip-gram gradient: observed minus expected
- **Meaning:** For one centre word c and outside word o, ∇_{v_c} log p(o \| c) = u_o − Σ_{x∈V} p(x \| c) u_x = u_o − E[u_w]: "'observed' - 'expected'" (notes 1 §3.4). The gradient of the per-pair loss −log p(o \| c) is its negative, Σ_x p(x \| c) u_x − u_o.
- **Key points:**
  - The derivation (notes 1 §3.4) splits the log into Part A, ∇_{v_c} log exp(u_oᵀv_c) = ∇_{v_c} u_oᵀv_c = u_o (shown dimension by dimension), and Part B, the log-sum-exp normalizer, worked by "derivative of log; chain rule", "linearity of derivative", "derivative of exponential; chain rule" and "derivative of dot product". Pulling the normalizer under the sum turns each weight into p(x \| c): "This is an expectation".
  - "So, the v_c vector is updated to be 'more like' the word vector that was actually observed than the word vector it expected" (notes 1 §3.4).
  - The gradient of a sum is the sum of the gradients (Eq. 9); "by convention, we set the gradient of any object to be the shape of that object" (u_o, not u_oᵀ).
  - In the deck: "We compute all vector gradients!" (slide 18). The plan gives "Word2vec objective function gradients" 25 minutes (slide 2), and slide 19 ("Interactive Session!") restates L(θ) and P(o \| c), but the PDF has no worked derivation; notes 1 supplies it.
  - Source defects in notes 1 §3.4: Eq. 10 prints the denominator as Σ u_wᵀv_c without exp (Eq. 11 has it); Part B writes u_w inside a sum over x, where u_x is meant; Part B's left side carries a minus sign its right side drops. The final result above is correct.

### C15 Gradient descent (step size / learning rate)
- **Meaning:** "Gradient Descent is an algorithm to minimize J(θ)"; "Idea: for current value of θ, calculate gradient of J(θ), then take small step in direction of negative gradient. Repeat." (slide 20).
- **Symbols:** θ^{new} = θ^{old} − α∇_θJ(θ); per parameter θ_j^{new} = θ_j^{old} − α ∂J(θ)/∂θ_j^{old}; "α = step size or learning rate" (slide 21).
- **Key points:**
  - The algorithm: `while True: theta_grad = evaluate_gradient(J,corpus,theta); theta = theta - alpha * theta_grad` (slide 21).
  - "To train a model, we gradually adjust parameters to minimize a loss"; "We optimize these parameters by walking down the gradient" (slide 18).
  - "Note: Our objectives may not be convex like this ☹ But life turns out to be okay ☺" (slide 20).
  - The gradient ∇_U f "represents the direction to (locally) move U in in order to maximally increase the value of f"; U and V start as small random draws, "U, V ∼ N(0, 0.001)^{\|V\|×d}", then U^{(i+1)} = U^{(i)} − α∇_U L(U^{(i)}, V^{(i)}) (notes 1 §3.3, Eq. 7).
- **Figures:** 1.18, 1.20, 1.21.

### C16 Stochastic gradient descent (SGD; mini-batch gradient descent)
- **Meaning:** "Repeatedly sample windows, and update after each one" (slide 22).
- **Key points:**
  - "Problem: J(θ) is a function of all windows in the corpus (potentially billions!)", so ∇_θJ(θ) "is very expensive to compute"; "You would wait a very long time before making a single update!"; "Very bad idea for pretty much all neural nets!" (slide 22).
  - The displayed SGD loop updates after one sampled window: `window = sample_window(corpus); theta_grad = evaluate_gradient(J,window,theta); theta = theta - alpha * theta_grad` (slide 22). Mini-batch gradient descent, named on the same slide, uses several sampled training instances for one update.
  - Notes 1's example: "approximating L(U, V) using a few samples for each step", by sampling documents d₁, …, d_ℓ ∼ D (notes 1 §3.3, Eq. 8).
- **Figures:** 1.22.

### C17 Position-independent prediction (word2vec as a "bag of words" model)
- **Meaning:** Given a centre word, word2vec predicts the same distribution over outside words at every position in the window: "'Bag of words' model! … The model makes the same predictions at each position" (slide 23).
- **Key points:**
  - The computation: matrices U ("outside") and V ("center"); U·v₄ᵀ gives the dot products and softmax(U·v₄ᵀ) the probabilities (slide 23).
  - "We want a model that gives a reasonably high probability estimate to all words that occur in the context (at all often)" (slide 23).
  - Skip-gram predicts context words "(position independent)" (slide 25).
- **Figures:** 1.23.

### C18 Continuous bag of words (CBOW)
- **Meaning:** "Predict center word from (bag of) context words" (slide 25), one of word2vec's "Two model variants".
- **Key points:**
  - Set against skip-gram, which predicts "context ('outside') words (position independent) given center word"; "We presented: Skip-gram model" (slide 25).
  - Notes 1's appendix A.1 "Continuous Bag-of-Words" is an empty heading in this draft.
- **Figures:** 1.25.

### C19 Naïve softmax and the cost of its normalizer (the partition function)
- **Meaning:** Training with the full softmax P(o \| c): "simple but expensive loss function, when many output classes" (slide 25); "The normalization term is computationally expensive (when many output classes)": "A big sum over words" (slide 26).
- **Key points:**
  - The training losses (slide 25): "1. Naïve softmax … 2. More optimized variants like hierarchical softmax 3. Negative sampling"; "So far, we explained naïve softmax". Hierarchical softmax is only named.
  - "it is cheap to compute the unnormalized score", but "expensive to compute the partition function (sum of scores for all words) in the denominator, since it requries a term for each word in the vocabulary" (notes 1 §3.5).
  - The partition function guarantees a probability; "From a learning perspective, the partition function 'pushes down' on all the words other than the observed words" (notes 1 §3.5).
  - Source defect: notes 1 §3.5 writes the unnormalized score as exp(u_cᵀv_o), with the roles swapped; elsewhere it is exp(u_oᵀv_c).
- **Figures:** 1.25, 1.26.

### C20 Skip-gram with negative sampling (SGNS)
- **Meaning:** "train binary logistic regressions to differentiate a true pair (center word and a word in its context window) versus several 'noise' pairs (the center word paired with a random word)"; "standard word2vec implements the skip-gram model with negative sampling" (slide 26).
- **Symbols:** J_neg-sample(u_o, v_c, U) = −log σ(u_oᵀv_c) − Σ_{k∈{K sampled indices}} log σ(−u_kᵀv_c): "sigmoid rather than softmax" (slide 27).
- **Key points:**
  - "We take k negative samples (using word probabilities)"; "Maximize probability that real outside word appears; minimize probability that random words appear around center word" (slide 27).
  - "Sample with P(w)=U(w)^{3/4}/Z, the unigram distribution U(w) raised to the 3/4 power"; "The power makes less frequent words be sampled more often" (slide 27).
  - Notes 1 §3.5 writes the objective as log σ(u_oᵀv_c) + Σ_{ℓ=1}^{k} log σ(−u_ℓᵀv_c), the negative of slide 27's J (to be maximized), and leaves p_neg undefined ("Think of this for now like the uniform distribution over V"). Its intuition: "if we randomly push down a few words at each step, then on average, things will work out sort of as if we always pushed down every word".
  - Slide 27 links Mikolov et al. 2013, the negative-sampling paper.
- **Figures:** 1.26, 1.27.

### C21 The logistic (sigmoid) function
- **Meaning:** σ(x) = 1/(1 + e^{−x}), an S-curve from 0 to 1 (slide 27).
- **Key points:**
  - Used in SGNS "rather than softmax"; "(we'll become good friends soon ☺)" (slide 27).
- **Figures:** 1.27.

### C22 Sparse gradient updates (an aside)
- **Meaning:** With SGD on windows, "we only have at most 2m + 1 words plus 2km negative words with negative sampling, so ∇_θJ_t(θ) is very sparse!" (slide 28).
- **Scope:** The sparse-update claim is for skip-gram with negative sampling: an update touches the centre vector, the observed outside vectors and the sampled negative outside vectors. With the naïve full softmax, the vocabulary-wide normalizer generally gives nonzero gradients for every outside vector, even for one sampled window.
- **Key points:**
  - The example gradient in ℝ^{2dV} is mostly 0, with entries ∇v_like, ∇u_I, ∇u_learning (slide 28).
  - "We might only update the word vectors that actually appear!"; "Solution: either you need sparse matrix update operations to only update certain rows of full embedding matrices U and V, or you need to keep around a hash for word vectors"; "Rows not columns in actual DL packages!" (slide 29).
  - "If you have millions of word vectors and do distributed computing, it is important to not have to send gigantic updates around!" (slide 29).
- **Figures:** 1.28, 1.29.

### C23 The co-occurrence matrix X (window vs document)
- **Meaning:** A word-by-context count matrix: "why don't we just accumulate all the statistics of what words appear near each other?!?", instead of "iterating through the whole corpus (perhaps many times)" (slide 30).
- **Key points:**
  - "2 options: windows vs. full document". A window "captures some syntactic and semantic information ('word space')"; a word-document matrix "will give general topics (all sports terms will have similar entries) leading to 'Latent Semantic Analysis' ('document space')" (slide 30).
  - Notes 1 §3.1 builds a document-level matrix in four steps (vocabulary; a \|V\|×\|V\| matrix of zeros; add each document's co-occurrence counts; "Normalize the rows by the sum"). "shorter windows … seem to encode syntactic properties"; larger windows "more semantic (and at extremes, topic-like) properties".
  - Count vectors "increase in size with vocabulary", are "Very high dimensional: require a lot of storage (though sparse)", and give "Subsequent classification models … sparsity issues → Models are less robust". The aim: store "'most' of the important information in a fixed, small number of dimensions: a dense vector", "Usually 25–1000 dimensions, similar to word2vec" (slide 32).
  - Representing explicit counts in \|V\|-sized vectors "ends up being a big mistake", and raw counts over-emphasize very common words like "the" (notes 1 §3.1).
  - Notation (notes 2 §1.2): X_ij = "the number of times word j occur in the context of word i"; X_i = Σ_k X_ik; P_ij = P(w_j \| w_i) = X_ij/X_i. "Populating this matrix requires a single pass through the entire corpus … it is a one-time up-front cost."
- **Example:** window length 1 ("more common: 5–10"), symmetric ("irrelevant whether left or right context"), over "I like deep learning / I like NLP / I enjoy flying" (slide 31). The 8×8 table includes a "." token, e.g. I–like = 2, like–deep = 1, learning–"." = 1; every cell checks against the three sentences with sentence-final periods.
- **Figures:** 1.30, 1.31, 1.32.
- **Extends:** ch01:C45, adding window vs document contexts, LSA, window-size effects and the case for low dimensions.

### C24 Dimensionality reduction by SVD of X
- **Meaning:** "Factorizes X into UΣVᵀ, where U and V are orthonormal (unit vectors and orthogonal)"; "Retain only k singular values, in order to generalize" (slide 33, "Classic Method: Dimensionality Reduction on X (HW1)").
- **Key points:**
  - "X̂ is the best rank k approximation to X, in terms of least squares. Classic linear algebra result. Expensive to compute for large matrices." (slide 33).
  - Notation: these U and V are SVD factors, not word2vec's outside and centre matrices.
- **Figures:** 1.33.
- **Extends:** ch01:C46, adding the best-rank-k least-squares property and "in order to generalize".

### C25 Scaling the counts (fixes for raw X)
- **Meaning:** "Running an SVD on raw counts doesn't work well!!!"; "Scaling the counts in the cells can help a lot" (slide 34).
- **Key points:**
  - "Problem: function words (the, he, has) are too frequent → syntax has too much impact" (slide 34).
  - The fixes (slide 34): "log the frequencies"; "min(X, t), with t ≈ 100"; "Ignore the function words"; "Ramped windows that count closer words more than further away words"; "Use correlations instead of counts, then set negative values to 0".
  - "Taking the log token frequency ends up being much more useful"; GloVe "taught us much more about what is wrong with the raw co-occurrence method" (notes 1 §3.1).
- **Figures:** 1.34.

### C26 Linear semantic patterns in scaled count vectors (COALS)
- **Meaning:** "Interesting semantic patterns emerge in the scaled vectors" (slide 35), shown with the COALS model (Rohde et al. ms., 2005).
- **Example:** verbs (drive, swim, teach, clean, learn, marry, treat, pray) and their agent or associated nouns (driver, swimmer, teacher, janitor, student, bride, doctor, priest) in a 2-D plot, with parallel arrows drive→driver, swim→swimmer and teach→teacher (slide 35). In the COALS manuscript the plot is a multidimensional-scaling projection (Figure 13, pp. 23–24), not an SVD.
- **Figures:** 1.35.

### C27 Ratios of co-occurrence probabilities (meaning components)
- **Meaning:** Ratios of co-occurrence probabilities pick out the meaning components that distinguish two words. Slide 36 asks: "How can we capture ratios of co-occurrence probabilities as linear meaning components in a word vector space?", under the title "Encoding meaning components in vector differences".
- **Example:** (GloVe paper Table 1, not on the slide) P(k \| ice)/P(k \| steam) is 8.9 for k = solid, 8.5×10⁻² for gas, 1.36 for water and 0.96 for fashion. "Only in the ratio does noise from non-discriminative words like water and fashion cancel out" (GloVe paper §3).
- **Figures:** 1.36, 1.37.

### C28 GloVe (Global Vectors)
- **Meaning:** "A: Log-bilinear model: w_i · w_j = log P(i \| j)", "with vector differences w_x · (w_a − w_b) = log [P(x \| a)/P(x \| b)]" (slide 37; GloVe [Pennington, Socher, and Manning, EMNLP 2014]).
- **Symbols:** Loss J = Σ_{i,j=1}^{V} f(X_ij)(w_iᵀw̃_j + b_i + b̃_j − log X_ij)² (slide 37). w_i is a target-word vector and w̃_j a separate context-word vector; b_i and b̃_j are their learned scalar biases; X_ij is the co-occurrence count and f(X_ij) its loss weight (GloVe paper §3, Eqs. 1, 7–9). f is plotted rising to 1 and then flat (slide 37); in the paper f(x) = (x/x_max)^α for x < x_max and 1 otherwise, with x_max = 100 and α = 3/4 (Eq. 9).
- **Key points:**
  - "Fast training"; "Scalable to huge corpora" (slide 37).
  - Count-based matrix-factorization methods (LSA, HAL) "effectively leverage global statistical information" but "do poorly on tasks such as word analogy"; shallow window-based methods (skip-gram, CBOW) "capture complex linguistic patterns beyond word similarity, but fail to make use of the global co-occurrence statistics"; GloVe "consists of a weighted least squares model that trains on global word-word co-occurrence counts" (notes 2 §1.1).
  - Notes 2 §1.3 derives the loss: skip-gram's implied global cross-entropy is J = −Σ_i Σ_j X_ij log Q_ij; normalizing Q is expensive, so it is dropped for a least-squares loss on the unnormalized P̂_ij = X_ij and Q̂_ij = exp(u_jᵀv_i); large X_ij makes that hard to optimize, so take logs, Σ X_i (u_jᵀv_i − log X_ij)²; and replace X_i with a general weighting f(X_ij).
  - It trains "only on the nonzero elements" and "consistently outperforms word2vec on the word analogy task, given the same corpus, vocabulary, window size, and training time" (notes 2 §1.4).
- **Figures:** 1.37.
- **Extends:** ch01:C47, which named GloVe; here its model and loss are taught.

### C29 Intrinsic vs extrinsic evaluation
- **Meaning:** Intrinsic: "Evaluation on a specific/intermediate subtask"; "Fast to compute"; "Helps to understand that system"; "Not clear if it's helpful unless correlation to real task is established". Extrinsic: "Evaluation on a real task"; "Can take a long time to compute accuracy"; "Unclear if the subsystem is the problem or its interaction or other subsystems"; "If replacing exactly one subsystem with another improves accuracy → Winning!" (slide 38).
- **Key points:**
  - Notes 2 §2.1 motivates intrinsic evaluation with a question-answering system whose word-vector subsystem has hyperparameters to tune: retraining the whole system after each change "is impractical", so "a requirement is that the intrinsic evaluation has a positive correlation with the final task performance". Notes 2 Figure 1 shows the expensive subsystem replaced by a simpler one for intrinsic evaluation.
- **Figures:** 1.38.

### C30 Word-vector analogy evaluation
- **Meaning:** For a:b :: c:?, d = argmax_i (x_b − x_a + x_c)ᵀx_i / ‖x_b − x_a + x_c‖ (slide 39; notes 2 §2.3).
- **Key points:**
  - "Evaluate word vectors by how well their cosine distance after addition captures intuitive semantic and syntactic analogy questions"; "Discarding the input words from the search (!)"; "Problem: What if the information is there but not linear?" (slide 39).
  - The formula is a cosine only if the x_i are unit length, since its denominator omits ‖x_i‖ (our note; the slide and notes call it cosine).
  - "we want x_b − x_a = x_d − x_c (For instance, queen – king = actress – actor)" (notes 2 §2.3). Caveats: shared city names ("there are at least 10 places in the US called Phoenix"); dated corpora ("prior to 1997 the capital of Kazakhstan was Almaty"). Syntactic analogies test superlatives (bad : worst :: big : biggest) and past tense (dancing : danced :: flying : flew).
  - Notes 2's side note: "nokia" can be close to "samsung" in one dimension and to "finland" in another, and a 2-D PCA view loses such structure.
- **Example:** man:woman :: king:?, with the parallel offsets plotted (slide 39). GloVe's visualization shows parallel gender offsets: man–woman, king–queen, uncle–aunt, brother–sister, nephew–niece, sir–madam, heir–heiress, earl–countess, duke–duchess, emperor–empress (slide 40).
- **Figures:** 1.39, 1.40.
- **Extends:** ch01:C50, adding the argmax formula, the intrinsic-evaluation framing and the caveats.

### C31 Hyperparameters and analogy accuracy
- **Meaning:** An intrinsic task such as analogy completion is used to tune embedding hyperparameters: "Dimension of word vectors", "Corpus size", "Corpus souce/type", "Context window size", "Context symmetry" (notes 2 §2.4).
- **Key points:**
  - From Table 5: "Performance is heavily dependent on the model used for word embedding"; "Performance increases with larger corpus sizes"; "Performance is lower for extremely low dimensional word vectors" (notes 2 §2.4). The notes illustrate insufficient capacity with a two-bit encoding of four words (king, queen, man, woman) using "gender" and "leadership"; this is not a claim that the words or their analogies need at least two real-valued dimensions.
  - Very high dimensions might be expected to overfit, but "Yin et al. show in On the Dimensionality of Word Embedding that skip-gram and GloVe are robust to this over-fitting" (notes 2 §2.4).
  - "A window size of 8 around each center word typically works well for GloVe embeddings" (notes 2 §2.4).
  - The plots: accuracy rises with training time (Fig. 2); it varies by corpus size and source (Fig. 3); in Fig. 4 gains from more dimensions diminish, larger windows help semantic analogies more, while syntactic accuracy peaks with smaller windows. These observations belong to those experiments.
- **Example:** Table 5's GloVe 300d on 42B tokens scores 75.0 total; SVD 300d on 6B scores 7.3 (notes 2 §2.4).
- **Figures:** 3.2, 3.3, 3.4.

### C32 Word-similarity correlation (WordSim353)
- **Meaning:** "Word vector distances and their correlation with human judgments" (slides 41–42); humans rate a pair's similarity "on a fixed scale (say 0-10)", compared with the vectors' cosine similarity (notes 2 §2.5).
- **Example:** WordSim353 means (slide 41): tiger–cat 7.35, tiger–tiger 10, book–paper 7.46, computer–internet 7.58, plane–car 5.77, professor–doctor 6.62, stock–phone 1.62, stock–CD 1.31, stock–jaguar 0.92. Slide 42's table (GloVe paper Table 3, "Spearman rank correlation"): GloVe 42B scores 75.9 / 83.6 / 82.9 / 59.6 / 47.8 on WS353 / MC / RG / SCWS / RW, the best of the listed models on all five; CBOW* 100B scores 68.4 / 79.6 / 75.4 / 59.4 / 45.5.
- **Figures:** 1.41, 1.42.

### C33 Named entity recognition as an extrinsic evaluation
- **Meaning:** "One example where good word vectors should help directly: named entity recognition: identifying references to a person, organization or location: Chris Manning lives in Palo Alto." (slide 43).
- **Example:** Slide 43's table (GloVe paper Table 4, "F1 score on NER task with 50d vectors"; Dev / Test / ACE / MUC7): GloVe 93.2 / 88.3 / 82.9 / 82.2; HPCA has the best Test score, 88.7; Discrete 91.0 / 85.4 / 77.4 / 73.4.
- **Figures:** 1.43.

### C34 Word senses and multiple prototypes (Huang et al. 2012)
- **Meaning:** One word can need different vectors for its different uses: "'run' is both a noun and a verb" (notes 2 §2.6).
- **Key points:**
  - The method (notes 2 §2.6): gather fixed-size context windows of all its occurrences ("for instance, 5 before and 5 after"); represent each context "by a weighted average of the context words' vectors (using idf-weighting)"; "Apply spherical k-means to cluster these context representations"; finally re-label each occurrence to its cluster and train one representation per cluster.
- **Extends:** ch01:C49, adding a method for separating senses.

### C35 Extrinsic tasks as classification
- **Meaning:** "Most NLP extrinsic tasks can be formulated as classification tasks" (notes 2 §3.1).
- **Symbols:** training set {x^(i), y^(i)}_1^N, where x^(i) is "a d-dimensional word vector generated by some word embedding technique" and y^(i) "a C-dimensional one-hot vector which indicates the labels we wish to eventually predict" (notes 2 §3.1).
- **Example:** sentiment (positive, negative or neutral); NER, where "Jim bought 300 shares of Acme Corp. in 2006" becomes "[Jim]Person bought 300 shares of [Acme Corp.]Organization in [2006]Time" (notes 2 §3.1). Word vectors split by a linear boundary "using techniques such as logistic regression and SVMs" (Fig. 5).
- **Figures:** 3.5.

### C36 Retraining word vectors on the extrinsic task
- **Meaning:** "In NLP applications however, we introduce the idea of retraining the input word vectors when we train for extrinsic tasks"; pretrained vectors "are initialized by optimizing them over a simpler intrinsic task" (notes 2 §3.1–3.2).
- **Key points:**
  - "retraining word vectors can be risky": with a small training set, the retrained words "are shifted in the word space and as a result, the performance over the final task could actually reduce" (notes 2 §3.2).
  - "word vectors should not be retrained if the training data set is small. If the training set is large, retraining may improve performance"; "For small datasets, retraining word vectors will likely worsen performance" (notes 2 §3.2).
- **Example:** "Telly", "TV" and "Television" are classified correctly before retraining; after retraining on a set that contains only "Telly" and "TV", the boundary moves and "Television" is misclassified (Figs. 6–7).
- **Figures:** 3.6, 3.7.

### C37 Softmax classification with cross-entropy loss
- **Meaning:** p(y_j = 1 \| x) = exp(W_j·x) / Σ_{c=1}^{C} exp(W_c·x), "the probability of word vector x being in class j", trained with the cross-entropy loss −Σ_j y_j log p(y_j = 1 \| x) (notes 2 §3.3).
- **Key points:**
  - With a one-hot y the loss is −log [exp(W_k·x) / Σ_c exp(W_c·x)] for the correct class k; over N examples, −Σ_i log [exp(W_{k(i)}·x^(i)) / Σ_c exp(W_c·x^(i))] (notes 2 §3.3).
  - Training the weights and the word vectors together updates "as many as C·d + \|V\|·d" parameters "for a simple linear classifier", which "is highly prone to overfitting" (notes 2 §3.3).

### C38 Regularization (L2 penalty)
- **Meaning:** "a regularization term which poses the Bayesian belief that the parameters (θ) should be small is magnitude (i.e. close to zero)": the loss plus λ Σ_{k=1}^{C·d+\|V\|·d} θ_k² (notes 2 §3.3).
- **Key points:**
  - It "reduces the likelihood of the parameters taking on extremely large values just to fit the training set well and may improve generalization if the relative objective weight λ is tuned well" (notes 2 §3.3).
  - It "becomes even more of a requirement once we explore more complex models (such as Neural Networks)" (notes 2 §3.3).

### C39 Window classification
- **Meaning:** Classify a word from a window of word vectors: x_window = [x^(i−2); x^(i−1); x^(i); x^(i+1); x^(i+2)] in place of the single x^(i) (notes 2 §3.4).
- **Key points:**
  - "Natural languages tend to use the same word for very different meanings": "to sanction" can mean "to permit" or "to punish" (notes 2 §3.4).
  - For this five-word window, x_window ∈ ℝ^{5d}, so the softmax weight matrix has shape C × 5d instead of C × d for a single-word input.
  - The gradient δ_window = [∇x^(i−2); …; ∇x^(i+2)] "will of course need to be distributed to update the corresponding word vectors in implementation" (notes 2 §3.4).
  - "Generally, narrower window sizes lead to better performance in syntactic tests while wider windows lead to better performance in semantic tests" (notes 2 §3.4).
- **Example:** "museums in Paris are amazing": a symmetric window of length 2 "may help disambiguate between the place Paris and the name Paris" (Fig. 8).
- **Figures:** 3.8.

### C40 Non-linear classifiers (why neural networks)
- **Meaning:** A linear classifier "misclassifies many datapoints"; "Using a non-linear decision boundary … we manage to classify all training points accurately" (notes 2 §3.5).
- **Key points:**
  - "Although oversimplified, this is a classic case demonstrating the need for non-linear decision boundaries" (notes 2 §3.5).
- **Figures:** 3.9, 3.10.

## Problems

What each end-of-chapter problem asks and which concepts it exercises. Record no solutions the source doesn't give.

| Problem | Asks | Exercises |
|---|---|---|

No problems: Lecture 2 releases no assignment, and its notes set no exercises. Assignment 1 (word vectors) was released with Lecture 1 and is mapped in ch01. Notes 1 poses open questions in passing (§3.2: why not every distribution over o given c can be represented; "Should this be good? Bad?"), recorded under C12.

## Forward references

Topics the chapter defers. None is taught here; later extraction EXTENDs from these hooks.

| Topic | Chapter | Claim / promised treatment |
|---|---|---|
| Neural networks as non-linear classifiers | Lecture 3 (ch03) | "In the next set of notes, we study neural networks as a class of non-linear models that have performed particularly well in deep learning applications" (notes 2 §3.5) |

## Studs

Concepts the chapter uses but does not teach. A stud means "not taught here", not "nothing more to learn".

<!-- Entry shape (numbering runs B01, B02, …):
 ### B01 <Label>
- **Meaning:** one line.
- **Used at:** §3.1 "short quote"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.
  (Book mode, overlay §8: when an earlier chapter's map teaches it, write `Taught in ch02:C05` instead.)
-->

### B01 Vectors, dot product and orthogonality
- **Meaning:** u · v = Σᵢ uᵢvᵢ; two vectors are orthogonal when their dot product is 0.
- **Used at:** slide 8 "these two vectors are orthogonal"; slide 10 "measuring similarity as the vector dot (scalar) product"; notes 1 §2.3, Eq. 3.
- **Status:** not taught here (ch01:B18 was also a stud). Slide 17 restates u·v = Σᵢ uᵢvᵢ as a tool. Further prerequisites are not assessed in this chapter.

### B02 Probability and conditional probability
- **Meaning:** P(o \| c), the probability of o given c.
- **Used at:** slide 11 "calculate the probability of o given c"; slides 14, 17.
- **Status:** not taught here (ch01:B10 was also a stud). Further prerequisites are not assessed in this chapter.

### B03 Likelihood, logarithms and exponentials
- **Meaning:** the likelihood of data under a model; log and exp and their rules.
- **Used at:** slide 14 "Data likelihood", "negative log likelihood"; slide 17 "Exponentiation makes anything positive"; notes 1 Eq. 5.
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B04 Derivatives, gradients and the chain rule
- **Meaning:** differentiation rules for log, exp and dot products; the gradient of a scalar with respect to a vector.
- **Used at:** notes 1 §3.4 "derivative of log; chain rule", "linearity of derivative".
- **Status:** not taught here (ch01:B05, gradient-based learning, was also a stud). Further prerequisites are not assessed in this chapter.

### B05 Expectation of a random variable
- **Meaning:** E[X] = Σ_x p(x) x.
- **Used at:** notes 1 §3.2 Eq. 5 E_{o,c}[…]; §3.4 "This is an expectation".
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B06 Convex functions
- **Meaning:** a function with a single bowl shape, whose local minimum is global.
- **Used at:** slide 20 "Our objectives may not be convex like this".
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B07 Binary logistic regression
- **Meaning:** a classifier that gives the probability of one of two classes through the sigmoid of a score.
- **Used at:** slide 26 "train binary logistic regressions".
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B08 Unigram (word-frequency) distribution
- **Meaning:** the probability of each word by its frequency in the corpus.
- **Used at:** slide 27 "the unigram distribution U(w)".
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B09 Singular value decomposition, orthonormal matrices and rank
- **Meaning:** X = UΣVᵀ with orthonormal U, V and singular values on Σ's diagonal; the rank of a matrix.
- **Used at:** slide 33 "Factorizes X into UΣVᵀ, where U and V are orthonormal"; "best rank k approximation".
- **Status:** not taught here (ch01:B17 was also a stud). Further prerequisites are not assessed in this chapter.

### B10 Weighted least-squares regression
- **Meaning:** fitting by minimizing a weighted sum of squared errors.
- **Used at:** notes 2 §1.3 "a weighted least squares model"; slide 37's squared loss.
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B11 Cosine similarity
- **Meaning:** the cosine of the angle between two vectors.
- **Used at:** slide 39 "cosine distance after addition"; notes 2 §2.3, §2.5.
- **Status:** Taught in ch01:C48.

### B12 Correlation (Spearman rank correlation)
- **Meaning:** how closely two rankings agree.
- **Used at:** slide 42 "correlation with human judgments"; GloVe paper Table 3 "Spearman rank correlation".
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B13 Clustering (k-means) and idf weighting
- **Meaning:** grouping vectors into k clusters; weighting words by inverse document frequency.
- **Used at:** notes 2 §2.6 "spherical k-means", "idf-weighting".
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B14 Overfitting, bias and variance
- **Meaning:** too little model capacity (high bias) or fitting noise (high variance).
- **Used at:** notes 2 §2.4 "a high bias problem", "high variance".
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B15 Classifiers and decision boundaries
- **Meaning:** a model assigning inputs to one of C classes; the boundary between the regions of each class; labels as one-hot vectors.
- **Used at:** notes 2 §3.1 "logistic regression and SVMs"; optimizers other than gradient descent are only named ("L-BFGS, Newton's method").
- **Status:** not taught here (gradient descent is taught as C15). Further prerequisites are not assessed in this chapter.

### B16 F1 score
- **Meaning:** the harmonic mean of precision P and recall R: F1 = 2PR/(P + R).
- **Used at:** slide 43's table (GloVe paper Table 4, "F1 score on NER task with 50d vectors").
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

## Prerequisites

Every row reads "From `prerequisite_for` To"; a learner walks backward from To. The prerequisite rows must form
no cycle. A **scope** means the prerequisite is needed only for that part of To.

| From | Predicate | To | Family | Status | Scope | Basis |
|---|---|---|---|---|---|---|
| B01 Vectors, dot product and orthogonality | `prerequisite_for` | C04 One-hot (localist) word vectors | prerequisite | stated | orthogonality | slide 8 "these two vectors are orthogonal" |
| B01 Vectors, dot product and orthogonality | `prerequisite_for` | C07 Dense word vectors (word embeddings; a distributed representation) | prerequisite | stated | dot-product similarity | slide 10 "measuring similarity as the vector dot (scalar) product" |
| B01 Vectors, dot product and orthogonality | `prerequisite_for` | C03 Word vectors from annotated properties | prerequisite | inferred | vectors and their dimensions | notes 1 §2.3, Eq. 3 "one could build word vectors that look something like" |
| C02 WordNet (a hand-built thesaurus as word meaning) | `prerequisite_for` | C03 Word vectors from annotated properties | prerequisite | stated | its WordNet-based semantic dimensions | notes 1 §2.3 "With such resources" |
| C05 Word type vs word token | `prerequisite_for` | C04 One-hot (localist) word vectors | prerequisite | inferred | one vector per type | notes 1 §2.2 "a single representation for each word type" |
| C06 Distributional semantics (a word's context window) | `prerequisite_for` | C07 Dense word vectors (word embeddings; a distributed representation) | prerequisite | stated | — | slide 10 "similar to vectors of words that appear in similar contexts" |
| C04 One-hot (localist) word vectors | `prerequisite_for` | C07 Dense word vectors (word embeddings; a distributed representation) | prerequisite | inferred | the dense vs localist contrast | slides 7, 10 "localist" vs "distributed representation" |
| C06 Distributional semantics (a word's context window) | `prerequisite_for` | C08 Word2vec (a framework for learning word vectors) | prerequisite | inferred | — | slide 11: context words o predicted from centre word c |
| C07 Dense word vectors (word embeddings; a distributed representation) | `prerequisite_for` | C08 Word2vec (a framework for learning word vectors) | prerequisite | stated | — | slide 11 "a framework for learning word vectors" |
| B02 Probability and conditional probability | `prerequisite_for` | C08 Word2vec (a framework for learning word vectors) | prerequisite | stated | — | slide 11 "calculate the probability of o given c" |
| C08 Word2vec (a framework for learning word vectors) | `prerequisite_for` | C09 Skip-gram (predict the context window from the centre word) | prerequisite | stated | — | notes 1 §3.2 "the skipgram word2vec algorithm" |
| C08 Word2vec (a framework for learning word vectors) | `prerequisite_for` | C11 Two vectors per word (centre v_w, context u_w) | prerequisite | inferred | — | slide 11's vectors and centre/context roles, given separate vectors on slide 15 |
| C09 Skip-gram (predict the context window from the centre word) | `prerequisite_for` | C10 The word2vec objective (average negative log-likelihood) | prerequisite | stated | — | slide 14 "predict context words within a window of fixed size m" |
| B02 Probability and conditional probability | `prerequisite_for` | C10 The word2vec objective (average negative log-likelihood) | prerequisite | stated | — | slide 14 P(w_{t+j} \| w_t; θ) |
| B03 Likelihood, logarithms and exponentials | `prerequisite_for` | C10 The word2vec objective (average negative log-likelihood) | prerequisite | stated | — | slide 14 "negative log likelihood" |
| C12 The word2vec prediction function | `prerequisite_for` | C10 The word2vec objective (average negative log-likelihood) | prerequisite | inferred | — | slide 15: "How to calculate P(w_{t+j} \| w_t; θ)?" is answered by P(o \| c) |
| C11 Two vectors per word (centre v_w, context u_w) | `prerequisite_for` | C12 The word2vec prediction function | prerequisite | stated | — | slide 15 "Then for a center word c and a context word o" |
| C13 The softmax function | `prerequisite_for` | C12 The word2vec prediction function | prerequisite | stated | the normalization | slide 17 "This is an example of the softmax function" |
| B01 Vectors, dot product and orthogonality | `prerequisite_for` | C12 The word2vec prediction function | prerequisite | stated | — | slide 17 "Dot product compares similarity of o and c" |
| B02 Probability and conditional probability | `prerequisite_for` | C13 The softmax function | prerequisite | stated | — | slide 17 "to a probability distribution p_i" |
| B03 Likelihood, logarithms and exponentials | `prerequisite_for` | C13 The softmax function | prerequisite | stated | exponentiation | slide 17 "Exponentiation makes anything positive" |
| C12 The word2vec prediction function | `prerequisite_for` | C14 The skip-gram gradient: observed minus expected | prerequisite | stated | — | notes 1 §3.4 differentiates log p_{U,V}(o \| c) |
| C10 The word2vec objective (average negative log-likelihood) | `prerequisite_for` | C14 The skip-gram gradient: observed minus expected | prerequisite | stated | the loss's sign and aggregation; the worked expression differentiates log probability | notes 1 §3.4 "the partial gradient of the loss"; Eq. 9 |
| C11 Two vectors per word (centre v_w, context u_w) | `prerequisite_for` | C14 The skip-gram gradient: observed minus expected | prerequisite | stated | — | notes 1 §3.4 u_o, v_c |
| B04 Derivatives, gradients and the chain rule | `prerequisite_for` | C14 The skip-gram gradient: observed minus expected | prerequisite | stated | — | notes 1 §3.4 "derivative of log; chain rule" |
| B05 Expectation of a random variable | `prerequisite_for` | C14 The skip-gram gradient: observed minus expected | prerequisite | stated | the "expected" reading | notes 1 §3.4 "This is an expectation" |
| B05 Expectation of a random variable | `prerequisite_for` | C10 The word2vec objective (average negative log-likelihood) | prerequisite | stated | Eq. 5's expectation over centre/outside pairs | notes 1 §3.2 Eq. 5 E_{o,c} |
| C10 The word2vec objective (average negative log-likelihood) | `prerequisite_for` | C15 Gradient descent (step size / learning rate) | prerequisite | stated | — | slide 20 "a cost function J(θ) we want to minimize" |
| B04 Derivatives, gradients and the chain rule | `prerequisite_for` | C15 Gradient descent (step size / learning rate) | prerequisite | stated | — | slide 20 "calculate gradient of J(θ)" |
| C14 The skip-gram gradient: observed minus expected | `prerequisite_for` | C15 Gradient descent (step size / learning rate) | prerequisite | inferred | computing word2vec's ∇J | slide 18 "We compute all vector gradients!" |
| B06 Convex functions | `prerequisite_for` | C15 Gradient descent (step size / learning rate) | prerequisite | stated | the non-convexity note | slide 20 "may not be convex" |
| C15 Gradient descent (step size / learning rate) | `prerequisite_for` | C16 Stochastic gradient descent (SGD; mini-batch gradient descent) | prerequisite | stated | — | slide 22: the SGD loop is slide 21's with sample_window |
| C09 Skip-gram (predict the context window from the centre word) | `prerequisite_for` | C16 Stochastic gradient descent (SGD; mini-batch gradient descent) | prerequisite | stated | the windows sampled | slide 22 "Repeatedly sample windows" |
| C09 Skip-gram (predict the context window from the centre word) | `prerequisite_for` | C17 Position-independent prediction (word2vec as a "bag of words" model) | prerequisite | stated | — | slide 25 "(position independent)" |
| C11 Two vectors per word (centre v_w, context u_w) | `prerequisite_for` | C17 Position-independent prediction (word2vec as a "bag of words" model) | prerequisite | stated | — | slide 23 U "outside", V "center" |
| C12 The word2vec prediction function | `prerequisite_for` | C17 Position-independent prediction (word2vec as a "bag of words" model) | prerequisite | stated | — | slide 23 U·v₄ᵀ |
| C13 The softmax function | `prerequisite_for` | C17 Position-independent prediction (word2vec as a "bag of words" model) | prerequisite | stated | — | slide 23 softmax(U·v₄ᵀ) |
| C08 Word2vec (a framework for learning word vectors) | `prerequisite_for` | C18 Continuous bag of words (CBOW) | prerequisite | stated | — | slide 25 "Two model variants" |
| C09 Skip-gram (predict the context window from the centre word) | `prerequisite_for` | C18 Continuous bag of words (CBOW) | prerequisite | stated | the contrast | slide 25 |
| C12 The word2vec prediction function | `prerequisite_for` | C19 Naïve softmax and the cost of its normalizer (the partition function) | prerequisite | stated | — | slide 26 P(o \| c) and "The normalization term" |
| C13 The softmax function | `prerequisite_for` | C19 Naïve softmax and the cost of its normalizer (the partition function) | prerequisite | stated | — | slide 25 "Naïve softmax" |
| C19 Naïve softmax and the cost of its normalizer (the partition function) | `prerequisite_for` | C20 Skip-gram with negative sampling (SGNS) | prerequisite | stated | the motivation | slide 26 "Hence, standard word2vec implements…" |
| C09 Skip-gram (predict the context window from the centre word) | `prerequisite_for` | C20 Skip-gram with negative sampling (SGNS) | prerequisite | stated | — | slide 26 "the skip-gram model with negative sampling" |
| C11 Two vectors per word (centre v_w, context u_w) | `prerequisite_for` | C20 Skip-gram with negative sampling (SGNS) | prerequisite | stated | — | slide 27 u_o, v_c, u_k |
| C21 The logistic (sigmoid) function | `prerequisite_for` | C20 Skip-gram with negative sampling (SGNS) | prerequisite | stated | — | slide 27 "sigmoid rather than softmax" |
| B07 Binary logistic regression | `prerequisite_for` | C20 Skip-gram with negative sampling (SGNS) | prerequisite | stated | — | slide 26 "train binary logistic regressions" |
| B08 Unigram (word-frequency) distribution | `prerequisite_for` | C20 Skip-gram with negative sampling (SGNS) | prerequisite | stated | the sampling distribution | slide 27 "the unigram distribution U(w)" |
| B03 Likelihood, logarithms and exponentials | `prerequisite_for` | C21 The logistic (sigmoid) function | prerequisite | stated | the exponential | slide 27 σ(x) = 1/(1 + e^{−x}) |
| C16 Stochastic gradient descent (SGD; mini-batch gradient descent) | `prerequisite_for` | C22 Sparse gradient updates (an aside) | prerequisite | stated | — | slide 28 "gradients at each window for SGD" |
| C20 Skip-gram with negative sampling (SGNS) | `prerequisite_for` | C22 Sparse gradient updates (an aside) | prerequisite | stated | the sparse-update claim as a whole | slide 28 "with negative sampling" |
| C11 Two vectors per word (centre v_w, context u_w) | `prerequisite_for` | C22 Sparse gradient updates (an aside) | prerequisite | stated | rows of U and V | slide 29 "rows of full embedding matrices U and V" |
| B04 Derivatives, gradients and the chain rule | `prerequisite_for` | C22 Sparse gradient updates (an aside) | prerequisite | stated | — | slide 28 ∇_θJ_t(θ) |
| C06 Distributional semantics (a word's context window) | `prerequisite_for` | C23 The co-occurrence matrix X (window vs document) | prerequisite | stated | — | slide 30 "what words appear near each other" |
| B01 Vectors, dot product and orthogonality | `prerequisite_for` | C23 The co-occurrence matrix X (window vs document) | prerequisite | inferred | — | rows of X are vectors (slide 32 "Co-occurrence vectors") |
| C23 The co-occurrence matrix X (window vs document) | `prerequisite_for` | C24 Dimensionality reduction by SVD of X | prerequisite | stated | — | slide 33 "Singular Value Decomposition of co-occurrence matrix X" |
| B09 Singular value decomposition, orthonormal matrices and rank | `prerequisite_for` | C24 Dimensionality reduction by SVD of X | prerequisite | stated | — | slide 33 |
| C23 The co-occurrence matrix X (window vs document) | `prerequisite_for` | C25 Scaling the counts (fixes for raw X) | prerequisite | stated | — | slide 34 "Hacks to X" |
| C24 Dimensionality reduction by SVD of X | `prerequisite_for` | C25 Scaling the counts (fixes for raw X) | prerequisite | stated | "Running an SVD on raw counts" | slide 34 |
| B03 Likelihood, logarithms and exponentials | `prerequisite_for` | C25 Scaling the counts (fixes for raw X) | prerequisite | stated | "log the frequencies" | slide 34 |
| C25 Scaling the counts (fixes for raw X) | `prerequisite_for` | C26 Linear semantic patterns in scaled count vectors (COALS) | prerequisite | stated | — | slide 35 "in the scaled vectors" |
| C23 The co-occurrence matrix X (window vs document) | `prerequisite_for` | C27 Ratios of co-occurrence probabilities (meaning components) | prerequisite | stated | P_ij from counts | notes 2 §1.2 P_ij = X_ij/X_i |
| B02 Probability and conditional probability | `prerequisite_for` | C27 Ratios of co-occurrence probabilities (meaning components) | prerequisite | stated | — | slide 36 "co-occurrence probabilities" |
| C27 Ratios of co-occurrence probabilities (meaning components) | `prerequisite_for` | C28 GloVe (Global Vectors) | prerequisite | stated | — | slides 36–37 Q and A |
| C23 The co-occurrence matrix X (window vs document) | `prerequisite_for` | C28 GloVe (Global Vectors) | prerequisite | stated | — | notes 2 §1.1 "trains on global word-word co-occurrence counts" |
| C07 Dense word vectors (word embeddings; a distributed representation) | `prerequisite_for` | C28 GloVe (Global Vectors) | prerequisite | stated | — | slide 36 "in a word vector space" |
| B03 Likelihood, logarithms and exponentials | `prerequisite_for` | C28 GloVe (Global Vectors) | prerequisite | stated | the logs | slide 37 log X_ij |
| B10 Weighted least-squares regression | `prerequisite_for` | C28 GloVe (Global Vectors) | prerequisite | stated | — | notes 2 §1.1 "a weighted least squares model" |
| C10 The word2vec objective (average negative log-likelihood) | `prerequisite_for` | C28 GloVe (Global Vectors) | prerequisite | stated | notes 2's derivation from skip-gram's cross-entropy | notes 2 §1.3 |
| C12 The word2vec prediction function | `prerequisite_for` | C28 GloVe (Global Vectors) | prerequisite | stated | notes 2's derivation (the softmax Q_ij) | notes 2 §1.3 |
| C07 Dense word vectors (word embeddings; a distributed representation) | `prerequisite_for` | C29 Intrinsic vs extrinsic evaluation | prerequisite | stated | — | slide 38 "How to evaluate word vectors?" |
| C29 Intrinsic vs extrinsic evaluation | `prerequisite_for` | C30 Word-vector analogy evaluation | prerequisite | stated | — | slide 39 "Intrinsic word vector evaluation" |
| C07 Dense word vectors (word embeddings; a distributed representation) | `prerequisite_for` | C30 Word-vector analogy evaluation | prerequisite | stated | — | slide 39 "Evaluate word vectors" |
| B11 Cosine similarity | `prerequisite_for` | C30 Word-vector analogy evaluation | prerequisite | stated | — | slide 39 "cosine distance after addition" |
| B01 Vectors, dot product and orthogonality | `prerequisite_for` | C30 Word-vector analogy evaluation | prerequisite | stated | vector addition and the dot product | slide 39 (x_b − x_a + x_c)ᵀx_i |
| C30 Word-vector analogy evaluation | `prerequisite_for` | C31 Hyperparameters and analogy accuracy | prerequisite | stated | — | notes 2 §2.4 "tuned using an intrinsic evaluation system (such as an analogy completion system)" |
| C07 Dense word vectors (word embeddings; a distributed representation) | `prerequisite_for` | C31 Hyperparameters and analogy accuracy | prerequisite | stated | — | notes 2 §2.4 "Dimension of word vectors" |
| B14 Overfitting, bias and variance | `prerequisite_for` | C31 Hyperparameters and analogy accuracy | prerequisite | stated | the dimension discussion | notes 2 §2.4 "high bias problem"; "high variance" |
| C29 Intrinsic vs extrinsic evaluation | `prerequisite_for` | C32 Word-similarity correlation (WordSim353) | prerequisite | stated | — | slide 41 "Another intrinsic word vector evaluation" |
| B11 Cosine similarity | `prerequisite_for` | C32 Word-similarity correlation (WordSim353) | prerequisite | stated | — | notes 2 §2.5 "cosine similarity" |
| B12 Correlation (Spearman rank correlation) | `prerequisite_for` | C32 Word-similarity correlation (WordSim353) | prerequisite | stated | — | slide 42 "correlation with human judgments" |
| C29 Intrinsic vs extrinsic evaluation | `prerequisite_for` | C33 Named entity recognition as an extrinsic evaluation | prerequisite | stated | — | slide 43 "Extrinsic word vector evaluation" |
| C07 Dense word vectors (word embeddings; a distributed representation) | `prerequisite_for` | C33 Named entity recognition as an extrinsic evaluation | prerequisite | stated | — | slide 43 "good word vectors should help directly" |
| C05 Word type vs word token | `prerequisite_for` | C34 Word senses and multiple prototypes (Huang et al. 2012) | prerequisite | inferred | one vector per type | notes 2 §2.6 "the same word with different vectors for its different uses" |
| C06 Distributional semantics (a word's context window) | `prerequisite_for` | C34 Word senses and multiple prototypes (Huang et al. 2012) | prerequisite | stated | — | notes 2 §2.6 "Gather fixed size context windows" |
| C07 Dense word vectors (word embeddings; a distributed representation) | `prerequisite_for` | C34 Word senses and multiple prototypes (Huang et al. 2012) | prerequisite | stated | — | notes 2 §2.6 "the context words' vectors" |
| B13 Clustering (k-means) and idf weighting | `prerequisite_for` | C34 Word senses and multiple prototypes (Huang et al. 2012) | prerequisite | stated | — | notes 2 §2.6 |
| C07 Dense word vectors (word embeddings; a distributed representation) | `prerequisite_for` | C35 Extrinsic tasks as classification | prerequisite | stated | — | notes 2 §3.1 "x^(i) is a d-dimensional word vector" |
| C33 Named entity recognition as an extrinsic evaluation | `prerequisite_for` | C35 Extrinsic tasks as classification | prerequisite | stated | the NER example | notes 2 §3.1 |
| B15 Classifiers and decision boundaries | `prerequisite_for` | C35 Extrinsic tasks as classification | prerequisite | stated | — | notes 2 §3.1 |
| C35 Extrinsic tasks as classification | `prerequisite_for` | C36 Retraining word vectors on the extrinsic task | prerequisite | stated | — | notes 2 §3.2 |
| C07 Dense word vectors (word embeddings; a distributed representation) | `prerequisite_for` | C36 Retraining word vectors on the extrinsic task | prerequisite | stated | — | notes 2 §3.2 "pretrained word vectors" |
| C15 Gradient descent (step size / learning rate) | `prerequisite_for` | C36 Retraining word vectors on the extrinsic task | prerequisite | inferred | training | notes 2 §3.1 "train weights using optimization techniques (such as gradient descent…)" |
| C35 Extrinsic tasks as classification | `prerequisite_for` | C37 Softmax classification with cross-entropy loss | prerequisite | stated | — | notes 2 §3.3 |
| C13 The softmax function | `prerequisite_for` | C37 Softmax classification with cross-entropy loss | prerequisite | stated | — | notes 2 §3.3 "the Softmax classification function" |
| B03 Likelihood, logarithms and exponentials | `prerequisite_for` | C37 Softmax classification with cross-entropy loss | prerequisite | stated | the log | notes 2 §3.3 |
| C10 The word2vec objective (average negative log-likelihood) | `prerequisite_for` | C37 Softmax classification with cross-entropy loss | prerequisite | inferred | cross-entropy, met in notes 1 Eq. 5 | notes 2 §3.3 "the Cross-entropy loss function" |
| C37 Softmax classification with cross-entropy loss | `prerequisite_for` | C38 Regularization (L2 penalty) | prerequisite | stated | — | notes 2 §3.3 |
| B14 Overfitting, bias and variance | `prerequisite_for` | C38 Regularization (L2 penalty) | prerequisite | stated | — | notes 2 §3.3 "To reduce overfitting risk" |
| C37 Softmax classification with cross-entropy loss | `prerequisite_for` | C39 Window classification | prerequisite | stated | — | notes 2 §3.4 "modify the previously discussed Softmax model" |
| C06 Distributional semantics (a word's context window) | `prerequisite_for` | C39 Window classification | prerequisite | stated | context windows | notes 2 §3.4 |
| B04 Derivatives, gradients and the chain rule | `prerequisite_for` | C39 Window classification | prerequisite | stated | distributing the window gradient to its word vectors | notes 2 §3.4 δ_window |
| C37 Softmax classification with cross-entropy loss | `prerequisite_for` | C40 Non-linear classifiers (why neural networks) | prerequisite | inferred | — | notes 2 Figs. 5, 9: the linear boundary |
| B15 Classifiers and decision boundaries | `prerequisite_for` | C40 Non-linear classifiers (why neural networks) | prerequisite | stated | — | notes 2 §3.5 "decision boundary" |
| B16 F1 score | `prerequisite_for` | C33 Named entity recognition as an extrinsic evaluation | prerequisite | stated | interpreting the table's metric | GloVe paper Table 4 caption "F1 score on NER task with 50d vectors" |

<!-- Row shape: | B01 Label | `prerequisite_for` | C01 Label | prerequisite | inferred | — | §3.1 "short quote" | -->

### Per-concept review

| Concept | Direct prerequisites and notes |
|---|---|
| C01 Denotational semantics | none in this lecture (Lecture 1 context: ch01:C43) |
| C02 WordNet | none |
| C03 Word vectors from annotated properties | B01 (vectors and their dimensions); C02 (its WordNet-based dimensions) |
| C04 One-hot (localist) word vectors | B01 (orthogonality); C05 (one vector per type) |
| C05 Word type vs word token | none |
| C06 Distributional semantics | none in this lecture (Lecture 1: ch01:C32) |
| C07 Dense word vectors | C06; C04 (the dense vs localist contrast); B01 (dot-product similarity) |
| C08 Word2vec | C06; C07; B02 |
| C09 Skip-gram | C08 |
| C10 The word2vec objective | C09; C12; B02; B03; B05 (Eq. 5's expectation) |
| C11 Two vectors per word | C08 |
| C12 The word2vec prediction function | C11; C13 (normalization); B01 |
| C13 The softmax function | B02; B03 (exponentiation) |
| C14 The skip-gram gradient | C12; C10 (sign and aggregation); C11; B04; B05 (the "expected" reading) |
| C15 Gradient descent | C10; B04; C14 (computing word2vec's ∇J); B06 (non-convexity note) |
| C16 Stochastic gradient descent | C15; C09 (windows sampled) |
| C17 Position-independent prediction | C09; C11; C12; C13 |
| C18 Continuous bag of words (CBOW) | C08; C09 (the contrast) |
| C19 Naïve softmax and the cost of its normalizer | C12; C13 |
| C20 Skip-gram with negative sampling | C19 (motivation); C09; C11; C21; B07; B08 (sampling distribution) |
| C21 The logistic (sigmoid) function | B03 (exponential) |
| C22 Sparse gradient updates | C16; C20 (the claim as a whole); C11 (rows of U and V); B04 |
| C23 The co-occurrence matrix X | C06; B01 |
| C24 Dimensionality reduction by SVD of X | C23; B09 |
| C25 Scaling the counts | C23; C24 (SVD on raw counts); B03 (log) |
| C26 Linear semantic patterns (COALS) | C25 |
| C27 Ratios of co-occurrence probabilities | C23 (P_ij from counts); B02 |
| C28 GloVe | C27; C23; C07; B03 (logs); B10; C10 and C12 (notes 2's derivation) |
| C29 Intrinsic vs extrinsic evaluation | C07 |
| C30 Word-vector analogy evaluation | C29; C07; B11; B01 (vector addition, dot product) |
| C31 Hyperparameters and analogy accuracy | C30; C07; B14 (dimension discussion) |
| C32 Word-similarity correlation | C29; B11; B12 |
| C33 Named entity recognition as an extrinsic evaluation | C29; C07; B16 (interpreting the F1 table) |
| C34 Word senses and multiple prototypes | C05 (one vector per type); C06; C07; B13 |
| C35 Extrinsic tasks as classification | C07; C33 (NER example); B15 |
| C36 Retraining word vectors | C35; C07; C15 (training) |
| C37 Softmax classification with cross-entropy loss | C35; C13; B03 (log); C10 (cross-entropy) |
| C38 Regularization (L2 penalty) | C37; B14 |
| C39 Window classification | C37; C06 (context windows); B04 (distributing the gradient) |
| C40 Non-linear classifiers | C37; B15 |

## Predicates

The controlled list is `predicates.md` in this pack. List here only predicates this map ADDED with the
Reviewer's agreement, each with its family and meaning.

## Links

Functional links from the text, each marked **stated** (the author says it) or **inferred** (our reading, with its reason).

| From | Predicate | To | Family | Status | Scope / original wording | Basis |
|---|---|---|---|---|---|---|
| C04 One-hot (localist) word vectors | `contrasts_with` | C07 Dense word vectors (word embeddings; a distributed representation) | functional | stated | "a localist representation" vs "a distributed representation" | slides 7, 10 |
| C02 WordNet (a hand-built thesaurus as word meaning) | `contrasts_with` | C06 Distributional semantics (a word's context window) | functional | stated | "Could try to rely on WordNet's list of synonyms … Instead: learn to encode similarity in the vectors themselves" | slides 8–9 |
| C09 Skip-gram (predict the context window from the centre word) | `kind_of` | C08 Word2vec (a framework for learning word vectors) | taxonomic | stated | "Two model variants: 1. Skip-grams (SG)" | slide 25; slide 11 figure "Skip-gram model" |
| C08 Word2vec (a framework for learning word vectors) | `has_parameters` | C11 Two vectors per word (centre v_w, context u_w) | part-whole | stated | "Every word in a fixed vocabulary is represented by a vector"; "two vectors per word" | slides 11, 15 |
| C12 The word2vec prediction function | `uses` | C13 The softmax function | functional | stated | "This is an example of the softmax function" | slide 17 |
| C12 The word2vec prediction function | `uses` | B01 Vectors, dot product and orthogonality | functional | stated | "Dot product compares similarity of o and c" | slide 17 |
| C10 The word2vec objective (average negative log-likelihood) | `aggregates` | C12 The word2vec prediction function | functional | stated | J sums −log P(w_{t+j} \| w_t; θ) over positions and the window | slide 14 |
| C14 The skip-gram gradient: observed minus expected | `applies_to` | C12 The word2vec prediction function | functional | stated | the gradient of log P(o \| c) with respect to v_c; "the partial gradient of the loss with respect to the parameter v_c" | notes 1 §3.4 |
| C15 Gradient descent (step size / learning rate) | `minimizes` | C10 The word2vec objective (average negative log-likelihood) | functional | stated | "Gradient Descent is an algorithm to minimize J(θ)" | slide 20 |
| C15 Gradient descent (step size / learning rate) | `adjusts` | C11 Two vectors per word (centre v_w, context u_w) | functional | stated | θ^{new} = θ^{old} − α∇_θJ(θ); "Optimize value of parameters" | slides 18, 21 |
| C16 Stochastic gradient descent (SGD; mini-batch gradient descent) | `kind_of` | C15 Gradient descent (step size / learning rate) | taxonomic | inferred | gradient descent on sampled windows; slide 22's loop is slide 21's with sample_window | slides 21–22 |
| C15 Gradient descent (step size / learning rate) | `uses` | C14 The skip-gram gradient: observed minus expected | functional | inferred | the update needs ∇J, which for word2vec is C14's gradient | slides 18, 21 |
| C17 Position-independent prediction (word2vec as a "bag of words" model) | `property_of` | C09 Skip-gram (predict the context window from the centre word) | functional | stated | "(position independent)"; "The model makes the same predictions at each position" | slides 23, 25 |
| C18 Continuous bag of words (CBOW) | `kind_of` | C08 Word2vec (a framework for learning word vectors) | taxonomic | stated | "Two model variants … 2. Continuous Bag of Words (CBOW)" | slide 25 |
| C18 Continuous bag of words (CBOW) | `contrasts_with` | C09 Skip-gram (predict the context window from the centre word) | functional | stated | predict centre from context vs context from centre | slide 25 |
| C19 Naïve softmax and the cost of its normalizer (the partition function) | `property_of` | C12 The word2vec prediction function | functional | stated | "The normalization term is computationally expensive" | slide 26 |
| C20 Skip-gram with negative sampling (SGNS) | `applies_to` | C09 Skip-gram (predict the context window from the centre word) | functional | stated | "the skip-gram model with negative sampling" | slides 26–27 |
| C20 Skip-gram with negative sampling (SGNS) | `uses` | C21 The logistic (sigmoid) function | functional | stated | "sigmoid rather than softmax" | slide 27 |
| C20 Skip-gram with negative sampling (SGNS) | `contrasts_with` | C19 Naïve softmax and the cost of its normalizer (the partition function) | functional | stated | "Hence, standard word2vec implements the skip-gram model with negative sampling" | slides 25–26 |
| C22 Sparse gradient updates (an aside) | `applies_to` | C16 Stochastic gradient descent (SGD; mini-batch gradient descent) | functional | stated | "We iteratively take gradients at each window for SGD" | slide 28 |
| C24 Dimensionality reduction by SVD of X | `applies_to` | C23 The co-occurrence matrix X (window vs document) | functional | stated | "Singular Value Decomposition of co-occurrence matrix X" | slide 33 |
| C25 Scaling the counts (fixes for raw X) | `adjusts` | C23 The co-occurrence matrix X (window vs document) | functional | stated | "Scaling the counts in the cells" | slide 34 |
| C26 Linear semantic patterns in scaled count vectors (COALS) | `illustrates` | C25 Scaling the counts (fixes for raw X) | functional | stated | "Interesting semantic patterns emerge in the scaled vectors" | slide 35 |
| C23 The co-occurrence matrix X (window vs document) | `contrasts_with` | C08 Word2vec (a framework for learning word vectors) | functional | stated | "Why not capture co-occurrence counts directly?" vs "iterating through the whole corpus" | slide 30 |
| C24 Dimensionality reduction by SVD of X | `produces` | C07 Dense word vectors (word embeddings; a distributed representation) | functional | stated | "a dense vector", "Usually 25–1000 dimensions", by SVD | slides 32–33 |
| C28 GloVe (Global Vectors) | `uses` | C23 The co-occurrence matrix X (window vs document) | functional | stated | "trains on global word-word co-occurrence counts" | notes 2 §1.1 |
| C28 GloVe (Global Vectors) | `uses` | C27 Ratios of co-occurrence probabilities (meaning components) | functional | stated | "Encoding meaning components in vector differences" | slides 36–37 |
| C28 GloVe (Global Vectors) | `contrasts_with` | C09 Skip-gram (predict the context window from the centre word) | functional | stated | "shallow window-based (e.g. the skip-gram and the CBOW models) … fail to make use of the global co-occurrence statistics" | notes 2 §1.1 |
| C30 Word-vector analogy evaluation | `kind_of` | C29 Intrinsic vs extrinsic evaluation | taxonomic | stated | intrinsic: "Intrinsic word vector evaluation: Word Vector Analogies" | slide 39 |
| C32 Word-similarity correlation (WordSim353) | `kind_of` | C29 Intrinsic vs extrinsic evaluation | taxonomic | stated | intrinsic: "Another intrinsic word vector evaluation" | slide 41 |
| C33 Named entity recognition as an extrinsic evaluation | `kind_of` | C29 Intrinsic vs extrinsic evaluation | taxonomic | stated | extrinsic: "Extrinsic word vector evaluation" | slide 43 |
| C30 Word-vector analogy evaluation | `assesses` | C07 Dense word vectors (word embeddings; a distributed representation) | functional | stated | "Evaluate word vectors by how well…" | slide 39 |
| C32 Word-similarity correlation (WordSim353) | `assesses` | C07 Dense word vectors (word embeddings; a distributed representation) | functional | stated | "Word vector distances and their correlation with human judgments" | slide 41 |
| C33 Named entity recognition as an extrinsic evaluation | `assesses` | C07 Dense word vectors (word embeddings; a distributed representation) | functional | stated | "good word vectors should help directly" | slide 43 |
| C31 Hyperparameters and analogy accuracy | `uses` | C30 Word-vector analogy evaluation | functional | stated | "tuned using an intrinsic evaluation system (such as an analogy completion system)" | notes 2 §2.4 |
| C33 Named entity recognition as an extrinsic evaluation | `illustrates` | C35 Extrinsic tasks as classification | functional | stated | NER: "given a context and a central word, we want to classify the central word" | notes 2 §3.1 |
| C37 Softmax classification with cross-entropy loss | `method_for` | C35 Extrinsic tasks as classification | functional | stated | "Let us consider using the Softmax classification function" | notes 2 §3.3 |
| C37 Softmax classification with cross-entropy loss | `uses` | C13 The softmax function | functional | stated | "the Softmax classification function" | notes 2 §3.3 |
| C38 Regularization (L2 penalty) | `applies_to` | C37 Softmax classification with cross-entropy loss | functional | stated | the λ term is added to that loss | notes 2 §3.3 |
| C36 Retraining word vectors on the extrinsic task | `adjusts` | C07 Dense word vectors (word embeddings; a distributed representation) | functional | stated | "retraining the input word vectors" | notes 2 §3.1 |
| C39 Window classification | `uses` | C07 Dense word vectors (word embeddings; a distributed representation) | functional | stated | x_window concatenates word vectors | notes 2 §3.4 |
| C40 Non-linear classifiers (why neural networks) | `contrasts_with` | C37 Softmax classification with cross-entropy loss | functional | inferred | non-linear vs C37's linear decision boundary | notes 2 Figs. 5, 9–10 |
| C01 Denotational semantics (signifier ⟺ signified) | `contrasts_with` | C06 Distributional semantics (a word's context window) | functional | inferred | two accounts of what supplies word meaning: "Commonest linguistic way of thinking of meaning" vs "A word's meaning is given by the words that frequently appear close-by" (the same distinction as ch01:C43) | slides 4, 9 |

<!-- Row shape: | C02 Label | `has_parameters` | C03 Label | part-whole | stated | — | "short quote" (§3.1) | -->
