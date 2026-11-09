# Concept map: *Stanford CS224N (Winter 2026)*, Chapter 1 (Lecture 1: History of NLP)

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

1. **The course and its goals** (intro 3–8). Three learning goals (intro 5): "The foundations of the effective modern methods for deep learning applied to NLP"; "A big picture understanding of human languages and the difficulties in understanding and producing them via computers"; and the ability to build systems for major language problems.
2. **The lecture's key learning** (intro 2, 9, 19): "understand the paradigm shifts in NLP and how our assumptions about language shape what is possible in each era" (C01). The lecture is split into "The course (10 mins)" and "The history of NLP: How we got here (65 mins)".
3. **Two histories.** First, "a creative way of looking at the history of NLP" (intro 9): an exhibition of painted panels credited to the Stanford NLP group ("check out Gates 3B for the exhibition"), slides 10–18. Then "a scientific way of looking at the history of NLP (60 mins)" (intro 19), given as a guest lecture by Christopher Manning, "the creator of CS224N" (intro 21). Intro 20 shows ChatGPT, Gemini and DeepSeek each recommending Manning when asked which one professor to invite to talk about the history of NLP.

4. **Four eras** (history 2; Dædalus pp. 128–130): Manning divides the history into four eras (C02). He treats NLP/computational linguistics and AI/neural networks as "only somewhat entwined fields" (C03), and the reading opens with why language matters at all (C04).

5. **Era 1, machine translation** (history 3–9, 19, 23): NLP research began with machine translation (C05). Weaver's idea of translation as decoding met Wiener's doubt (C06); the Georgetown–IBM demonstration of 1954 was oversold (C07); the ALPAC report of 1966 was the reality check (C08). The same era produced BASEBALL and ELIZA (C09).

6. **Era 1, AI and neural networks begin** (history 10–16): The Dartmouth summer workshop (1956) marks the start of AI (C10). Neural networks take their inspiration from neurons (C11): McCulloch and Pitts give the first formal neuron (C12), and Rosenblatt's perceptron learns, amid press hype (C13). The decade ends with two visions of AI, symbolic AI and cybernetics (C14).

7. **Era 1, retrieval, evaluation and institutions** (history 17–23): Bush's memex as a vision for information retrieval (C15); Cleverdon's Cranfield tests invent benchmark evaluation (C16); the field forms its society, the ACL, in 1962 (C17). Stanford's early lineage appears in people slides: David Hays (history 20), Raj Reddy (history 21) and Joyce Friedman (history 22).

8. **Era 2, hand-built symbolic NLP** (history 24–31): rules and knowledge written by hand, increasingly formal (C18). Understanding systems such as SHRDLU worked within narrow worlds (C19); grammars became formal and unification-based (C20); meaning was composed up a parse tree (C21); knowledge bases drove inference (C22); and Newell and Simon's hypothesis made symbol systems the theory of intelligence (C23). Then symbolic AI "hit the wall" (history 31).

9. **Era 3, statistical NLP** (history 32–37): digital text and ideas from speech recognition turn NLP empirical (C24). Shannon's noisy channel (C25) stands behind the probabilistic modeling of language (C26), and human-annotated corpora make supervised learning the method of the era (C27). In the 2000s the gains consolidate and the data grows a hundredfold (history 37).

10. **Era 4a, neural NLP** (history 38–47): Manning argues that language is symbolic but its processor need not be (C29). Deep networks break through, first in speech (C30); words become vectors (C31) learned from their contexts (C32); encoder–decoder networks translate (C33) and overtake statistical MT within two years (C34). All of it sits in the era of neural NLP (C28).

11. **Era 4b, large language models** (history 48–57): generative AI becomes a mass phenomenon (C35). The language model, an idea running from Markov through Shannon to IBM (C36), was scaled up in waves held back by compute, model flexibility and data, until all three arrived (C37). Two 2023 studies close the lecture: large gains on some tasks, clear limits on others (C38).

12. **The reading: why pretrained models changed everything** (Dædalus pp. 129–136). Self-supervised pretraining (C39) on transformers (C40) yields one model adaptable by fine-tuning or prompting (C41). It replaces task pipelines across translation, question answering and generation (C42); it learns meaning in the sense of a network of connections (C43); and it is the first of a broader class, foundation models (C44), with risks and limits.

13. **Assignment 1, released with the lecture** (A1, due Tue Jan 13): students build count-based word vectors from a co-occurrence matrix (C45) and reduce them with truncated SVD (C46). They then explore learned GloVe vectors (C47) with cosine similarity (C48), finding where neighbours conflate senses and opposites (C49), where analogies work and fail (C50), and where the vectors encode bias (C51).

## Assumed background

What the chapter uses without teaching it is recorded as **studs** (B01…, below).

**Chapter scope** names what was read, including footnotes or notes wherever the converter placed them.

Source note: the lecture-plan slides intro 9 and 19 are titled "Lecture 1: Introduction and Word Vectors", a leftover from earlier offerings; intro 2 has "Introduction and the History of NLP".

Chapter scope: Lecture 1 of the Winter 2026 syllabus, from the source bundle `work/prepared/ch01/` (`SOURCES.md`): the intro deck (slides 1–21, cited "intro N"), the history deck (slides 1–57, cited "history N"), the suggested reading (Manning 2022, *Dædalus* 151(2): 127–138, with its endnotes, cited "Dædalus p. N"), and Assignment 1 (cited "A1 Qn"). Papers the slides cite are consulted only to check a claim.

## Figures

Concepts cite figures by number. Images are linked, never committed.

Figure numbers: 1.N is intro slide N; 2.N is history slide N; 3.1 is the reading's Figure 1 (Dædalus p. 131). Image links are relative to the published lecture folder, where the slide images sit beside the map.

| Fig | Shows | Teaches | Image |
|---|---|---|---|
| 1.10 | "History of NLP" exhibition: 27 painted panels in roughly historical order, from the Turing test, Shannon's entropy and ELIZA to scaling curves, tool use and "The Thinking LLM". No panel is explained on its slide (intro slides 10–18) | C01 | [1.10](materials/slides/cs224n-2026-lecture01-intro/p10.png) |
| 1.11 | Exhibition panel: n-gram factorizations of P(this, is, a, sentence) for N = 1, 2, 3 | C36 | [1.11](materials/slides/cs224n-2026-lecture01-intro/p11.png) |
| 2.6 | New York Times front page and story, 8 January 1954: "Russian Is Turned Into English By a Fast Electronic Translator", with a punch card and its translation (history slides 6–8) | C07 | [2.6](materials/slides/cs224n-2026-lecture01-history/p06.png) |
| 2.9 | Newsreel title card: "Electronic 'Brain' Translates Russian to English" (Paramount News) | C07 | [2.9](materials/slides/cs224n-2026-lecture01-history/p09.png) |
| 2.11 | Labelled diagram of biological neurons: dendrites, soma, axon, myelin sheath, terminal buttons | C11 | [2.11](materials/slides/cs224n-2026-lecture01-history/p11.png) |
| 2.13 | McCulloch–Pitts neuron: inputs x₁…xₙ, weights w₁ⱼ…wₙⱼ, a summing transfer function giving netⱼ, an activation function φ with threshold θⱼ, output oⱼ; with a step-function graph of 𝟏(Wx > θ) | C12 | [2.13](materials/slides/cs224n-2026-lecture01-history/p13.png) |
| 2.17 | Vannevar Bush with the memex quotation from "As We May Think" (1945) | C15 | [2.17](materials/slides/cs224n-2026-lecture01-history/p17.png) |
| 2.19 | Timeline of the 1950s origins of NLP/CL (from Ruth Camburn's "A Short History of Computational Linguistics"), beside bullet points on the decade | C05, C06 | [2.19](materials/slides/cs224n-2026-lecture01-history/p19.png) |
| 2.23 | Timeline of NLP in the 1960s, with the caption "ALPAC report (1966): crap, this is really hard!" | C05, C08, C09 | [2.23](materials/slides/cs224n-2026-lecture01-history/p23.png) |
| 2.28 | Semantic composition tree for "The red apple is on the table": λ-terms at each word, composed up the parse to on(ι(λx(apple(x) ∧ red(x))), ι(λy.table(y))) | C21 | [2.28](materials/slides/cs224n-2026-lecture01-history/p28.png) |
| 2.30 | Norvig's knowledge-representation network: nodes such as situation, event, action, having, giving, and their roles, linked by labelled arcs | C22 | [2.30](materials/slides/cs224n-2026-lecture01-history/p30.png) |
| 2.34 | Shannon's noisy channel diagram: information source → transmitter → signal → (noise source) → received signal → receiver → destination | C25 | [2.34](materials/slides/cs224n-2026-lecture01-history/p34.png) |
| 2.35 | An HMM-style chain of hidden and observed states; a parse tree for "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo"; Fred Jelinek's photo | C26 | [2.35](materials/slides/cs224n-2026-lecture01-history/p35.png) |
| 2.36 | Probabilistic spelling correction: candidate corrections for "acress" scored by word-sequence probabilities | C26 | [2.36](materials/slides/cs224n-2026-lecture01-history/p36.png) |
| 2.41 | "Deep Learning Breakthroughs by Hinton… at the University of Toronto, 2009–2012": deep acoustic networks beneath an HMM state chain over a spectrogram; a convolutional network diagram; a photo of three researchers; an ImageNet panel with a portrait (none labelled on the slide) | C30 | [2.41](materials/slides/cs224n-2026-lecture01-history/p41.png) |
| 2.42 | Word error rates on RT03S FSH / Hub5 SWB: traditional GMM vs deep learning (2012, 2017) | C30 | [2.42](materials/slides/cs224n-2026-lecture01-history/p42.png) |
| 2.43 | The word "versatile" as an 8-dimensional vector: "Similar vectors = similar meaning" | C31 | [2.43](materials/slides/cs224n-2026-lecture01-history/p43.png) |
| 2.44 | Firth's quotation, with two contexts of "laptops" that "will represent laptops" | C32 | [2.44](materials/slides/cs224n-2026-lecture01-history/p44.png) |
| 2.45 | RNN encoder–decoder translating "I am a student" into "Je suis étudiant", with hₜ = tanh(W[xₜ] + Uhₜ₋₁ + b) | C33 | [2.45](materials/slides/cs224n-2026-lecture01-history/p45.png) |
| 2.46 | Three-layer LSTM encoder–decoder (Sutskever et al. 2014) translating "Die Proteste waren am Wochenende eskaliert": encoder, decoder, "Feeding in last word", "Bottleneck" | C33 | [2.46](materials/slides/cs224n-2026-lecture01-history/p46.png) |
| 2.47 | MT progress 2013–2016, cased BLEU on WMT newstest2013 En–De: phrase-based SMT, syntax-based SMT, neural MT | C34 | [2.47](materials/slides/cs224n-2026-lecture01-history/p47.png) |
| 2.49 | FactSet chart: number of S&P 500 earnings calls citing "AI", 2015–2025, rising after the ChatGPT launch past 40% and then 50% of the S&P 500 | C35 | [2.49](materials/slides/cs224n-2026-lecture01-history/p49.png) |
| 2.51 | "The History of Language Models": Markov 1913, Shannon 1948, Jelinek 1975 | C36 | [2.51](materials/slides/cs224n-2026-lecture01-history/p51.png) |
| 2.52 | "The History of Large Language Models", 1998–2020, with the overlays "Not enough compute!", "Not enough model flexibility!", "Not enough data!", "LLMs go brrr!" (history slides 52–53) | C37 | [2.52](materials/slides/cs224n-2026-lecture01-history/p53.png) |
| 2.54 | ChatGPT 4o's sonnet praising Wikipedia, every line beginning with W | C35 | [2.54](materials/slides/cs224n-2026-lecture01-history/p54.png) |
| 2.55 | Distribution of output quality for BCG consultants with and without GPT-4 | C38 | [2.55](materials/slides/cs224n-2026-lecture01-history/p55.png) |
| 2.56 | The abstract of "Art or Artifice? Large Language Models and the False Promise of Creativity" | C38 | [2.56](materials/slides/cs224n-2026-lecture01-history/p56.png) |
| 3.1 | "Details of the Attention Calculations in One Part of a Transformer Neural Net Model": inputs [CLS] Judiciary [MASK] Annual Report, each word plus position; Q₂ compared with K₀…K₄, values averaged, 12 heads and 12 layers; "committee" predicted at the mask (Dædalus Figure 1, p. 131) | C40 | [3.1](materials/figures/daedalus-fig1-p131.png) |

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

### C01 Paradigm shift in NLP
- **Meaning:** The lecture's organizing idea. NLP's history is a series of eras, each defined by a different assumption about what language is and how a machine should handle it, and that assumption limits what each era's systems can do. A paradigm shift is the move from one such assumption and method to the next.
- **Key points:**
  - The lecture's stated key learning: "understand the paradigm shifts in NLP and how our assumptions about language shape what is possible in each era" (intro 2, 9, 19).
- **Example:** The exhibition (intro 10–18) shows the succession in pictures. The history deck gives it content as four eras (from U2).
- **Scope:** Framed in the intro deck; the eras themselves are entries of their own.

### C02 Four eras of NLP
- **Meaning:** Manning's division of NLP's history into four periods, each with its own dominant method.
- **Key points:**
  - The eras: "1940–1969: Early Explorations"; "1970–1992: Hand-built symbolic NLP systems, of increasing formalization"; "1993–2012: Statistical or Probabilistic NLP and then more general Supervised ML for NLP"; "2013–now: Deep Learning or Artificial Neural Networks; Unsupervised or Self-Supervised NLP; Reinforcement Learning" (history 2).
  - The deck splits era 4 into "4a. Neural NLP" (2013–2021) and "4b. Large Language Models" (2022–present) (history 38, 48).
  - The reading dates era 1 "from 1950 to 1969" (Dædalus p. 128). It adds that self-supervised learning (2018) "may well be viewed as the fundamental change, and the third era might be extended until 2017" (p. 130).
  - The reading's characterizations: era 1 systems "provided little more than word-level translation lookups" (p. 128); era 2 systems "were all hand-built, rule-based" (p. 128); era 3 was "a fundamental reorientation of the field around empirical machine learning models" (p. 129); in era 4, words and sentences "are represented by a position in a (several hundred- or thousand-dimensional) real-valued vector space" (p. 129).
- **Scope:** An overview. Each era's methods are entries of their own (U3–U9).

### C03 NLP/CL and AI/neural networks as partly separate lineages
- **Meaning:** Computational linguistics and AI/neural-network research are separate fields whose histories only sometimes meet. The lecture follows both.
- **Key points:**
  - "Only somewhat entwined fields: NLP / Computational Linguistics; AI / Neural Networks" (history 2).
  - "the field of natural language processing (NLP) emerged in tandem with the earliest developments in artificial intelligence"; machine translation work, including the Georgetown–IBM demonstration of 1954, "slightly preceded the coining of the term 'artificial intelligence' in 1956" (Dædalus p. 128).

### C04 Language as human societal intelligence
- **Meaning:** The reading's motivating claim. Language matters because it lets people pool their intelligence, and writing extends that pooling across time and space.
- **Key points:**
  - Language "gives groups of people a way to network human brains together" (Dædalus p. 127).
  - "An individual human may not be much more intelligent than our close relatives of chimpanzees or bonobos" (p. 127).
  - Writing allowed "knowledge to be communicated across distances of time and space" (p. 127).
  - "language will retain an important role in a future world in which human abilities are augmented by artificial intelligence tools" (p. 128).
- **Scope:** From the suggested reading only; not on the slides.

### C05 Era 1: Early explorations (1940–1969)
- **Meaning:** The first era. NLP research began with machine translation, attempted with almost no knowledge of language and with tiny machines, while the field's mathematical foundations were being laid. The same era saw early question-answering and dialogue programs (C09).
- **Key points:**
  - "Foundational work on automata, formal languages, probabilistic modeling, and information theory"; "First speech systems (Davis et al.)"; "MT heavily funded by military — huge overconfidence"; "But using machines dumber than a pocket calculator"; "Little understanding of syntax, semantics, pragmatics" (history 19).
  - "NLP research began as research in machine translation"; "almost nothing was known about the structure of human language, artificial intelligence, or machine learning. The amount of computation and data available was, in retrospect, comically small" (Dædalus p. 128).
  - Systems "provided little more than word-level translation lookups and some simple, not very principled rule-based mechanisms" (Dædalus p. 128).

### C06 Machine translation as code-breaking
- **Meaning:** The founding idea of NLP: translation could be treated as decryption, with a foreign text seen as English in a code. Wiener objected at once that word meanings do not line up across languages.
- **Key points:**
  - "Machine Translation: The origin of NLP/Computational Linguistics" (history 4).
  - Weaver: "When I look at an article in Russian, I say: 'This is really written in English, but it has been coded in some strange symbols. I will now proceed to decode.'" (history 4–5, citing "Weaver (1955:18, quoting a letter he wrote in 1947)"; the passage is in Weaver's 1949 memorandum).
  - Wiener (April 1947): "the [semantic] boundaries of words in different languages are too vague … to make any quasi-mechanical translation scheme very hopeful" (history 5).
  - It was "imagined that translation could quickly build on the great successes of computers in code breaking during World War II"; on both sides of the Cold War, researchers sought to translate "the scientific output of other nations" (Dædalus p. 128).
  - Timeline: 1946, Warren Weaver and Andrew Donald Booth begin an MT project at Birkbeck College, London; 1949, Weaver's memorandum on translation (history 19).
- **Example:** History 5 sets Weaver ("a mathematician & engineer… science funder at the Rockefeller Foundation", co-author of an introduction to information theory with Shannon) against Wiener ("MIT originator of cybernetics").

### C07 Georgetown–IBM demonstration (1954)
- **Meaning:** The first public demonstration of machine translation: Russian sentences translated into English on an IBM 701, by Georgetown University and IBM.
- **Key points:**
  - "A public demonstration of what is believed to be the first successful use of a machine to translate meaningful texts from one language to another took place here yesterday afternoon" (NYT, 8 January 1954; history 7).
  - "So far the system has a vocabulary of only 250 words"; the rules of syntax were reduced "to six instructions for the data-processing calculator" (history 7–8).
  - An example output: "We transmit thoughts by means of speech" (history 8).
  - The promise: such a device "should be ready within three to five years" (history 8).
  - It "slightly preceded the coining of the term 'artificial intelligence' in 1956" (Dædalus p. 128).
- **Example:** The punch card "KACHYESTVO UGLYA OPRYEDYELYAYETSYA KALORYIYNOSTJYU", printed as "THE QUALITY OF COAL IS DETERMINED BY CALORY CONTENT" (history 8).

### C08 MT overconfidence and the ALPAC report (1966)
- **Meaning:** Early machine translation was oversold. ALPAC criticized contemporary MT progress and prospects; the lecture presents it as a reality check on early overconfidence.
- **Key points:**
  - "ALPAC report (1966): crap, this is really hard!" (history 23).
  - Timeline: "ALPAC Report by the US National Academy of Sciences (condemned MT field)" (history 23).
  - "MT heavily funded by military — huge overconfidence" (history 19); "initial systems were promoted with great fanfare" (Dædalus p. 128).
  - The report itself distinguishes machine translation from machine-aided translation (ALPAC, printed p. 32).
  - The report: "while we have machine-aided translation of general scientific text, we do not have useful machine translation. Further, there is no immediate or predictable prospect of useful machine translation" (ALPAC, printed p. 32).

### C09 Early question-answering and dialogue programs (BASEBALL 1961, ELIZA 1966)
- **Meaning:** Two early programs that worked in English. BASEBALL answers questions about a restricted baseball database; ELIZA generates replies through keyword-triggered decomposition and reassembly rules (Green et al. 1961, p. 219; Weizenbaum 1966, p. 36).
- **Key points:**
  - Timeline: "BASEBALL question-answering system (Green et al.)" (1961) and "ELIZA program (Weizenbaum)" (1966) (history 23).
- **Example:** The exhibition panel shows an ELIZA exchange: "> i'm unintelligent / ELIZA: How long have you been unintelligent?" (intro 10).
- **Scope:** Only named on the slides; the mechanisms come from the cited papers.

### C10 The Dartmouth Summer Research Project (1956)
- **Meaning:** The 1956 summer workshop the lecture marks as the start of AI.
- **Key points:**
  - "AI Begins: The Dartmouth Summer Research Project 1956" (history 10).
  - Timeline, 1956: "1st AI conference"; "Dartmouth Conference introduces term 'Artificial Intelligence'" (history 19). *Source correction:* the term already appears in the title and opening of McCarthy, Minsky, Rochester and Shannon's proposal for the workshop, dated 31 August 1955.

### C11 Connectionism (neural networks modelled on neurons)
- **Meaning:** The approach that models intelligence as networks of simple neuron-like units, inspired by the biology of neurons.
- **Key points:**
  - "Neural Networks: The inspiration from knowledge of neurons" (history 11).
  - The timeline marks three seminal works in connectionism: "1943: Seminal work in connectionism: McCulloch & Pitts on formal neurons"; "1949: … Seminal work in connectionism: Hebb on cell assemblies & synapses"; "1958: … Seminal work in connectionism: Rosenblatt on perceptrons" (history 19).
- **Example:** The neuron diagram of history 11 (Figures).

### C12 McCulloch–Pitts threshold unit
- **Meaning:** A formal binary neuron that outputs 1 when its weighted input sum exceeds its threshold. McCulloch and Pitts's 1943 paper used networks of such units as the first mathematical model of a neural network.
- **Symbols:** inputs xᵢ, weights wᵢⱼ, net input netⱼ = Σᵢ wᵢⱼxᵢ, threshold θⱼ, output oⱼ = 𝟏(netⱼ > θⱼ); in vector form 𝟏(Wx > θ) = 𝟏(Wx − θ > 0) (history 13).
- **Key points:**
  - "In 1943, published 'A Logical Calculus of Ideas Immanent in Nervous Activity', the first mathematical model of a neural network, showing it could implement a Turing Machine" (history 12). The Turing-machine claim concerns networks of these units, not one threshold unit.
  - "This function has no slope, so, no gradient-based learning" (history 13). That is, the hard step has derivative zero away from the threshold and is undefined at the threshold, so direct backpropagation through it supplies no useful learning gradient.
- **Example:** History 12 sketches the authors: McCulloch, "Neurophysiologist (MD), psychiatrist, and cyberneticist"; Pitts, "Self-taught logician and computational neuroscientist, lived homeless at Chicago and then MIT".
- **Scope:** The threshold model only; how networks of units are trained comes later in the course.

### C13 Rosenblatt's perceptron
- **Meaning:** Frank Rosenblatt's perceptron (1958), a neural network that learned from examples. Rosenblatt also built it in hardware as the Mark I Perceptron at Cornell.
- **Key points:**
  - "Frank Rosenblatt: The (Mark I) Perceptron" (history 14): photographs of Rosenblatt at the machine and of its wired plugboard.
  - "Early AI Hype!" New York Times, 8 July 1958: "NEW NAVY DEVICE LEARNS BY DOING… The Navy revealed the embryo of an electronic computer today that it expects will be able to walk, talk, see, write, reproduce itself and be conscious of its existence" (history 15).
  - That July 1958 demonstration ran on the Weather Bureau's IBM 704, not on the pictured Mark I: "the Weather Bureau's $2,000,000 '704' computer—learned to differentiate between right and left after fifty attempts" (history 15; the Cornell story linked on history 14 confirms this).
  - Its limits: in 1969, Minsky & Papert "showed that perceptrons may be less powerful than a universal Turing machine" (history 23).

### C14 Two visions of AI in the 1950s
- **Meaning:** Early AI split into two camps. Symbolic AI manipulates symbols and rules; cybernetics models communication, control and feedback in brain-like systems.
- **Key points:**
  - "(Symbolic) Artificial Intelligence: Minsky, McCarthy, Simon, Newell" versus "Cybernetics: Wiener, Rosenblatt" (history 16).
  - Cybernetics "sought to tie together communication, control, and feedback in living things and computers" (history 5, on Wiener).

### C15 Information retrieval and Bush's memex
- **Meaning:** Information retrieval (IR) is finding the stored documents relevant to a need. The lecture presents Vannevar Bush's 1945 "memex" as its founding vision: a personal mechanized library that can be consulted quickly.
- **Key points:**
  - "Consider a future device for individual use, which is a sort of mechanized private file and library. It needs a name, and, to coin one at random, 'memex' will do. A memex is a device in which an individual stores all his books, records, and communications, and which is mechanized so that it may be consulted with exceeding speed and flexibility. It is an enlarged intimate supplement to his memory." (Bush 1945, "As We May Think"; history 17). The slide's text matches the Atlantic original.
  - Bush chaired the National Defense Research Committee in 1940–1941, then directed the Office of Scientific Research and Development, which coordinated major U.S. wartime research. (History 17 runs the two roles together as "chairman of the National Defense Research Committee in WWII (which had huge funding, directing all wartime science, including the Manhattan Project)"; the Atlantic text's opening names him OSRD director.)

### C16 Benchmark evaluation (Cleverdon's Cranfield tests)
- **Meaning:** Measuring a language system against a fixed test collection with known correct answers. Cyril Cleverdon's Cranfield tests (1957–1967) defined the idea.
- **Key points:**
  - "Cyril Cleverdon introduces benchmarking"; "Cranfield tests (1957–1967)"; "Defined the idea of benchmark tests for language, with a document collection, queries, and correct answers" (history 18).
  - "He had exhaustive answers on document relevance over a small corpus!" (history 18).

### C17 The Association for Computational Linguistics (ACL), founded 1962
- **Meaning:** The field's professional society. It was founded in 1962 under a name that shows its MT origins: the Association for Machine Translation and Computational Linguistics.
- **Key points:**
  - Timeline, 1962: "Founding of Association of Computational Linguistics (ACL), originally called 'Association for MT & Computational Linguistics' (AMTCL)" (history 23).
  - "1960s: David Glenn Hays: ACL founder and dependency parsing" (history 20).

### C18 Era 2: Hand-built symbolic NLP (1970–1992)
- **Meaning:** The era of rules and knowledge written by hand, growing more formal, demonstrated on narrow domains.
- **Key points:**
  - "a whole series of NLP demonstration systems that showed sophistication and depth in handling phenomena like syntax and reference in human languages"; "These were all hand-built, rule-based systems"; "Some systems were even deployed operationally for tasks like database querying" (Dædalus p. 128).
  - In the era's second decade, "a new generation of hand-built systems emerged, which had a clear separation between declarative linguistic knowledge and its procedural processing" (Dædalus p. 128).
  - "Logic programming, rules-driven AI, deterministic algorithms for syntactic parsing (e.g., LFG)"; "But symbolic AI hit the wall: 'AI winter'" (history 31).
  - Meanwhile, speech recognition did "Foundational work… stochastic modeling, hidden Markov models, the 'noisy channel'", and "Ideas from this work would later revolutionize NLP!" (history 31).

### C19 Natural-language-understanding demonstration systems (SHRDLU and peers)
- **Meaning:** Programs that understood English within one narrow world or database, built from hand-written rules.
- **Key points:**
  - "Increasing interest in natural language understanding: SHRDLU, LUNAR, CHAT-80" (history 31); timeline: 1972 "SHRDLU program (Winograd)", 1973 "LUNAR program (Woods)", 1978 "LIFER program (Hendrix et al.)" (history 31).
  - The reading adds "Roger Schank's systems such as SAM" and "GUS by Danny Bobrow" (Dædalus p. 128).
  - History 25: Terry Winograd ("1946–, Stanford faculty 1973–2014; AI/NLP 1973–1984"), with the lineage Seymour Papert → Terry Winograd → Stuart Shieber. The slide's right half is an embedded video that does not render.
- **Example:** The exhibition panel of coloured blocks and a pyramid (intro 11). Reading it as SHRDLU's blocks world is our interpretation; the panel has no label.

### C20 Formal and unification-based grammars
- **Meaning:** Precise, mathematically defined grammar formalisms used for parsing in era 2's second decade. *Our gloss:* they combine feature structures by unification. The slides name them without explaining them.
- **Key points:**
  - "Formal and Unification-based Grammars for NLP: Martin Kay" (history 26); Fernando Pereira and Stuart Shieber (history 27).
  - LFG is the slides' example of a parsing formalism (history 31).

### C21 Compositional semantics
- **Meaning:** Building a sentence's meaning in three steps: tokenize it into words, parse it into a tree, then combine the meanings of the parts up the tree.
- **Symbols:** λ (abstraction), ι (the definite description, for "the"), ∧ (and).
- **Key points:**
  - "Constructing the meaning of human language: 1967–2017" (history 28).
  - "Tokenize a sentence into words"; "Parse it into a tree or graph data structure (using context-free grammars and beyond)"; "Construct its meaning by: (i) lexical lookup, then (ii) semantic composition, using a 'rule-to-rule' approach working up the tree (e.g., PP: α(β) → P: α NP: β)" (history 28).
- **Example:** "The red apple is on the table" ↦ on(ι(λx(apple(x) ∧ red(x))), ι(λy.table(y))). The lexicon: the = λP.ι(P), red = λP.λx(P(x) ∧ red(x)), apple = λx.apple(x), is = λP.P, on = λy.λx(on(x, y)), table = λy.table(y) (history 28).

### C22 Knowledge-based inference for text understanding (Norvig 1986)
- **Meaning:** Understanding a text by drawing inferences over a knowledge base, beyond what the text literally says.
- **Key points:**
  - Peter Norvig's thesis, "A Unified Theory of Inference for Text Understanding (1986)" (history 29).
  - "As we have just seen, a suitable knowledge base is a prerequisite for making proper inferences" (p. 4); "It's built to enable inferences" (history 30).
  - "System had 6 general forms of inference; 2 pairs, so 4 basic types" (history 30):
    1. Elaboration fills missing connections between entities ("Filling a slot to connect two entities"): "John got piggybank for REASON have money for REASON buy present".
    2. Reference resolution identifies what a word refers to ("Hey – it's coreference!!!").
    3. View application reads a word in a different frame: in "The Red Sox killed the Yankees", "KILLED is not animal; KILLING is viewed as a DEFEAT-CONVINCINGLY".
    4. Concretization infers something more specific: "TRAVELLING in an AUTOMOBILE is an instance of DRIVING".
- **Example:** The Chang Lee story (history 29). The inferences required include "(a) There is a sea, which surrounds the island, is used by the villagers for fishing, and forms part of the coast of China", "(c) The word which refers to the fish" and "(d) The word they refers to Chang and his mother".

### C23 The physical symbol system hypothesis (Newell & Simon)
- **Meaning:** Physical symbol systems are necessary for general intelligence and, if sufficiently large and appropriately organized, can support general intelligent action.
- **Key points:**
  - Timeline, 1975: "ACM Turing Award lecture by Allen Newell & Herbert Simon formalized the 'physical symbol hypothesis'" (history 31).
  - The authors' statement: "A physical symbol system has the necessary and sufficient means for general intelligent action" (Newell & Simon, November 1975 manuscript, PDF p. 10).
- **Scope:** A necessary/sufficient-means hypothesis, not an identity between any symbol manipulation and intelligence. History 40 (U8) argues against designing NLP systems this way.

### C24 Era 3: Statistical NLP and supervised machine learning (1993–2012)
- **Meaning:** The era in which NLP reoriented around models learned from large amounts of text: first by counting, then mainly by supervised learning on annotated data.
- **Key points:**
  - "digital text became abundantly available"; "This led to a fundamental reorientation of the field around empirical machine learning models of NLP, an orientation that still dominates the field today" (Dædalus p. 129).
  - Early text collections "were generally in the low tens of millions of words", and models were extracted "largely by counting particular facts" (Dædalus p. 129).
  - The 2000s (history 37): "Availability of much more powerful machines"; "Consolidation of the gains of the statistical revolution"; "More sophisticated statistical modeling and machine learning algorithms: MaxEnt, SVMs, Bayes Nets, LDA, etc."; "Big big data: 100x growth of web, massive server farms"; "Focus shifting from supervised to unsupervised learning"; "Revived interest in higher-level semantic applications".

### C25 The noisy channel model (Shannon 1948)
- **Meaning:** Shannon's model of communication. A message from an information source is sent as a signal, corrupted by a noise source, and reconstructed by a receiver for its destination.
- **Key points:**
  - "A Mathematical Theory of Communication (1948)"; "The noisy channel model" (history 34).
  - In the 1970s–80s, speech recognition's foundational work included "stochastic modeling, hidden Markov models, the 'noisy channel'" (history 31).
- **Example:** The diagram on history 34 (Figures).

### C26 The statistical revolution: probabilistic modeling of language
- **Meaning:** Treating language with probability models estimated from corpus counts, using ideas borrowed from electrical engineering and speech recognition, and accepting narrower goals than full understanding.
- **Key points:**
  - "Influx of new ideas from EE & ASR: probabilistic modeling, corpus statistics, supervised learning, empirical evaluation" (history 35).
  - "Lowered expectations: forget full semantic understanding, let's do text cat, part-of-speech tagging, NER, and parsing!" (history 35).
  - "Tools: Naïve Bayes classifiers, HMMs, PCFGs, CRFs (CS228!)" (history 35).
  - Timeline, 1985: "Church's work encourages reintroduction of stochastic techniques" (history 31).
- **Example:**
  - Probabilistic spelling correction (history 36). For "She is a stellar and versatile acress whose combination of sass and glamour attracts", compare candidates by the probability of the word sequence: P(actress\|versatile) = .0019 and P(whose\|actress) = .0043, against P(across\|versatile) = .000092 and P(whose\|across) = .000026. "actress" wins by a wide margin.
  - *Source defect:* the slide's products are misprinted. They use .00019 for P(actress\|versatile), and both printed results (817,000 ×10⁻¹⁰ and 239 ×10⁻¹⁰) are ten times too large. The correct values are .0019 × .0043 ≈ 81,700 ×10⁻¹⁰ and .000092 × .000026 ≈ 23.9 ×10⁻¹⁰. The comparison, about 3,400 to 1, is unchanged.
  - The parse tree for "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo" (history 35): a grammatical but highly ambiguous string, the kind a probabilistic grammar must resolve.

### C27 Supervised learning from annotated corpora
- **Meaning:** Humans label text (parse trees, names, word senses), and models learn from those labels to produce the same labels on new text.
- **Key points:**
  - "New sources of data: explosion of machine-readable text; human-annotated training data (e.g., the Penn Treebank)"; "Annotated data + algorithms + probabilistic prediction" (history 35).
  - "counts on words only go so far in providing language understanding devices"; this "led most of the field to concentrate on constructing annotated linguistic resources, such as labeling the sense of words, instances of person or company names in texts, or the grammatical structure of sentences in treebanks, followed by the use of supervised machine learning techniques to build models that could produce similar labels on new pieces of text at runtime" (Dædalus p. 129).

### C28 Era 4a: Neural NLP (2013–2021)
- **Meaning:** Deep learning replaced hand-built features: words and sentences became vectors, and similarity of meaning became proximity.
- **Key points:**
  - "4a. Neural NLP 2013–2021" (history 38).
  - Words and sentences "are represented by a position in a (several hundred- or thousand-dimensional) real-valued vector space, and similarities of meaning or syntax are represented by proximity in this space" (Dædalus p. 129).
  - From 2013 to 2018, "it was easier to model longer distance contexts, and models generalized better to words or phrases with similar meanings"; "Nevertheless, the approach was unchanged in building supervised machine learning models to perform particular analysis tasks" (Dædalus p. 129).
- **Scope:** The task-specific supervised approach is the reading's description of 2013–2018, not of the whole period to 2021.

### C29 Symbol systems versus their processors
- **Meaning:** Manning's argument that language is a symbol system, but the brain that processes it need not be one, so NLP systems need not be built as one either.
- **Key points:**
  - Symbolic Systems "studies systems of meaningful symbols that represent the world about us—like human languages, logics, and programming languages—and the systems that work with these symbols—like brains, computers, and complex social systems" (history 39, "Stanford, the home of 'Symbolic Systems'"; Jon Barwise pictured).
  - "Language is the symbolic system par excellence; we should study and make use of its symbolic structure"; "This does not show that the main processor of these symbols—the human brain—is implemented as a physical symbol system"; "We need not design NLP systems as physical symbol systems"; "The brain is more like a neural network model"; "Artificial neural network models scale better and can capture the world represented by symbols" (history 40).

### C30 Deep learning's breakthrough (2009–2012), first in speech
- **Meaning:** Deep neural networks trained on large datasets improved markedly on GMM-based speech-recognition baselines, the first field where deep learning broke through.
- **Key points:**
  - "Deep Learning Breakthroughs by Hinton… at the University of Toronto, 2009–2012" (history 41).
  - "Speech recognition had first shown the breakthrough success of probabilistic methods: HMMs and GMMs"; "The first breakthrough results of 'deep learning' on large datasets also happened in speech recognition"; "George Dahl et al. (2010/2012): Context-Dependent Pre-trained Deep Neural Networks for Large Vocabulary Speech Recognition" (history 42). Dahl's system is itself a probabilistic DNN–HMM hybrid.
  - The slide's word-error-rate table (RT03S FSH / Hub5 SWB): Traditional GMM 27.4 / 23.6; Deep Learning (Dahl et al. 2012) 18.5 (−33%) / 16.1 (−32%); Deep Learning (Saon et al. 2017) 8.0 (−71%) / 5.5 (−77%) (history 42).
  - *Attribution note:* Dahl et al. 2012 (TASLP) reports a Bing business-search dataset (relative error reductions of 16.0% and 23.2%), not these RT03S/Hub5 figures. Saon et al. 2017 reports RT'03 = 8.0% in Table 9's final row (PDF p. 5), after periods are removed from letter spellings in both references and outputs; before that normalization the table reports 8.3%. Its SWB result is 5.5%.

### C31 Word vectors
- **Meaning:** Each word is represented as a vector of real numbers learned from data, so that "Similar vectors = similar meaning" (history 43).
- **Key points:**
  - "We learn to represent a word as a vector of numbers" (history 43).
  - "Word Vectors are often used as a fundamental component for downstream NLP tasks, e.g. question answering, text generation, translation" (A1 "Word Vectors").
  - "word vectors" and "word embeddings" are used interchangeably; "embedding" refers to "encoding aspects of a word's meaning in a lower dimensional space" (A1 "Word Vectors").
- **Example:** versatile = (0.286, 0.792, −0.177, −0.107, 0.109, −0.542, 0.349, 0.271) (history 43).
- **Scope:** Introduced here; developed in Lecture 2 and Assignment 1 (U11).

### C32 Distributional similarity
- **Meaning:** A word's meaning can be learned from the contexts it appears in.
- **Key points:**
  - "You shall know a word by the company it keeps" (J. R. Firth 1957: 11; history 44). Firth's chapter has the line on p. 11.
  - "How do we learn those word vectors?"; "Defining similarity via contextual distributions in texts is one of the most successful ideas of modern computational linguistics" (history 44).
  - "Many word vector implementations are driven by the idea that similar words, i.e., (near) synonyms, will be used in similar contexts" (A1 Part 1).
- **Example:** The contexts "any devices with a web browser, from laptops and tablets to smart phones" and "Users can download it for home computers or laptops from Microsoft Update website": "These words will represent laptops" (history 44).

### C33 Neural encoder–decoder for translation (seq2seq)
- **Meaning:** One network, the encoder, reads the source sentence into a vector state. A second, the decoder, generates the translation word by word, feeding in each word it has just produced.
- **Symbols:** The basic RNN recurrence, hₜ = tanh(W[xₜ] + Uhₜ₋₁ + b): each new hidden state hₜ combines the current input word xₜ with the previous state hₜ₋₁ (history 45).
- **Key points:**
  - "An RNN encoder-decoder network" (history 45), using the basic recurrence above.
  - "An LSTM encoder-decoder network [Sutskever et al. 2014]": a separate, three-layer LSTM architecture; "Encoder: Builds up sentence meaning"; "Feeding in last word"; "Translation generated" (history 46).
  - "Bottleneck" (history 46): the whole source sentence is compressed into a fixed-size encoder state that is passed to the decoder.
- **Example:** "I am a student <EOS>" → "Je suis étudiant <EOS>" (history 45); "Die Proteste waren am Wochenende eskaliert" → "The protests escalated over the weekend" (history 46).

### C34 Neural MT overtakes statistical MT (2015–2016)
- **Meaning:** Within two years neural MT went from behind to ahead of phrase-based and syntax-based statistical MT, and Google switched to it in 2016.
- **Key points:**
  - "MT progress over time [Edinburgh En-De WMT newstest2013 Cased BLEU; NMT 2015 from U. Montréal]" (history 47). The chart shows neural MT below both statistical systems in 2015 and above both in 2016.
  - "in late 2016, Google's machine translation improved markedly when they switched to the use of neural machine translation" (Dædalus p. 132; history 47 links the New York Times story "The Great A.I. Awakening").

### C35 Era 4b: Large language models (2022–present)
- **Meaning:** The current era, in which very large neural language models became general-purpose tools and a mass phenomenon.
- **Key points:**
  - "4b. Large Language Models 2022–present" (history 48).
  - "The Rise of Generative AI: S&P 500 Companies Discussing 'AI'": FactSet's count of earnings calls citing "AI" rises sharply after the "ChatGPT launch", passing "40% of S&P 500" and then "50% of S&P 500" (history 49).
- **Example:** ChatGPT 4o asked to "Write a sonnet praising Wikipedia in which each line begins with the letter W" produces "Wikipedia: A Sonnet of Wonder" (history 54). History 50 shows a GPT-4o image of "a large language model as a big machine that takes in text and produces fresh writing on an endless scroll".

### C36 Language model (probabilistic next-token prediction)
- **Meaning:** A model that assigns probabilities to what comes next in a text, given what came before. It is an old idea, running from Markov through Shannon to IBM.
- **Key points:**
  - 1913: "Andrey A. Markov explores consonant-vowel transition probabilities in Alexander Pushkin novel, Eugene Onegin, developing Markov Models" (history 51).
  - 1948: Claude Shannon "explores character/word-level n-gram models, entropy, text generation" (history 51). *Source note:* the slide calls the work "A Mathematical Theory of Information"; the title is "A Mathematical Theory of Communication" (history 34).
  - 1975: "Frederick Jelinek's group at IBM defines and names the modern concept of a (probabilistic) 'language model' for next token prediction. Used for spelling correction, speech recognition, machine translation…." (history 51).
  - "But this technology wasn't considered a pathway to machines with artificial intelligence: For that we needed models of memory, knowledge representations, planning systems, and reasoning over abstract concepts" (history 51).
- **Example:** The n-gram factorizations of P(this, is, a, sentence) on the exhibition panel (intro 11): N=1, P(this)P(is)P(a)P(sentence); N=2, P(this)P(is\|this)P(a\|is)P(sentence\|a); N=3, P(this)P(is\|this)P(a\|this, is)P(sentence\|is, a). Probabilistic spelling correction (history 36, C26) is one of the uses named on history 51.

### C37 Scaling up language models
- **Meaning:** "Large" language models are older than neural ones. Each wave was held back by compute, model flexibility or data, until all three arrived together.
- **Key points (history 52):**
  - 1998: "CPAT-Tree-Based Language Models with an Application for Text Verification in Chinese. ROCLing 1998. First use of 'LLM' trigram I know of; 200M word corpus".
  - 2000: "A Neural Probabilistic Language Model. Bengio, Ducharme & Vincent NIPS 2000. First neural language model built on 32 million token corpus, 31K vocab".
  - 2007: "Large Language Models in Machine Translation. Brants, Popat, Xu, Och and Dean. EMNLP 2007. 2 trillion token corpus n-gram model of up to 5-grams".
  - 2018: "GPT (Radford, Narasimhan, Salimans & Sutskever) and BERT (Devlin, Chang, Lee & Toutanova). 3.3 billion token corpus". *Source correction:* the 3.3 billion figure is BERT's corpus, BooksCorpus (800M words) plus Wikipedia (2,500M words), counted in words; GPT used BooksCorpus alone (BERT PDF pp. 5, 14; GPT §4.1).
  - 2020–: "100+ billion parameter neural language models trained on > 1 trillion tokens: GPT-3, GPT-4, PaLM 2, Llama 3, Nemotron-4, …." *Source correction:* this does not hold for every named model. GPT-3 trained on 300B tokens (Brown et al., Table 2.1). Sizes and token counts that the models' authors did not disclose are not inferred from the group label.
  - History 53 overlays "Not enough compute!" on the 2000 row, "Not enough model flexibility!" on 2007, "Not enough data!" on 2018 and "LLMs go brrr!" on 2020–. Reading each label as the limit of the row beneath it is our interpretation of the placement.

### C38 Evidence on what LLMs can and cannot do (2023 studies)
- **Meaning:** Two studies the lecture uses to show that LLMs help people a great deal on some tasks and still fall short on others.
- **Key points:**
  - "Consultants (from Boston Consulting Group) using GPT-4 finished 12.2% more tasks on average, completed tasks 25.1% more quickly, and produced 40% higher quality results than those not using AI"; "Use of an LLM particularly improved the performance of lower-performing humans"; "Results vary depending on the task" (history 55, citing Dell'Acqua et al. 2023 and Mollick 2023).
  - *Scope, from the paper:* the gains are for 18 tasks "within the frontier of AI capabilities". For a task selected to be outside the frontier, subjects using AI were 19 percentage points less likely to produce correct solutions (84.5% correct without AI, against 60% and 70.6% in the two AI groups; PDF p. 12). The 2026 published version reports quality "enhanced… by an average of 32%"; the slide's 40% is the 2023 working paper's figure. The paper confirms that "the most significant beneficiaries of using AI are the bottom-half-skill subjects".
  - "Can GPT-4 write fiction that matches the quality of New Yorker fiction? Good news! No! (At least in 2023….) GPT-4 is still 3–10x worse at creative writing!" (history 56). The paper's claim is narrower: TTCW is the study's 14 binary creative-writing tests, judged by human writers, and "LLM-generated stories pass 3-10X less TTCW tests than stories written by professionals" (Chakrabarty et al., abstract). The 3–10× comparison counts tests passed; it is not a general scale of creativity.

### C39 Self-supervised pretraining and large pretrained language models (LPLMs)
- **Meaning:** A model learns from raw text by setting itself prediction problems (the next word, or a masked word), gains broad knowledge of language and the world, and can then be adapted to many tasks.
- **Key points:**
  - "Everything changed in 2018, when NLP was the first major success of very large scale self-supervised neural network learning" (Dædalus p. 129).
  - The system creates "from the text its own prediction challenges, such as successively identifying each next word in the text given the previous words or filling in a masked word or phrase in a text"; this way "general knowledge of a language and the world is accumulated" (pp. 129–130).
  - One pretrained model "can be very easily adapted, via fine-tuning or prompting" (p. 130). Fine-tuning, the standard "from 2018 to 2020", is "a small amount of additional supervised learning, training it on the exact task of interest" (p. 131).
  - Masked-word prediction seems "a kind of humorless Mad Libs" but "this task turns out to be very powerful because it is universal: every form of linguistic and world knowledge… help one to do this task better" (p. 131).
  - "these models learn and represent the syntactic structure of a sentence and will learn to memorize many facts of the world" (p. 131).
  - "In hindsight, the development of large-scale self-supervised learning approaches may well be viewed as the fundamental change" (p. 130).

### C40 The transformer and attention
- **Meaning:** The dominant neural architecture for NLP since 2018. Its core is attention, "by which a representation at a position is computed as a weighted combination of representations from other positions" (Dædalus p. 130).
- **Symbols:** For each position, a query Q, key K and value V (Fig. 1).
- **Key points:**
  - "Since 2018, the dominant neural network model for NLP applications has been the transformer neural network" (p. 130).
  - Each position yields "vectors that represent a query, key, and value"; the attention weights produce "a weighted average of the values at all positions"; this is "repeated many times at each level", then manipulated "through a fully connected neural net layer and through use of normalization layers and residual connections", with many layers stacked (p. 130).
  - *Source note:* the text says "The query at a position is compared with the value at every position". Figure 1 and the standard formulation compare the query with the keys; the values are what get averaged.
- **Example:** Figure 1 (p. 131). From "[CLS] Judiciary [MASK] Annual Report", the model predicts "committee" at the mask.

### C41 Prompting (in-context task specification)
- **Meaning:** Getting a large model to do a new task just by describing it or showing examples in its input, with no further training.
- **Key points:**
  - The largest models, "such as GPT-3 (Generative Pre-trained Transformer-3), can perform novel tasks very well with just a prompt. Give them a human language description or several examples of what one wants them to do, and they can perform many tasks for which they were never otherwise trained" (Dædalus pp. 131–132).

### C42 From NLP pipelines to adapted pretrained models
- **Meaning:** Traditional NLP chained separately built components; companies are replacing those pipelines with one pretrained model, usually fine-tuned for the task.
- **Key points:**
  - Traditional models were "elaborately composed from several usually independently developed components, frequently built into a pipeline"; "companies have started to replace such traditional NLP solutions with LPLMs, usually fine-tuned to perform particular tasks" (Dædalus p. 132).
  - Google Translate: statistical from 2006; neural in late 2016; then transformer-based in 2020, as "one huge neural net that was simultaneously trained on all languages that Google Translate covers, with input simply marked by a distinct token that indicates the language" (p. 132).
  - "Modern neural network question-answering systems have high accuracy in extracting an answer present in a text and are even fairly good at working out that no answer is present" (p. 133).
  - "there are still reasons to be skeptical as to whether these systems really understand what they are doing, or whether they are just very elaborate rewriting systems, bereft of meaning" (p. 134).
- **Example:** The Le Monde passage on Charles Trenet, translated (p. 132); the Samsung Galaxy Note 20 Ultra questions answered with UnifiedQA ("What is the optical zoom of the Galaxy Note 20 Ultra? 5x"; p. 133); a system-generated radiology impression next to the radiologist's (pp. 133–134).

### C43 Meaning as a network of connections (denotational vs distributional semantics)
- **Meaning:** Manning's answer to whether models trained on text learn meaning. Denotational semantics holds that a form's meaning is "the set of objects or situations in the world that it describes"; distributional semantics, that "the meaning of a word is simply a description of the contexts in which it appears". Manning's view: "meaning arises from understanding the network of connections between a linguistic form and other things, whether they be objects in the world or other linguistic forms" (Dædalus p. 134).
- **Key points:**
  - Against those who call distributional semantics "just a regurgitation of distributional or syntactic facts" (p. 134, n. 17): "Meaning is not all or nothing" (p. 134).
  - "there can be no doubt that pretrained language models learn meanings", but their knowledge is "often very incomplete and cry out for being augmented with other sensory data and knowledge" (p. 135).
- **Example:** The shehnai (pp. 134–135): someone who has held one, heard one, been told it is "like a traditional Indian oboe", or met it only in two textual contexts each has some meaning for the word. The reader who knows only the textual contexts can know cultural associations that someone who has merely held a shehnai lacks; that reader's understanding is therefore not simply a subset of the holder's.

### C44 Foundation models
- **Meaning:** "the general class of models with millions of parameters trained on copious data via self-supervision that can then easily be adapted to perform a wide range of downstream tasks" (Dædalus p. 135).
- **Key points:**
  - "LPLMs like BERT… and GPT-3 are early examples of foundation models" (p. 135).
  - Directions: connecting models with "a knowledge graph… or as a large supply of text to be consulted at runtime"; most promising, foundation models that "take in other sensory data", such as DALL·E (p. 135).
  - Risks of convergence on a few models: "the groups capable of building these models may have excessive power and influence, many end users might suffer from any biases present in these models, and it will be difficult to tell if models are safe to use" (p. 136).
  - "they are likely to remain limited, lacking a human-level ability for careful logical or causal reasoning", yet "they will give people in the coming decade their first glimpses of a more general form of artificial intelligence" (p. 136).

### C45 Co-occurrence matrix (count-based word vectors)
- **Meaning:** "a symmetric word-by-word matrix in which Mᵢⱼ is the number of times wⱼ appears inside wᵢ's window among all documents". The window is "the n preceding and n subsequent words" (A1 Part 1). Its rows give count-based word vectors.
- **Symbols:** M (the matrix), n (window size), wᵢ (the centre word), wᵢ₋ₙ … wᵢ₋₁ and wᵢ₊₁ … wᵢ₊ₙ (its window).
- **Key points:**
  - Many "old school" approaches to word vectors "relied on word counts" (A1 Part 1).
  - `<START>` and `<END>` tokens "are included in co-occurrence counts, encapsulating each document" (A1 Part 1).
  - The rows "provide word vectors based on word-word co-occurrence, but they can be large" (A1 Part 1).
- **Example:** The n=1 matrix for Document 1 "all that glitters is not gold" and Document 2 "all is well that ends well" (A1 Part 1); every cell checks out against the two documents.

### C46 Dimensionality reduction by truncated SVD
- **Meaning:** Shortening count-based vectors by keeping only the top k singular components of the co-occurrence matrix.
- **Symbols:** A ≈ Uₖ Sₖ Vₖᵀ (rank-k truncation); the embeddings are Uₖ Sₖ.
- **Key points:**
  - "To reduce dimensionality, we employ Singular Value Decomposition (SVD), akin to PCA, selecting the top k principal components" (A1 Part 1). These are the top singular components of the uncentered matrix; equivalence with PCA requires centering.
  - "TruncatedSVD returns U*S, so we need to normalize the returned vectors, so that all the vectors will appear around the unit circle (therefore closeness is directional closeness)" (Q1.5). Normalizing assumes nonzero rows.
  - A1's illustrative expectation: "*doctor* and *hospital* will be closer than *doctor* and *dog*" (A1 Part 1). This is the assignment's expectation for these vectors, not a guarantee of SVD.
  - "scalable techniques such as Truncated SVD exist for extracting the top k vector components efficiently" (A1 Part 1). Background readings: Baker's SVD tutorial and CS168 lectures 7–9 (`a1-readings/`).

### C47 Learned dense word vectors (word2vec, GloVe)
- **Meaning:** Word vectors learned by optimizing an objective rather than read off raw counts. A1 groups them as "prediction-based". The mechanisms differ: word2vec predicts local context words, while GloVe learns vectors by weighted regression on global co-occurrence counts (GloVe paper pp. 1532–1535).
- **Key points:**
  - "more recently prediction-based word vectors have demonstrated better performance, such as word2vec and GloVe (which also utilizes the benefit of counts)" (A1 Part 2).
  - A1 uses 200-dimensional GloVe vectors, reduced to 2 dimensions for plotting (A1 Part 2).
- **Scope:** Named and used here; the algorithms are taught in Lecture 2 ("revisit the class notes and lecture slides", A1 Part 2). A1's prediction-versus-counts grouping is its terminology, not a claim that GloVe uses no counts.

### C48 Cosine similarity (and cosine distance)
- **Meaning:** Similarity measured by the angle between two vectors, rather than by the distance between them as points.
- **Symbols:** s = (p · q) / (‖p‖ ‖q‖), with s ∈ [−1, 1], for nonzero vectors p and q (A1 "Cosine Similarity"). Cosine distance = 1 − s (Q2.3).
- **Key points:**
  - "L1 and L2 Distances help quantify the amount of space 'we must travel' to get between these two points. Another approach is to examine the angle between two vectors" (A1 "Cosine Similarity").
  - `most_similar` "ranks all other words in the vocabulary with respect to their cosine similarity to the given word" (Q2.2).

### C49 What nearest neighbours conflate: polysemy and antonymy
- **Meaning:** One vector per word can mix the word's senses, and words used in the same contexts can be close even when they are opposite in meaning.
- **Key points:**
  - "Polysemes and homonyms are words that have more than one meaning" (Q2.2).
  - Q2.3 asks for a triple in which the antonym is closer than the synonym.
- **Example:** "leaves" has both "go_away" and "a_structure_of_a_plant" meanings in its top 10, and "scoop" has both "handed_waffle_cone" and "lowdown" (Q2.2); "happy" is closer to "sad" than to "cheerful" (Q2.3).

### C50 Word-vector analogies
- **Meaning:** "Word vectors have been shown to *sometimes* exhibit the ability to solve analogies" (Q2.4). An analogy x : y :: a : b is solved by finding the word most similar to the positive words and most dissimilar from the negative ones.
- **Key points:**
  - `most_similar` "finds words that are most similar to the words in the `positive` list and most dissimilar from the words in the `negative` list (while omitting the input words, which are often the most similar; see this paper)" (Q2.4; Schluter 2018). "The answer to the analogy will have the highest cosine similarity" (Q2.4).
- **Example:** "man : grandfather :: woman : x" (Q2.4), with near misses "granddaughter", "daughter" and "mother" (Q2.5); the failure "hand : glove :: foot : **sock**", which returns "45,000-square" (Q2.6); the exhibition's king/queen panel (intro 14).

### C51 Bias in word vectors
- **Meaning:** Word embeddings absorb social biases from the text they are trained on.
- **Key points:**
  - "It's important to be cognizant of the biases (gender, race, sexual orientation etc.) implicit in our word embeddings. Bias can be dangerous because it can reinforce stereotypes through applications that employ these models" (Q2.7).
- **Example:** Comparing the terms most similar to "man" and "profession" and most dissimilar to "woman" with those for "woman" and "profession" against "man" (Q2.7).

## Problems

What each end-of-chapter problem asks and which concepts it exercises. Record no solutions the source doesn't give.

| Problem | Asks | Exercises |
|---|---|---|
| A1 Q1.1 `distinct_words` [code] | Return the sorted distinct word types of the corpus | C45 (setup) |
| A1 Q1.2 `compute_co_occurrence_matrix` [code] | Build the co-occurrence matrix for a window of n (default 4) | C45 |
| A1 Q1.3 `reduce_to_k_dim` [code] | Reduce the matrix to k-dimensional embeddings with TruncatedSVD | C46 |
| A1 Q1.4 `plot_embeddings` [code] | Plot 2-D word vectors with labels | C46 |
| A1 Q1.5 [written] | On the 2-D co-occurrence plot of movie-review words: name two clusters and explain them; name what fails to cluster that might be expected to | C45, C46 |
| A1 Q2.1 [written] | Compare the 2-D GloVe plot with Q1.5's: one difference, one similarity, and why they might differ | C46, C47 |
| A1 Q2.2 [code + written] | Find a polysemous word whose top 10 neighbours show two senses; explain why most attempts fail | C48, C49 |
| A1 Q2.3 [code + written] | Find w₁, w₂ (synonym), w₃ (antonym) with cosine distance (w₁, w₃) < (w₁, w₂); explain | C48, C49 |
| A1 Q2.4 [written] | Write the vector expression whose cosine similarity is maximized in man : grandfather :: woman : x | C50 |
| A1 Q2.5 [code + written] | Explain the near misses (granddaughter, daughter, mother); find an analogy that holds | C50 |
| A1 Q2.6 [code + written] | Explain why hand : glove :: foot : sock fails; find another failing analogy | C50 |
| A1 Q2.7 [written] | Compare the male- and female-associated profession lists and explain the gender bias | C51 |
| A1 Q2.8 [code + written] | Find another analogy pair that shows bias in the vectors | C50, C51 |
| A1 Q2.9 [written] | Explain how bias enters word vectors; describe one way to mitigate it | C51 |

## Forward references

Topics the chapter defers. None is taught here; later extraction EXTENDs from these hooks.

| Topic | Chapter | Claim / promised treatment |
|---|---|---|
| Word vectors | ch02 (Lecture 2) | Named among the course's "basics first: word vectors, recurrent networks, attention, transformers" (intro 5); exhibition panels on word analogies (king/queen) and GloVe co-occurrence (intro 14) |
| Backpropagation and neural network basics | ch03 (Lecture 3) | Course syllabus; intro 8: Hw2 "covers neural network foundations and calculating tensor derivatives" |
| Recurrent networks and language models | ch04 (Lecture 4) | Intro 5 "recurrent networks"; exhibition panel on sequence-to-sequence models (intro 15) |
| Attention and transformers | ch05 (Lecture 5) | Intro 5 "attention, transformers"; exhibition panel "Is that really all you need" / "Attention" (intro 15); intro 8: Hw3 "coding a transformer from scratch" |
| Pretraining and scaling | ch07 (Lecture 7) | Intro 5 "pretraining"; exhibition panels on BERT and a training-loss-vs-FLOPs curve (intro 16, 18) |
| Post-training | ch08 (Lecture 8) | Intro 5 "post-training" |
| Efficient adaptation | ch09 (Lecture 9) | Intro 5 "efficient adaptation" |
| Agents, tool use and RAG | ch10 (Lecture 10) | Intro 5 "agents"; intro 5 goal 3 "RAG, agentic systems and tool use"; exhibition panels on retrieval and tool use (intro 17, 18) |
| Benchmarks and evaluation | ch11 (Lecture 11) | Intro 5 "LLMs evaluation"; exhibition panels on SQuAD and GLUE (intro 15, 16); intro 8: Hw4 "LLMs evaluation and redteaming" |
| Reasoning | ch12, ch13 (Lectures 12–13) | Intro 5 "reasoning"; exhibition panel "The Thinking LLM" (intro 18) |
| Multilinguality | ch14 (Lecture 14) | Intro 5 "multilinguality"; exhibition panel "mBERT" (intro 16) |
| Interpretability | ch15 (Lecture 15) | Intro 5 "interpretability" |
| Multimodality | ch17 (Lecture 17) | Intro 5 "multimodality" |
| Gradient-based learning and backpropagation | ch03 (Lecture 3) | History 13: the threshold unit "has no slope, so, no gradient-based learning" |
| Dependency parsing | Assignment 2, released with ch03 (Lecture 3) | Named on history 20 ("David Glenn Hays: ACL founder and dependency parsing"); intro 8: Hw2 builds "a small network for an NLP task around dependency parsing" |
| Benchmarks and evaluation | ch11 (Lecture 11) | Cranfield's "benchmark tests for language" (history 18) |
| RNNs and LSTMs | ch04 (Lecture 4) | The basic RNN recurrence and an LSTM encoder–decoder (history 45–46) |
| Attention | ch05 (Lecture 5) | The seq2seq "Bottleneck" (history 46) that attention removes; the exhibition panel "Is that really all you need" (intro 15) |
| Pretraining and scaling of LLMs | ch07 (Lecture 7) | "100+ billion parameter neural language models trained on > 1 trillion tokens" (history 52) |
| Tokenization | ch14 (Lecture 14) | "next token prediction" (history 51) |
| Evaluating LLMs | ch11 (Lecture 11) | The BCG and TTCW studies (history 55–56) |
| Transformers | ch05 (Lecture 5) | Dædalus pp. 130–131 and Fig. 1 |
| Post-training | ch08 (Lecture 8) | Fine-tuning a pretrained model for a task (Dædalus p. 131) |
| Prompting and efficient adaptation | ch09 (Lecture 9) | GPT-3 performing "novel tasks very well with just a prompt" (Dædalus p. 132) |
| Retrieval-augmented models | ch10 (Lecture 10) | "a large supply of text to be consulted at runtime" (Dædalus p. 135) |
| Multimodal foundation models | ch17 (Lecture 17) | "foundation models that also take in other sensory data" (Dædalus p. 135) |
| word2vec and GloVe algorithms | ch02 (Lecture 2) | A1 Part 2: "revisit the class notes and lecture slides for more details on the word2vec and GloVe algorithms" |
| Bias and broader impacts | ch16 (Lecture 16) | A1 Q2.7–Q2.9 |

## Studs

Concepts the chapter uses but does not teach. A stud means "not taught here", not "nothing more to learn".

<!-- Entry shape (numbering runs B01, B02, …):
 ### B01 <Label>
- **Meaning:** one line.
- **Used at:** §3.1 "short quote"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.
  (Book mode, overlay §8: when an earlier chapter's map teaches it, write `Taught in ch02:C05` instead.)
-->

### B01 Artificial intelligence (the field)
- **Meaning:** The field that seeks to build machines with intelligent behaviour; the term appears in the 1955 proposal for the 1956 Dartmouth workshop.
- **Used at:** history 2 "AI / Neural Networks"; Dædalus p. 128 "the coining of the term 'artificial intelligence' in 1956"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B02 Machine translation (the task)
- **Meaning:** Automatically translating text from one human language to another.
- **Used at:** history 4 "Machine Translation: The origin of NLP/Computational Linguistics"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B03 Cryptography and code-breaking
- **Meaning:** Methods for encoding messages so others cannot read them, and for decoding them without the key.
- **Used at:** history 4 "the powerful new mechanized methods in cryptography"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B04 Turing machine
- **Meaning:** The abstract model of a general-purpose computer.
- **Used at:** history 12 "showing it could implement a Turing Machine"; history 23 "less powerful than a universal Turing machine"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B05 Gradient-based learning
- **Meaning:** Training a model by adjusting its parameters in the direction that reduces error, using derivatives.
- **Used at:** history 13 "This function has no slope, so, no gradient-based learning"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter. (Forward: Lecture 3.)

### B06 Vectors and matrix–vector products
- **Meaning:** Lists of numbers, and the product of a weight matrix with an input vector.
- **Used at:** history 13 "𝟏(Wx > θ)"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B07 Context-free grammars and parse trees
- **Meaning:** Grammars of rewrite rules, and the trees that show how a sentence is built from them.
- **Used at:** history 28 "Parse it into a tree or graph data structure (using context-free grammars and beyond)"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B08 First-order logic and λ-calculus notation
- **Meaning:** Logical predicates, connectives and quantifiers, and λ-abstraction for writing functions.
- **Used at:** history 28 "on(ι(λx(apple(x) ∧ red(x))), ι(λy.table(y)))"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B09 Knowledge base
- **Meaning:** Structured facts and relations available to an inference system.
- **Used at:** history 30 "a suitable knowledge base is a prerequisite for making proper inferences"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B10 Conditional probability
- **Meaning:** The probability of an event given another, written P(a\|b).
- **Used at:** history 36 "P(actress\|versatile) = .0019"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B11 Classic statistical models (Naïve Bayes, HMMs, PCFGs, CRFs, MaxEnt, SVMs)
- **Meaning:** The standard probabilistic and machine-learning models of era 3.
- **Used at:** history 35 "Tools: Naïve Bayes classifiers, HMMs, PCFGs, CRFs (CS228!)"; history 37 "MaxEnt, SVMs, Bayes Nets, LDA"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B12 LSTM internals
- **Meaning:** The gated cell structure inside a long short-term memory network.
- **Used at:** history 46 "An LSTM encoder-decoder network [Sutskever et al. 2014]"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter. (Forward: Lecture 4.)

### B13 BLEU (MT evaluation metric)
- **Meaning:** An automatic score of machine translation quality against reference translations.
- **Used at:** history 47 "Edinburgh En-De WMT newstest2013 Cased BLEU"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B14 Entropy (information theory)
- **Meaning:** Shannon's measure of the uncertainty of a probability distribution.
- **Used at:** history 51 "explores character/word-level n-gram models, entropy, text generation"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B15 Model parameters
- **Meaning:** The learned numbers (weights) inside a model; their count measures model size.
- **Used at:** history 52 "100+ billion parameter neural language models"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B16 Tokens
- **Meaning:** The units (words or pieces of words) a language model reads and predicts.
- **Used at:** history 51 "next token prediction"; history 52 "> 1 trillion tokens"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter. (Forward: Lecture 14.)

### B17 Singular value decomposition
- **Meaning:** The factorization of a matrix into U, S, Vᵀ, with singular values on the diagonal of S.
- **Used at:** A1 Part 1 "The SVD process decomposes the co-occurrence matrix A into singular values in the diagonal S matrix and new, shorter word vectors"
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

### B18 Dot product and vector norm
- **Meaning:** p · q = Σᵢ pᵢqᵢ, and ‖p‖, the length of a vector.
- **Used at:** A1 "Cosine Similarity" s = (p · q)/(‖p‖ ‖q‖)
- **Status:** not taught here. Further prerequisites are not assessed in this chapter.

## Prerequisites

Every row reads "From `prerequisite_for` To"; a learner walks backward from To. The prerequisite rows must form
no cycle. A **scope** means the prerequisite is needed only for that part of To.

| From | Predicate | To | Family | Status | Scope | Basis |
|---|---|---|---|---|---|---|
| B01 Artificial intelligence (the field) | `prerequisite_for` | C03 NLP/CL and AI/neural networks as partly separate lineages | prerequisite | inferred | — | history 2 "AI / Neural Networks" |
| C02 Four eras of NLP | `prerequisite_for` | C05 Era 1: Early explorations (1940–1969) | prerequisite | inferred | — | history 2 "1940–1969: Early Explorations" |
| B02 Machine translation (the task) | `prerequisite_for` | C06 Machine translation as code-breaking | prerequisite | inferred | — | history 4 |
| B03 Cryptography and code-breaking | `prerequisite_for` | C06 Machine translation as code-breaking | prerequisite | inferred | Weaver's argument | history 4 "mechanized methods in cryptography" |
| B02 Machine translation (the task) | `prerequisite_for` | C07 Georgetown–IBM demonstration (1954) | prerequisite | inferred | — | history 7 "translate meaningful texts from one language to another" |
| B02 Machine translation (the task) | `prerequisite_for` | C08 MT overconfidence and the ALPAC report (1966) | prerequisite | inferred | — | history 23 "condemned MT field" |
| B01 Artificial intelligence (the field) | `prerequisite_for` | C10 The Dartmouth Summer Research Project (1956) | prerequisite | inferred | — | history 10 "AI Begins" |
| B01 Artificial intelligence (the field) | `prerequisite_for` | C14 Two visions of AI in the 1950s | prerequisite | inferred | — | history 16 |
| C11 Connectionism (neural networks modelled on neurons) | `prerequisite_for` | C12 McCulloch–Pitts threshold unit | prerequisite | inferred | — | history 11–13 |
| B06 Vectors and matrix–vector products | `prerequisite_for` | C12 McCulloch–Pitts threshold unit | prerequisite | inferred | the formula | history 13 "𝟏(Wx > θ)" |
| B05 Gradient-based learning | `prerequisite_for` | C12 McCulloch–Pitts threshold unit | prerequisite | inferred | the "no slope" claim | history 13 |
| B04 Turing machine | `prerequisite_for` | C12 McCulloch–Pitts threshold unit | prerequisite | inferred | the Turing-machine claim | history 12 |
| C12 McCulloch–Pitts threshold unit | `prerequisite_for` | C13 Rosenblatt's perceptron | prerequisite | inferred | — | history 13–14 |
| C15 Information retrieval and Bush's memex | `prerequisite_for` | C16 Benchmark evaluation (Cleverdon's Cranfield tests) | prerequisite | inferred | — | history 18 "document collection, queries, and correct answers" |
| C02 Four eras of NLP | `prerequisite_for` | C18 Era 2: Hand-built symbolic NLP (1970–1992) | prerequisite | inferred | — | history 2 "1970–1992: Hand-built symbolic NLP systems" |
| B07 Context-free grammars and parse trees | `prerequisite_for` | C21 Compositional semantics | prerequisite | inferred | — | history 28 |
| B08 First-order logic and λ-calculus notation | `prerequisite_for` | C21 Compositional semantics | prerequisite | inferred | the formulas | history 28 |
| B09 Knowledge base | `prerequisite_for` | C22 Knowledge-based inference for text understanding (Norvig 1986) | prerequisite | stated | — | history 30 "a suitable knowledge base is a prerequisite for making proper inferences" |
| B01 Artificial intelligence (the field) | `prerequisite_for` | C23 The physical symbol system hypothesis (Newell & Simon) | prerequisite | inferred | — | history 31 |
| C02 Four eras of NLP | `prerequisite_for` | C24 Era 3: Statistical NLP and supervised machine learning (1993–2012) | prerequisite | inferred | — | history 2 "1993–2012: Statistical or Probabilistic NLP" |
| B10 Conditional probability | `prerequisite_for` | C26 The statistical revolution: probabilistic modeling of language | prerequisite | inferred | the spelling-correction calculation (history 36) | history 36 |
| B11 Classic statistical models (Naïve Bayes, HMMs, PCFGs, CRFs, MaxEnt, SVMs) | `prerequisite_for` | C26 The statistical revolution: probabilistic modeling of language | prerequisite | inferred | the "Tools" bullet | history 35 |
| C02 Four eras of NLP | `prerequisite_for` | C28 Era 4a: Neural NLP (2013–2021) | prerequisite | inferred | — | history 2, 38 |
| C23 The physical symbol system hypothesis (Newell & Simon) | `prerequisite_for` | C29 Symbol systems versus their processors | prerequisite | inferred | — | history 40 "We need not design NLP systems as physical symbol systems" |
| C11 Connectionism (neural networks modelled on neurons) | `prerequisite_for` | C29 Symbol systems versus their processors | prerequisite | inferred | "The brain is more like a neural network model" | history 40 |
| C31 Word vectors | `prerequisite_for` | C32 Distributional similarity | prerequisite | inferred | learning word vectors from context | history 44 "How do we learn those word vectors?" |
| C31 Word vectors | `prerequisite_for` | C33 Neural encoder–decoder for translation (seq2seq) | prerequisite | inferred | — | history 45 |
| B06 Vectors and matrix–vector products | `prerequisite_for` | C33 Neural encoder–decoder for translation (seq2seq) | prerequisite | inferred | the recurrence | history 45 |
| B12 LSTM internals | `prerequisite_for` | C33 Neural encoder–decoder for translation (seq2seq) | prerequisite | inferred | understanding slide 46's cells | history 46 |
| B13 BLEU (MT evaluation metric) | `prerequisite_for` | C34 Neural MT overtakes statistical MT (2015–2016) | prerequisite | inferred | the chart | history 47 |
| C33 Neural encoder–decoder for translation (seq2seq) | `prerequisite_for` | C34 Neural MT overtakes statistical MT (2015–2016) | prerequisite | inferred | — | history 45–47 |
| C02 Four eras of NLP | `prerequisite_for` | C35 Era 4b: Large language models (2022–present) | prerequisite | inferred | — | history 2, 48 |
| C37 Scaling up language models | `prerequisite_for` | C35 Era 4b: Large language models (2022–present) | prerequisite | inferred | — | history 52–53 |
| B10 Conditional probability | `prerequisite_for` | C36 Language model (probabilistic next-token prediction) | prerequisite | inferred | the n-gram factorizations | intro 11 |
| B16 Tokens | `prerequisite_for` | C36 Language model (probabilistic next-token prediction) | prerequisite | inferred | — | history 51 "next token prediction" |
| C36 Language model (probabilistic next-token prediction) | `prerequisite_for` | C37 Scaling up language models | prerequisite | inferred | — | history 51–52 |
| B15 Model parameters | `prerequisite_for` | C37 Scaling up language models | prerequisite | inferred | — | history 52 |
| C35 Era 4b: Large language models (2022–present) | `prerequisite_for` | C38 Evidence on what LLMs can and cannot do (2023 studies) | prerequisite | inferred | — | history 55–56 |
| C36 Language model (probabilistic next-token prediction) | `prerequisite_for` | C39 Self-supervised pretraining and large pretrained language models (LPLMs) | prerequisite | inferred | — | Dædalus pp. 129–130 "identifying each next word" |
| B05 Gradient-based learning | `prerequisite_for` | C39 Self-supervised pretraining and large pretrained language models (LPLMs) | prerequisite | inferred | "back-propagation of errors" | Dædalus p. 130 |
| C31 Word vectors | `prerequisite_for` | C40 The transformer and attention | prerequisite | inferred | — | Dædalus p. 130 "These models represent everything via vectors of real numbers" |
| B06 Vectors and matrix–vector products | `prerequisite_for` | C40 The transformer and attention | prerequisite | inferred | — | Dædalus p. 130 |
| C39 Self-supervised pretraining and large pretrained language models (LPLMs) | `prerequisite_for` | C41 Prompting (in-context task specification) | prerequisite | inferred | — | Dædalus pp. 131–132 |
| C39 Self-supervised pretraining and large pretrained language models (LPLMs) | `prerequisite_for` | C42 From NLP pipelines to adapted pretrained models | prerequisite | inferred | — | Dædalus p. 132 |
| C32 Distributional similarity | `prerequisite_for` | C43 Meaning as a network of connections (denotational vs distributional semantics) | prerequisite | inferred | — | Dædalus p. 134 |
| C39 Self-supervised pretraining and large pretrained language models (LPLMs) | `prerequisite_for` | C44 Foundation models | prerequisite | inferred | — | Dædalus p. 135 |
| C32 Distributional similarity | `prerequisite_for` | C45 Co-occurrence matrix (count-based word vectors) | prerequisite | stated | — | A1 Part 1 "Most word vector models start from the following idea: You shall know a word by the company it keeps" |
| B06 Vectors and matrix–vector products | `prerequisite_for` | C45 Co-occurrence matrix (count-based word vectors) | prerequisite | inferred | — | A1 Part 1 |
| C45 Co-occurrence matrix (count-based word vectors) | `prerequisite_for` | C46 Dimensionality reduction by truncated SVD | prerequisite | inferred | — | A1 Part 1 |
| B17 Singular value decomposition | `prerequisite_for` | C46 Dimensionality reduction by truncated SVD | prerequisite | inferred | — | A1 Part 1 |
| C31 Word vectors | `prerequisite_for` | C47 Learned dense word vectors (word2vec, GloVe) | prerequisite | inferred | — | A1 Part 2 |
| C31 Word vectors | `prerequisite_for` | C48 Cosine similarity (and cosine distance) | prerequisite | inferred | — | A1 "Cosine Similarity" |
| B18 Dot product and vector norm | `prerequisite_for` | C48 Cosine similarity (and cosine distance) | prerequisite | inferred | — | A1 "Cosine Similarity" |
| C48 Cosine similarity (and cosine distance) | `prerequisite_for` | C49 What nearest neighbours conflate: polysemy and antonymy | prerequisite | inferred | — | Q2.2–Q2.3 |
| C48 Cosine similarity (and cosine distance) | `prerequisite_for` | C50 Word-vector analogies | prerequisite | stated | — | Q2.4 "The answer to the analogy will have the highest cosine similarity" |
| C31 Word vectors | `prerequisite_for` | C51 Bias in word vectors | prerequisite | inferred | — | Q2.7 |
| C50 Word-vector analogies | `prerequisite_for` | C51 Bias in word vectors | prerequisite | inferred | A1's analogy-based bias probes | Q2.8 |

<!-- Row shape: | B01 Label | `prerequisite_for` | C01 Label | prerequisite | inferred | — | §3.1 "short quote" | -->

### Per-concept review

| Concept | Direct prerequisites and notes |
|---|---|
| C01 Paradigm shift in NLP | None: a framing concept. Its content comes from the era entries. |
| C02 Four eras of NLP | None. |
| C03 NLP/CL and AI/neural networks as partly separate lineages | B01. |
| C04 Language as human societal intelligence | None. |
| C05 Era 1: Early explorations (1940–1969) | C02. |
| C06 Machine translation as code-breaking | B02; B03 (for Weaver's argument). |
| C07 Georgetown–IBM demonstration (1954) | B02. |
| C08 MT overconfidence and the ALPAC report (1966) | B02. C07 illustrates it but is not needed (#61). |
| C09 Early question-answering and dialogue programs | None. |
| C10 The Dartmouth Summer Research Project (1956) | B01. |
| C11 Connectionism (neural networks modelled on neurons) | None. |
| C12 McCulloch–Pitts threshold unit | C11; B06 (formula); B05 ("no slope" claim); B04 (Turing-machine claim). |
| C13 Rosenblatt's perceptron | C12. |
| C14 Two visions of AI in the 1950s | B01. |
| C15 Information retrieval and Bush's memex | None. |
| C16 Benchmark evaluation (Cleverdon's Cranfield tests) | C15. |
| C17 The Association for Computational Linguistics (ACL), founded 1962 | None. |
| C18 Era 2: Hand-built symbolic NLP (1970–1992) | C02. |
| C19 Natural-language-understanding demonstration systems | None. |
| C20 Formal and unification-based grammars | None (named, not explained). |
| C21 Compositional semantics | B07; B08 (the formulas). |
| C22 Knowledge-based inference for text understanding | B09. |
| C23 The physical symbol system hypothesis | B01. |
| C24 Era 3: Statistical NLP and supervised machine learning | C02. |
| C25 The noisy channel model | None. |
| C26 The statistical revolution | B10 (spelling-correction calculation); B11 ("Tools" bullet). |
| C27 Supervised learning from annotated corpora | None: it teaches learning from labelled examples itself (#73). |
| C28 Era 4a: Neural NLP | C02. |
| C29 Symbol systems versus their processors | C23; C11 (the brain-as-network point). |
| C30 Deep learning's breakthrough, first in speech | None. |
| C31 Word vectors | None. |
| C32 Distributional similarity | C31 (for learning vectors from context). |
| C33 Neural encoder–decoder for translation | C31; B06 (recurrence); B12 (slide 46's cells). |
| C34 Neural MT overtakes statistical MT | C33; B13 (the chart). |
| C35 Era 4b: Large language models | C02; C37. |
| C36 Language model | B10 (the factorizations); B16. |
| C37 Scaling up language models | C36; B15. |
| C38 Evidence on what LLMs can and cannot do | C35. |
| C39 Self-supervised pretraining and LPLMs | C36; B05 ("back-propagation of errors"). |
| C40 The transformer and attention | C31; B06. |
| C41 Prompting | C39. |
| C42 From NLP pipelines to adapted pretrained models | C39. |
| C43 Meaning as a network of connections | C32. |
| C44 Foundation models | C39. |
| C45 Co-occurrence matrix | C32; B06. |
| C46 Dimensionality reduction by truncated SVD | C45; B17. |
| C47 Learned dense word vectors | C31. |
| C48 Cosine similarity | C31; B18. |
| C49 Polysemy and antonymy in nearest neighbours | C48. |
| C50 Word-vector analogies | C48. |
| C51 Bias in word vectors | C31; C50 (for A1's analogy-based probes). |

## Predicates

The controlled list is `predicates.md` in this pack. List here only predicates this map ADDED with the
Reviewer's agreement, each with its family and meaning.

## Links

Functional links from the text, each marked **stated** (the author says it) or **inferred** (our reading, with its reason).

| From | Predicate | To | Family | Status | Scope / original wording | Basis |
|---|---|---|---|---|---|---|
| C02 Four eras of NLP | `illustrates` | C01 Paradigm shift in NLP | functional | inferred | the four eras are the lecture's instance of its paradigm-shift framing | intro 2 "paradigm shifts in NLP"; history 2 "Four eras of NLP" |
| C02 Four eras of NLP | `organizes` | C05 Era 1: Early explorations (1940–1969) | functional | inferred | era 1 of the four | history 2 |
| C05 Era 1: Early explorations (1940–1969) | `organizes` | C06 Machine translation as code-breaking | functional | inferred | era 1's MT strand | history 4, 19 |
| C05 Era 1: Early explorations (1940–1969) | `organizes` | C07 Georgetown–IBM demonstration (1954) | functional | inferred | era 1's MT strand | history 6–9, 19 |
| C05 Era 1: Early explorations (1940–1969) | `organizes` | C08 MT overconfidence and the ALPAC report (1966) | functional | inferred | era 1's MT strand | history 19, 23 |
| C07 Georgetown–IBM demonstration (1954) | `illustrates` | C08 MT overconfidence and the ALPAC report (1966) | functional | inferred | the demo's "three to five years" promise is the overconfidence ALPAC answered | history 8, 23 |
| C11 Connectionism (neural networks modelled on neurons) | `organizes` | C12 McCulloch–Pitts threshold unit | functional | stated | "Seminal work in connectionism: McCulloch & Pitts on formal neurons" | history 19 |
| C11 Connectionism (neural networks modelled on neurons) | `organizes` | C13 Rosenblatt's perceptron | functional | stated | "Seminal work in connectionism: Rosenblatt on perceptrons" | history 19 |
| C13 Rosenblatt's perceptron | `illustrates` | C14 Two visions of AI in the 1950s | functional | inferred | Rosenblatt is listed under cybernetics | history 16 |
| C16 Benchmark evaluation (Cleverdon's Cranfield tests) | `assesses` | C15 Information retrieval and Bush's memex | functional | inferred | the Cranfield tests evaluate document retrieval | history 18 |
| C02 Four eras of NLP | `organizes` | C18 Era 2: Hand-built symbolic NLP (1970–1992) | functional | inferred | era 2 of the four | history 2 |
| C18 Era 2: Hand-built symbolic NLP (1970–1992) | `organizes` | C19 Natural-language-understanding demonstration systems (SHRDLU and peers) | functional | inferred | an era-2 strand | history 31 |
| C18 Era 2: Hand-built symbolic NLP (1970–1992) | `organizes` | C20 Formal and unification-based grammars | functional | inferred | an era-2 strand | history 26–27, 31 |
| C18 Era 2: Hand-built symbolic NLP (1970–1992) | `organizes` | C21 Compositional semantics | functional | inferred | an era-2 strand | history 28 |
| C18 Era 2: Hand-built symbolic NLP (1970–1992) | `organizes` | C22 Knowledge-based inference for text understanding (Norvig 1986) | functional | inferred | an era-2 strand | history 29–30 |
| C02 Four eras of NLP | `organizes` | C24 Era 3: Statistical NLP and supervised machine learning (1993–2012) | functional | inferred | era 3 of the four | history 2 |
| C24 Era 3: Statistical NLP and supervised machine learning (1993–2012) | `organizes` | C26 The statistical revolution: probabilistic modeling of language | functional | inferred | an era-3 strand | history 35 |
| C24 Era 3: Statistical NLP and supervised machine learning (1993–2012) | `organizes` | C27 Supervised learning from annotated corpora | functional | inferred | an era-3 strand | history 35; Dædalus p. 129 |
| C26 The statistical revolution: probabilistic modeling of language | `uses` | C25 The noisy channel model (Shannon 1948) | functional | inferred | the field took its probabilistic ideas from speech recognition's noisy-channel work | history 31, 35 |
| C02 Four eras of NLP | `organizes` | C28 Era 4a: Neural NLP (2013–2021) | functional | inferred | era 4a | history 2, 38 |
| C28 Era 4a: Neural NLP (2013–2021) | `organizes` | C30 Deep learning's breakthrough (2009–2012), first in speech | functional | inferred | an era-4a strand | history 41–42 |
| C28 Era 4a: Neural NLP (2013–2021) | `organizes` | C31 Word vectors | functional | inferred | an era-4a strand | history 43; Dædalus p. 129 |
| C28 Era 4a: Neural NLP (2013–2021) | `organizes` | C33 Neural encoder–decoder for translation (seq2seq) | functional | inferred | an era-4a strand | history 45–46 |
| C28 Era 4a: Neural NLP (2013–2021) | `organizes` | C34 Neural MT overtakes statistical MT (2015–2016) | functional | inferred | an era-4a strand | history 47 |
| C29 Symbol systems versus their processors | `contrasts_with` | C23 The physical symbol system hypothesis (Newell & Simon) | functional | stated | "We need not design NLP systems as physical symbol systems" | history 40 |
| C32 Distributional similarity | `method_for` | C31 Word vectors | functional | stated | "Learn vectors via distributional similarity" | history 44 |
| C33 Neural encoder–decoder for translation (seq2seq) | `uses` | C31 Word vectors | functional | inferred | words enter the networks as vectors | history 45–46 |
| C33 Neural encoder–decoder for translation (seq2seq) | `applies_to` | B02 Machine translation (the task) | functional | stated | the slides' examples are translations | history 45–47 |
| C02 Four eras of NLP | `organizes` | C35 Era 4b: Large language models (2022–present) | functional | inferred | era 4b | history 2, 48 |
| C26 The statistical revolution: probabilistic modeling of language | `uses` | C36 Language model (probabilistic next-token prediction) | functional | stated | language models "Used for spelling correction"; the spelling-correction example | history 36, 51 |
| C39 Self-supervised pretraining and large pretrained language models (LPLMs) | `uses` | C40 The transformer and attention | functional | stated | "A common self-supervision objective in a transformer model is to mask out occasional words in a text" | Dædalus p. 130 |
| C39 Self-supervised pretraining and large pretrained language models (LPLMs) | `kind_of` | C44 Foundation models | taxonomic | stated | "LPLMs like BERT… and GPT-3 are early examples of foundation models" | Dædalus p. 135 |
| C41 Prompting (in-context task specification) | `uses` | C39 Self-supervised pretraining and large pretrained language models (LPLMs) | functional | inferred | prompting adapts a pretrained model | Dædalus p. 130 "adapted, via fine-tuning or prompting" |
| C42 From NLP pipelines to adapted pretrained models | `uses` | C39 Self-supervised pretraining and large pretrained language models (LPLMs) | functional | inferred | the replacement is a fine-tuned LPLM | Dædalus p. 132 |
| C45 Co-occurrence matrix (count-based word vectors) | `method_for` | C31 Word vectors | functional | stated | "those derived from co-occurrence matrices" | A1 "Word Vectors" |
| C45 Co-occurrence matrix (count-based word vectors) | `uses` | C32 Distributional similarity | functional | stated | "similar words… will be used in similar contexts" | A1 Part 1 |
| C46 Dimensionality reduction by truncated SVD | `applies_to` | C45 Co-occurrence matrix (count-based word vectors) | functional | stated | "decomposes the co-occurrence matrix A" | A1 Part 1 |
| C47 Learned dense word vectors (word2vec, GloVe) | `kind_of` | C31 Word vectors | taxonomic | stated | "prediction-based word vectors" | A1 Part 2 |
| C49 What nearest neighbours conflate: polysemy and antonymy | `uses` | C48 Cosine similarity (and cosine distance) | functional | stated | "top-10 most similar words (according to cosine similarity)" | Q2.2 |
| C50 Word-vector analogies | `uses` | C48 Cosine similarity (and cosine distance) | functional | stated | "The answer to the analogy will have the highest cosine similarity" | Q2.4 |
| C51 Bias in word vectors | `uses` | C50 Word-vector analogies | functional | stated | "Use the most_similar function to find another pair of analogies that demonstrates some bias" | Q2.8 |

<!-- Row shape: | C02 Label | `has_parameters` | C03 Label | part-whole | stated | — | "short quote" (§3.1) | -->
