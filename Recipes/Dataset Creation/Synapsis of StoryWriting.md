Here’s the relevant part of the existing calibration set’s category breakdown:
Category Summary (602 samples total)

General chat — 3.99%

Instruction and Reasoning tuning — 2.33%

Multilingual — 5.98%

Tool use — 16.61%

Code / Programming / Software Engineering / DevOps — 55.81%

Math — 1.99%

Sciences — 2.66%

Medical — 1.33%

Finance — 1.33%

Business — 2.66%

Humanities and Philosophy — 1.33%

Creative Writing, Adventure, Roleplay — 2.16%

General Knowledge & Pop Culture — 0.33%

Specialized skills — 1.33%

Misc — 0.17% 



---

1) What the existing category mix tells us

This calibration set emphasizes code/engineering content (~56%), with smaller slices of other domains. That’s because the goal was to preserve and calibrate code-related capabilities during quantization. 

Each category serves a purpose in ensuring that the quantized model still handles different kinds of prompts:

“General chat” for everyday conversational competence

“Instruction & reasoning” to check reasoning and instruction following

“Multilingual” to ensure language breadth

Domain-specific categories (like tool use, math, sciences) for specialized capabilities

Creative writing & roleplay even in this mainly code-focused set to avoid losing general language generation quality 



---

2) How that translates to writing & storytelling

For a writing-focused calibration set, you want a broader emphasis on creative text generation, narrative structure, voice, style, and genre diversity, while still including just enough other content to maintain general reasoning, instruction, and conversational ability.

Here’s a recommended category mix tailored for writing & storytelling calibration:

Category Role in Calibration Suggested %

Narrative Fiction Key for story generation skills (plot, character, pacing) 30–35%
Dialogue & Character Voice Tests natural spoken voice, subtext, dialogue consistency 15–20%
Creative Nonfiction Essays, memoirs, reflective pieces — broadens stylistic range 10–15%
Genre-specific Prompts (fantasy, sci-fi, mystery, horror, romance, etc.) Ensures versatility across genres 10–15%
Instruction & Writing Advice Prompts about craft, revision strategies, writing tips 5–10%
Roleplay / Improvisational Prompts Unscripted character responses & dynamic storytelling 5–10%
General Chat / Conversational Maintains baseline general capability 5–10%
Multilingual Creative Prompts Ensures quality in other languages 5–10%
Misc / General Knowledge Catch-all to preserve broad understanding 2–5%


Note: Total may exceed 100% if categories overlap; just treat these as target ratios.


---

3) What each category should cover

Narrative Fiction

Short stories

Opening/closing scenes

Plot twists

Multiple POVs

Tone and theme prompts


Dialogue & Character Voice

Write a conversation between characters with hidden motivations

Dialogue in specific dialects or emotional states


Creative Nonfiction

Personal essays

Cultural reflections

Journalistic narrations


Genre-specific Prompts

Spec-com pair tasks (e.g., “continue this fantasy scene”)

Genre-flavored challenges to test structural rules


Instruction & Writing Advice

“Suggest improvements and reasons”

“Rewrite in passive/active voice”


Decision Notes

Keep some multilingual creative examples so the model doesn’t lose quality in other languages (similar to the multilingual slice in the original dataset). 

Preserve a small amount of general chat to ensure everyday language fluency isn’t eroded. 



---

4) Practical calibration dataset strategy

When building your calibration YAML:

1. Define category templates (like in the code-focused dataset) so they can be reused.


2. Populate each category with curated content (e.g., writing prompts with gold standard outputs, actual short stories, dialogue datasets, etc.).


3. Balance samples according to your chosen ratios; you may shuffle as in the original.


4. Include metadata like creative tags, genre, length, and style if useful for analysis later.




---

Summary

To build a writing & storytelling calibration dataset similar in spirit to the software engineering calibration dataset:

Shift emphasis from code to narrative and creative text as the dominant category.

Still include supporting categories like instruction, multilingual, general chat, and miscellaneous to preserve broad language function.

Use the calibration set mix as guidance (e.g., 50–60% focus area) but pivot to your domain (creative writing) proportions.


If you’d like, I can help draft a starter YAML schema (with category definitions and example entries) tailored to writing & storytelling!

