# Paper Writing Guidelines & Constraints

## 1. Writing Style & Tone
- Use concise, objective, conference-style academic English.
- Avoid long, verbose sentences and nested clauses.
- Prefer clear logical flow over decorative or dramatic language.
- Do NOT write like a tutorial, textbook, or explanation document. Assume the reader is a domain expert.
- **BANNED WORDS (Do NOT use these typical AI words):** delve into, revolutionize, paramount, pivotal, crucial, arguably, shed light on, underscore, landscape, realm, testament.

## 2. Structure Rules
- Organize sections by **claims and insights**, not by figures/tables.
- Do NOT create one subsection per figure.
- Each subsection must implicitly answer: "What scientific or engineering conclusion does this part prove?"

## 3. Experiments Section Rules
The Experiments section must strictly follow this logical flow:
1. Forecasting performance and interval reliability (overall metrics).
2. Temporal diagnostic / Limits of marginal calibration (deep dive).
3. Risk-aware decision evaluation (economic impact).
- Baseline comparisons must be integrated into the main performance narrative. Do NOT create a separate subsection solely for baselines.

## 4. Result Presentation
- Always explain:
  - What is being compared.
  - The physical/economic insight behind the comparison.
- Avoid simply transcribing numbers from tables into text.
- Compare relatively (e.g., "reduced by X%") and state the *implication* of that reduction.

## 5. LaTeX & Formatting Output Rules
- Output ONLY valid LaTeX code. Do not wrap the output in markdown code blocks (```latex) unless requested.
- Use `\cite{TODO}` for any literature references.
- Use `\ref{fig:xxx}` or `\ref{tab:xxx}` when referencing figures and tables.
- Use standard LaTeX formatting for bolding (`\textbf{}`) and math environments (`$ $` or `\begin{equation}`).

## 6. Method Section Style
- Explain components in a modular, problem-driven way.
- Avoid unnecessary coding or implementation details (e.g., hyperparameter tuning specifics, unless in the setup section).
- Focus strictly on:
  - What theoretical or physical problem this module solves.
  - The mathematical rationale behind it.

## 7. Terminology Consistency
Strictly adhere to the following terminology without alternating:
- Use "Prediction Intervals (PIs)" (NOT confidence intervals).
- Use "Marginal Calibration" and "Conditional Coverage".
- Use "GAT-Informer" (the base model).
- Use "Risk-Aware" when referring to the decision-making process.
- Use "Quantile Crossing" (NOT overlapping).

## 8. General Constraints
- Target: A short, high-impact conference paper (about 6 pages, IEEE IAS format).
- Keep writing compact, focused, and punchy.
- Avoid unnecessary extensions or side discussions outside the scope of EV grid scheduling.