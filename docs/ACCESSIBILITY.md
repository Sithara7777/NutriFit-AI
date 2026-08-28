# Accessibility conformance

Target: **WCAG 2.1 Level AA**.

This addresses the Cardiff Met EDGE **ETHICAL** attribute in the assessment
brief: *"accessibility becomes a moral imperative, demanding software that
caters to a diverse range of users, promoting inclusivity by adhering to
established accessibility guidelines."*

---

## What was wrong before

An audit of the frontend found **zero** accessibility affordances:

```
aria-*  0    role=  0    alt=  0    htmlFor  0
scope=  0    <caption>  0    sr-only  0    skip link  none
```

Concretely, that meant: charts were silent to screen readers, form labels were
associated only by nesting, tables had no header semantics, keyboard users had
to tab through the whole navigation on every page, the SPA kept one page title
forever, and several text colours failed contrast.

## What was implemented

| WCAG criterion | Level | Implementation |
|---|---|---|
| **1.1.1** Non-text Content | A | `AccessibleChart` wraps every Recharts figure: `<figure>` + text summary, SVG hidden from the accessibility tree, underlying data exposed as a visually-hidden table |
| **1.3.1** Info and Relationships | A | `<th scope="col">` / `<th scope="row">` on all 4 tables; `<caption>` on each; nav is a `<ul>`; meal lists are `<ul>`/`<li>` |
| **1.4.1** Use of Colour | A | Selected week/slot carries `aria-current` / `aria-pressed`; alerts carry a visually-hidden text prefix ("Error:", "Warning:"); the protein line on dual-axis charts is dashed as well as coloured |
| **1.4.3** Contrast (Minimum) | AA | Body text moved `slate-400/500` → `slate-600/700`; brand from `#2e8b57` (3.9:1) → `#24704a` (6.5:1) and `#1b563a` (9:1) |
| **2.1.1** Keyboard | A | All controls are native `<button>`/`<a>`/`<input>`; horizontally scrolling tables get `tabIndex={0}` + `role="region"` so they can be scrolled by keyboard |
| **2.4.1** Bypass Blocks | A | Skip link to `#main-content`, visible on focus |
| **2.4.2** Page Titled | A | `usePageTitle` hook sets a distinct `<title>` per route |
| **2.4.7** Focus Visible | AA | `focus-visible:ring-2` on every button, link, nav item and input |
| **3.3.1** Error Identification | A | `aria-invalid` + `aria-describedby` link inputs to their error text |
| **3.3.2** Labels or Instructions | A | `Field` injects `id`/`htmlFor` into its child via `cloneElement` — explicit association, not nesting |
| **4.1.2** Name, Role, Value | A | `aria-pressed` on toggles, `aria-current="page"` on nav (via `NavLink`), `aria-label` on landmarks and scroll regions |
| **4.1.3** Status Messages | AA | `role="alert"` (assertive) for errors, `role="status"` (polite) for success/loading |
| **2.3.3** Animation from Interactions | AAA | Spinner respects `motion-reduce:animate-none` |

### Verification

```
aria-label 9   aria-live 3        aria-hidden 10   aria-current 3
aria-pressed 3 aria-describedby 3 aria-invalid 5   role= 9
htmlFor 2      scope= 14          sr-only 14       <caption> 4
focus-visible 6  tabIndex 5       usePageTitle 13
```

`htmlFor` appears twice rather than once per input because the association is
generated centrally in the `Field` component (`useId` + `cloneElement`), so a
single implementation labels every form control in the application.

---

## The chart problem, and why it matters

Recharts renders an SVG of unlabelled `<path>` elements. To a screen reader it
is silence — the user is told nothing at all about their weight trend.

Three options were considered:

1. `aria-label` on the SVG — one sentence, loses all the data.
2. A visible data table beside every chart — accessible, but doubles the page
   length for sighted users.
3. **A visually-hidden data table plus a spoken summary** — chosen.

Option 3 gives screen-reader users the *information* (every data point, in a
properly marked-up table) without changing the visual design. The picture is
decorative; the data is not. This is the distinction WCAG 1.1.1 actually draws.

---

## Known limitations — state these honestly in the report

* **Not tested with a real screen reader.** The markup follows the standard, but
  no NVDA/JAWS/VoiceOver session has been run. Doing so is the obvious next
  step and would strengthen the UAT chapter considerably.
* **No automated audit in CI.** An `axe-core` or Lighthouse check should run on
  each build; currently conformance is verified by inspection.
* **Tooltips are mouse/keyboard-hover only.** Recharts tooltips are not exposed
  to assistive technology; the hidden data table compensates, but the
  interactive experience is not equivalent.
* **No internationalisation.** All copy is English. The Cardiff Met EDGE GLOBAL
  attribute also names *localization and internationalization*; the project
  addresses the cultural-bias half of that (see `data/reference/README.md`) but
  not the language half. Sinhala/Tamil localisation is documented as future work.
* **Colour contrast verified by calculation, not by tooling.** Ratios were
  computed from the palette rather than sampled from rendered pixels.

---

## How to re-check after changing the UI

```bash
cd frontend && npm run build          # must succeed
```

Then, manually:

1. **Tab from the top of the page** — the first stop must be "Skip to main
   content", and every interactive element must show a visible focus ring.
2. **Navigate with the keyboard only** — no mouse. Every action must be
   reachable.
3. **Zoom to 200 %** — no content should be lost or overlap.
4. Run Lighthouse (Chrome DevTools → Lighthouse → Accessibility) and record the
   score in your Testing chapter.
