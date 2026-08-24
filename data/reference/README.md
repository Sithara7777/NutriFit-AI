# Reference data (curated, version-controlled)

Unlike `data/raw/` (third-party downloads, git-ignored) and `data/processed/`
(regenerable output, git-ignored), this folder is **committed to the
repository**. It contains data compiled as part of this project.

---

## `sri_lankan_foods.csv` — 42 items

### Why this exists

The two public datasets the project trains on are overwhelmingly Western. A
keyword audit of the 593-item catalogue built from them found:

| Term | Matches |
|---|---|
| hopper, roti, dhal, sambol, kottu, paratha | **0** |
| curry | 1 |
| naan | 1 |
| biryani | 1 |

A representative sample of what that catalogue recommended: *Churros,
Marshmallow, Potato Salad, Acai Bowl.* For the system's stated target user — a
gym-going adult in Sri Lanka — this is a **culturally biased recommender**. It
would return technically correct macros attached to food the user does not eat,
which is an adherence failure, not merely an aesthetic one.

This directly engages two assessment criteria:

* **Proposal §11.4 "Limited Food Database Coverage"**, which anticipated exactly
  this limitation — this dataset is the design *responding* to it rather than
  only restating it in a Limitations chapter.
* **Cardiff Met EDGE — GLOBAL**: *"Cultural sensitivity guides the design
  process, ensuring the software avoids biases and promotes inclusivity."*

### Coverage

42 items across all four meal slots, chosen to cover the everyday Sri Lankan
diet rather than festival or restaurant-only dishes:

| Slot | Items | Examples |
|---|---|---|
| breakfast | 10 | String hoppers, kiribath, pol roti, egg hoppers, pittu, kola kanda |
| lunch | 10 | Rice & curry (chicken/fish/dhal/beef), lamprais, kottu, yellow rice |
| dinner | 10 | Prawn curry, ambul thiyal, godamba roti, devilled chicken, red rice & dhal |
| snack | 12 | Pol sambol, ulundu vadai, isso vadai, fish cutlet, watalappan, curd & treacle, thambili |

Both vegetarian (dhal curry, vegetable kottu, vegetable rice and curry) and
high-protein options (devilled chicken, beef curry, prawn curry) are included so
the recommender can serve all three fitness goals.

### Provenance and accuracy — read this before citing

Values are **compiled estimates per typical Sri Lankan restaurant/home
portion**, cross-referenced against:

* Department of National Nutrition, Sri Lanka — *Food Composition Tables*
* USDA FoodData Central, for the constituent ingredients (rice, coconut,
  chicken, lentils, wheat flour)
* Standard portion conventions used in Sri Lankan dietetic practice

**These are estimates, not laboratory assays.** A "plate of rice and curry"
varies substantially between households and restaurants. State this openly in
your Dataset Review and Limitations chapters — it is a genuine constraint, and
declaring it is stronger than implying a precision the data does not have.

The measurement basis is **per serving**, matching the Kaggle items and
differing from the USDA rows (per 100 g). The `serving_description` column
records the assumed portion for every item so the assumption is auditable.

### How it is used

`nutrifit.data.load_local_foods()` reads this file, and
`nutrifit.foods.build_catalogue(..., local_foods=...)` merges it. Enabled by
default; disable with `--no-local-foods` on `prepare_data.py`, or
`USE_LOCAL_FOODS = False` in notebook 02, to reproduce the biased baseline for
comparison.

### Extending it

Add rows in the same format. The only hard requirements are `food_item`,
`meal_type` (one of breakfast/lunch/dinner/snack) and the four macros
`calories`, `protein_g`, `carbs_g`, `fat_g`. Adding another regional cuisine
follows the identical pattern — the loader is not Sri-Lanka-specific.
